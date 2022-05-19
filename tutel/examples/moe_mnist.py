#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from tutel import system
from tutel import moe
from tutel import net

penv = system.init_data_model_parallel(backend='nccl' if torch.cuda.is_available() else 'gloo')


class Net(nn.Module):
    def __init__(self, use_moe):
        super(Net, self).__init__()
        self.use_moe = use_moe

        if self.use_moe:
            self.moe_ffn = moe.moe_layer(
                gate_type = {'type': 'top', 'k': 2, 'capacity_factor': -2.5},
                experts = {'type': 'ffn',
                    'count_per_node': 1,
                    'hidden_size_per_expert': 128,
                    'output_dim': 10,
                    'activation_fn': lambda x: self.dropout2(F.relu(x))
                },
                model_dim = 9216,
                seeds = (1, penv.global_rank + 1),
                scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
                a2a_ffn_overlap_degree = 1,
            )
        else:
            torch.manual_seed(1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        torch.manual_seed(1)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        if self.use_moe:
            x = self.moe_ffn(x)
        else:
            x = self.fc1(x)
            x = self.dropout2(F.relu(x))
            x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            penv.dist_print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    penv.dist_print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-moe', action='store_true', default=False,
                        help='if disabling moe layer and using ffn layer instead')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = penv.local_device
    torch.manual_seed(1)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(use_moe=not args.no_moe).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    penv.dist_print('Model = %s.\nShared parameter items = (%d as replicas), expert parameter items = (%d local x %d devices).' % (
        model,
        len([x for x in model.parameters() if not hasattr(x, 'skip_allreduce')]),
        len([x for x in model.parameters() if hasattr(x, 'skip_allreduce')]),
        penv.global_size,
    ))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        for p in model.parameters():
            if not hasattr(p, 'skip_allreduce'):
                p.grad = net.simple_all_reduce(p.grad)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
