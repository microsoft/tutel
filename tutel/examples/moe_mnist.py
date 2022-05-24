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

import logging

penv = system.init_data_model_parallel(backend='nccl' if torch.cuda.is_available() else 'gloo')


class Net(nn.Module):
    DATASET_TARGET = datasets.MNIST

    def __init__(self, use_moe):
        super(Net, self).__init__()
        self.use_moe = use_moe

        if self.use_moe:
            self.moe_ffn = moe.moe_layer(
                gate_type = {'type': 'top', 'k': 1, 'capacity_factor': 0, 'gate_noise': 1.0},
                experts = {'type': 'ffn',
                    'count_per_node': 1,
                    'hidden_size_per_expert': 128,
                    'output_dim': 10,
                    'activation_fn': lambda x: self.dropout2(F.relu(x))
                },
                model_dim = 9216,
                seeds = (1, penv.global_rank + 1),
                scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
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

    def forward(self, x, top_k=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        if self.use_moe:
            x = self.moe_ffn(x, top_k=top_k)
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

        for p in model.parameters():
            if not hasattr(p, 'skip_allreduce') and p.grad is not None:
                p.grad = net.simple_all_reduce(p.grad)

        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total_items = int(output.size(0))

        if batch_idx % args.log_interval == 0:
            penv.dist_print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                correct, total_items, 100.0 * correct / total_items))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    correct = {1: 0, 2: 0, 8: 0}

    original_level = logging.root.level
    logging.root.setLevel(logging.INFO)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for k in correct:
                output = model(data, top_k=k)
                pred = output.argmax(dim=1, keepdim=True)
                correct[k] += pred.eq(target.view_as(pred)).sum().item()
    logging.root.setLevel(original_level)

    for k in correct:
        correct[k] *= 100.0 / len(test_loader.dataset)

    penv.dist_print('\nTest set: Validate Accuracy: (Top-1) {:.2f}%, (Top-2) {:.2f}%, (Top-8) {:.2f}%\n'.format(
        correct[1], correct[2], correct[8]
    ))
    return max(correct[1], correct[2], correct[8])


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
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

    if int(torch.os.environ.get('LOCAL_RANK', 0)) == 0:
        dataset1 = Net.DATASET_TARGET('/tmp/data', train=True, download=True,
                           transform=transform)
        dataset2 = Net.DATASET_TARGET('/tmp/data', train=False,
                           transform=transform)
        net.barrier()
    else:
        net.barrier()
        dataset1 = Net.DATASET_TARGET('/tmp/data', train=True, download=False,
                           transform=transform)
        dataset2 = Net.DATASET_TARGET('/tmp/data', train=False,
                           transform=transform)
    net.barrier()

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(use_moe=not args.no_moe).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    penv.dist_print('Model = %s.\nShared parameter items = %d (as replicas), expert parameter items = %d (as local x %d devices).' % (
        model,
        len([x for x in model.parameters() if not hasattr(x, 'skip_allreduce')]),
        len([x for x in model.parameters() if hasattr(x, 'skip_allreduce')]),
        penv.global_size,
    ))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    torch.manual_seed(penv.global_rank + 1)

    peak_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        peak_accuracy = max(peak_accuracy, test(model, device, test_loader))
        scheduler.step()

    penv.dist_print('Peak validation accuracy = {:.2f}%'.format(peak_accuracy))


if __name__ == '__main__':
    main()
