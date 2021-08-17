#!/usr/bin/env python3

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from moe.moe_layer import MOELayer

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-tokens', type=int, default=512)
parser.add_argument('--model-dim', type=int, default=2048)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--num-local-experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--top', type=int, default=1)
args = parser.parse_args()


batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
assert top_value > 0 and top_value <= 2

activation_fn = lambda x: x

device = 'cuda'
torch.manual_seed(1)

if args.dtype == 'float32':
  torch.set_default_dtype(torch.float32)
elif args.dtype == 'float16':
  torch.set_default_dtype(torch.float16)
else:
  raise Exception(f'Unrecognized data type specified: {args.dtype}')

class ExpertModel(torch.nn.Module):
    def __init__(self, model_dim, hidden_size, activation_fn = lambda x: x):
        super().__init__()
        self.fc1 = torch.nn.Linear(model_dim, hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(hidden_size, model_dim, bias=True)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gate_type = 'Top1Gate' if top_value == 1 else 'Top2Gate'
        # self._moe_layer = MOELayer(gate_type, model_dim, external_experts=[ExpertModel(model_dim, hidden_size, activation_fn) for i in range(num_local_experts)]).to(device)
        self._moe_layer = MOELayer(gate_type, model_dim, builtin_experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_size, 'activation_fn': activation_fn}, allow_approximation=True).to(device)

    def forward(self, input):
        result = self._moe_layer(input)

        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)

for i in range(10):
  torch.cuda.synchronize()
  t_start = time.time()
  optimizer.zero_grad()

  output = model(x)
  loss = F.nll_loss(output, y)
  loss.backward()
  optimizer.step()

  torch.cuda.synchronize()
  t_stop = time.time()
  print(f'STEP-{i}: DONE, loss = {loss.data}, step_time = {t_stop - t_start} sec.')
