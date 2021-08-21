#!/usr/bin/env python3

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-tokens', type=int, default=512)
parser.add_argument('--model-dim', type=int, default=2048)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--num-local-experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--fp32-gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
args = parser.parse_args()

try:
    if dist.is_available():
        dist.init_process_group('nccl')
    dist_rank = dist.get_rank()
except:
    dist_rank = 0

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
local_rank = args.local_rank

assert top_value in (1, 2), "Only support top_value = 1, 2 in this version."

activation_fn = lambda x: x
device = torch.device('cuda', args.local_rank)
torch.manual_seed(dist_rank + 1)

if args.dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif args.dtype == 'float16':
    torch.set_default_dtype(torch.float16)
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)

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

        from baseline_moe.moe_layer import MOELayer
        self._moe_layer = MOELayer(gate_type, model_dim, external_experts=[ExpertModel(model_dim, hidden_size, activation_fn) for i in range(num_local_experts)], fp32_gate=args.fp32_gate).to(device)

        # Distinguish different parameter types: gate, local_experts
        local_experts_param_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='local_experts')])
        shared_gate_param_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='gate')])

        print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_experts_param_count, shared_gate_param_count))

    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (args.dtype, args.fp32_gate, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value)
print('[Benchmark] dtype = %s, fp32-gate = %s, model-dim = %s, hidden-size = %s, samples = %s, num-local-experts = %s, topK = %s' % tuples)

average_time, num_steps = 0, 20

for i in range(num_steps):
    torch.cuda.synchronize()
    t_start = time.time()
    optimizer.zero_grad()

    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t_stop = time.time()
    print('STEP-%s: DONE, loss = %s, step_time = %s sec.' % (i, float(loss.data), t_stop - t_start))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
