#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse
import deepspeed

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_tokens', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=2048)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--use_tutel', default=False, action='store_true')
parser.add_argument('--num_steps', type=int, default=100)
args = parser.parse_args()

try:
    if dist.is_available():
        dist.init_process_group('nccl')
    dist_rank = dist.get_rank()
    dist_world_size = dist.get_world_size()

    def dist_print(*args):
        if dist_rank == 0:
            print(*args)
except:
    dist_rank = 0
    dist_world_size = 1
    dist_print = print

args.local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get('LOCAL_RANK', 0))

torch.cuda.set_device(args.local_rank)

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
local_rank = args.local_rank

if args.top != 1:
    args.use_tutel = False

device = torch.device('cuda', args.local_rank)

if args.dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif args.dtype == 'float64':
    torch.set_default_dtype(torch.float64)
elif args.dtype == 'float16':
    torch.set_default_dtype(torch.float16)
elif args.dtype == 'bfloat16':
    torch.set_default_dtype(torch.bfloat16)
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)

assert deepspeed.version == '0.5.6'

torch.manual_seed(0)
deepspeed.init_distributed()
deepspeed.utils.groups.initialize(ep_size=dist_world_size)

class ExpertModel(torch.nn.Module):
    def __init__(self, model_dim, hidden_size, activation_fn):
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

        self._moe_layer = deepspeed.moe.layer.MoE(
                hidden_size = hidden_size,
                expert = ExpertModel(model_dim, hidden_size, lambda x: F.relu(x)),
                num_experts = num_local_experts * dist_world_size,
                k = top_value,
                use_tutel = args.use_tutel
        )

        for name, param in self._moe_layer.named_parameters():
            if '.experts.' in name:
                setattr(param, 'skip_allreduce', True)

        # Summary of different parameter types: gate, local_experts
        local_count = sum([torch.numel(param) for name, param in self._moe_layer.named_parameters() if '.experts.' in name])
        shared_count = sum([torch.numel(param) for name, param in self._moe_layer.named_parameters() if '.gate.' in name])
        dist_print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_count, shared_count))

    def forward(self, input):
        result, _, _ = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel().to(device)
dist_print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

torch.manual_seed(0)
x = torch.tensor(torch.randn([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach().numpy(), dtype=torch.get_default_dtype(), requires_grad=False, device=device)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (dist_world_size, args.dtype, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, device = `%s`' % tuples)
logging.info('Tutel optimized Deepspeed MoE Top-%s = %s' % (top_value, args.use_tutel))

average_time, num_steps = 0, args.num_steps

params_for_all_reduce = [p for p in model.parameters() if not hasattr(p, 'skip_allreduce') and getattr(p, 'requires_grad', False)]

for i in range(num_steps):

    torch.cuda.synchronize()
    t_start = time.time()

    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    if dist_world_size > 1:
        for p in params_for_all_reduce:
            p.grad /= dist_world_size
            dist.all_reduce(p.grad)
    optimizer.step()

    torch.cuda.synchronize()
    t_stop = time.time()

    num_global_experts = num_local_experts * dist_world_size
    args.top = min(args.top, num_global_experts)
    tflops = (batch_size * num_tokens * model_dim * hidden_size) * 4 * args.top * 3 * 1e-12 / (t_stop - t_start)
    dist_print('STEP-%s: loss = %.5f, step_time = %.6f sec, perf = %.2f tflops.' % (i, float(loss.data), t_stop - t_start, tflops))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
dist_print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
