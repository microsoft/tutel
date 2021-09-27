#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse

from tutel import moe as tutel_moe

assert torch.__version__ >= '1.8.0', "DDP-based MoE requires Pytorch >= 1.8.0"

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_tokens', type=int, default=1024)
parser.add_argument('--model_dim', type=int, default=2048)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

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

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
local_rank = args.local_rank


device = torch.device('cuda', args.local_rank)

if args.dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif args.dtype == 'float16':
    torch.set_default_dtype(torch.float16)
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._ddp_params_and_buffers_to_ignore = list()

        self._moe_layer = tutel_moe.moe_layer(
            gate_type = 'Top%dGate' % top_value,
            model_dim = model_dim,
            experts = {'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_size, 'activation_fn': lambda x: F.relu(x)},
            fp32_gate = args.fp32_gate,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds = (1, dist_rank + 1),
        ).to(device)

        # Distinguish different parameter types: gate, local_experts
        local_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='local_experts')])
        shared_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='gate')])
        dist_print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_count, shared_count))

    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)


model = ExampleModel()

for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)
if torch.distributed.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

dist_print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (dist_world_size, args.dtype, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, device = `%s`' % tuples)

average_time, num_steps = 0, 100

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
    dist_print('STEP-%s: DONE, loss = %s, step_time = %s sec.' % (i, float(loss.data), t_stop - t_start))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
dist_print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
