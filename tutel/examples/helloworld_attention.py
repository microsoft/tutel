#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Recommend to initialize NUMA status at the most program begining (before any other imports)
from tutel import system_init
system_init.init_affinity_at_program_beginning()

import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse

from tutel import moe as tutel_moe

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_tokens', type=int, default=1024)
parser.add_argument('--model_dim', type=int, default=2048)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--l_aux_wt', type=float, default=0.0)
args = parser.parse_args()

if args.local_rank < 0:
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

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
elif args.dtype == 'bfloat16':
    torch.set_default_dtype(torch.bfloat16)
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def multihead_attention(input):
           ## input.shape == [num_local_heads, batched_tokens, model_dim * (Q, K, V)]
           x = input.view(-1, 3).permute(1, 0).view([3, input.size(0), args.batch_size, args.num_tokens, input.size(2) // 3]).contiguous()
           ## x.shape == [(Q, K, V), num_local_heads, batch_size, num_local_tokens, model_dim]
           att_scores = torch.matmul(x[0, :], x[1, :].permute(0, 1, 3, 2)) / model_dim
           att_props = F.softmax(att_scores.view(-1, att_scores.size(2) * att_scores.size(3)), dim=1).view(*att_scores.shape)
           att_output = torch.matmul(att_props, x[2, :])
           result = att_output
           return result.view(x.size(1), x.size(2) * x.size(3), x.size(4))

        self._moe_attention = tutel_moe.moe_layer(
            gate_type = {'type': 'top', 'k': top_value, 'fp32_gate': args.fp32_gate},
            experts = {'type': 'attention', 'num_heads_per_node': 2, 'attention_fn': multihead_attention},
            model_dim = model_dim,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds = (1, dist_rank + 1, 1),
        ).to(device)


    def forward(self, input):
        result = self._moe_attention(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel()
dist_print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (dist_world_size, args.dtype, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, device = `%s`' % tuples)

average_time, num_steps = 0, 100

params_for_all_reduce = [p for p in model.parameters() if not hasattr(p, 'skip_allreduce') and getattr(p, 'requires_grad', False)]

for i in range(num_steps):

    torch.cuda.synchronize()
    t_start = time.time()
    optimizer.zero_grad()

    output = model(x)
    loss = F.nll_loss(output, y)
    if args.l_aux_wt:
        loss += args.l_aux_wt * model._moe_attention.l_aux
    loss.backward()
    if dist_world_size > 1:
        for p in params_for_all_reduce:
            p.grad /= dist_world_size
            dist.all_reduce(p.grad)
    optimizer.step()

    torch.cuda.synchronize()
    t_stop = time.time()
    dist_print('STEP-%s: DONE, loss = %s, step_time = %s sec.' % (i, float(loss.data), t_stop - t_start))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
dist_print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
