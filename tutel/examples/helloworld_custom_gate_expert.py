#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import argparse
import logging

from tutel import system
from tutel import moe as tutel_moe
from tutel import net

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_tokens', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--l_aux_wt', type=float, default=0.0)
parser.add_argument('--a2a_ffn_overlap_degree', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--eval', default=False, action='store_true')

args = parser.parse_args()

parallel_env = system.init_data_model_parallel(backend='nccl' if args.device == 'cuda' else 'gloo')
dist_rank, dist_world_size, dist_print = parallel_env.global_rank, parallel_env.global_size, parallel_env.dist_print
args.local_rank = parallel_env.local_device.index

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
num_local_experts = args.num_local_experts
top_value = args.top
a2a_ffn_overlap_degree = args.a2a_ffn_overlap_degree
device = parallel_env.local_device

if args.dtype in ('float32', 'float64', 'float16', 'bfloat16'):
    torch.set_default_dtype(eval(f'torch.{args.dtype}'))
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)


class CustomGate(torch.nn.Module):
    def __init__(self, **custom_options):
        super().__init__()
        for key in custom_options:
            logging.warning(f'Receive an option for CustomGate: `{key}` = `{custom_options[key]}`')
            if key == 'top_k':
                self.top_k = custom_options[key]
            elif key in ('model_dim', 'num_global_experts'):
                setattr(self, key, custom_options[key])
            else:
                raise Exception(f'Receive an option key `{key}` for CustomGate, but it is not handled')

        self.my_wg = torch.nn.Linear(self.model_dim, self.num_global_experts, bias=False)

    def forward(self, x):
        return self.my_wg(x)

class CustomExpert(torch.nn.Module):
    def __init__(self, **custom_options):
        super().__init__()
        for key in custom_options:
            logging.warning(f'Receive an option for CustomExpert: `{key}` = `{custom_options[key]}`')
            if key in ('model_dim', 'num_experts_per_device', 'sharded_count'):
                setattr(self, key, custom_options[key])
            elif key == 'my_activ_type':
                self.activ_type = custom_options[key]
            else:
                raise Exception(f'Receive an option key `{key}` for CustomExpert, but it is not handled')

        self.W = torch.nn.Parameter(torch.empty(self.num_experts_per_device, model_dim, model_dim))
        self.my_activation = torch.nn.functional.relu if self.activ_type == 'relu' else None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
          self.W.normal_(0, 0.001)

    def extra_repr(self):
        return '..'

    def forward(self, x, ctx):
        if ctx.sharded_count > 1:
          raise Exception("`sharded_count > 1` is not implemented within this expert, Model parallel is disabled.")
        y = torch.matmul(x, self.W)
        if self.my_activation is not None:
          y = self.my_activation(y)
        return y


class FullyCustomExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._moe_layer = tutel_moe.moe_layer(
            gate_type = {'type': 'custom', 'module': CustomGate, 'top_k': top_value},
            experts = {'type': 'custom', 'module': CustomExpert, 'num_experts_per_device': num_local_experts, 'my_activ_type': 'relu'},
            model_dim = model_dim,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds = (1, dist_rank + 1, 1),
            a2a_ffn_overlap_degree = a2a_ffn_overlap_degree,
        )

    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result


model = FullyCustomExampleModel().to(device)
dist_print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

torch.manual_seed(0)
x = torch.tensor(torch.randn([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach().numpy(), dtype=torch.get_default_dtype(), requires_grad=False, device=device)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (dist_world_size, args.dtype, model_dim, batch_size * num_tokens, num_local_experts, top_value, a2a_ffn_overlap_degree, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, device = `%s`' % tuples)

average_time, num_steps = 0, args.num_steps
params_for_all_reduce = [p for p in model.parameters() if not hasattr(p, 'skip_allreduce') and getattr(p, 'requires_grad', False)]

for i in range(num_steps):
    t_start = system.record_time()

    if not args.eval:
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        if args.l_aux_wt:
            loss += args.l_aux_wt * model._moe_layer.l_aux
        loss.backward()
        if dist_world_size > 1:
            for p in params_for_all_reduce:
                p.grad /= dist_world_size
                p.grad = net.simple_all_reduce(p.grad)
        optimizer.step()
    else:
        with torch.no_grad():
            output = model(x)
            loss = F.nll_loss(output, y)

    t_stop = system.record_time()

    num_global_experts = tutel_moe.moe_layer.global_expert_count(num_local_experts, group=system.get_local_session().model_group)
    mm_ceof, cap_ceof = 1 if args.eval else 3, min(args.top, num_global_experts)
    dist_print('STEP-%s: loss = %.5f, step_time = %.6f sec.' % (i, float(loss.data), t_stop - t_start))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
dist_print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
