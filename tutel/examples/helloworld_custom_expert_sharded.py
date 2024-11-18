#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import argparse

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
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--l_aux_wt', type=float, default=0.0)
parser.add_argument('--a2a_ffn_overlap_degree', type=int, default=1)
parser.add_argument('--allreduce_degree', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--use_2dh', default=False, action='store_true')
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--capacity_factor', type=float, default=1.0)  # 0.0 for dMoE (dropless-MoE), negative for no-padded capacity.

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


class CustomExpertDemo(torch.nn.Module):

    def _create_sharded_param(self, *full_shape, **kwargs):
        full_shape = torch.Size(full_shape)
        sharded_shape = (full_shape.numel() + self.sharded_count - 1) // self.sharded_count
        return torch.nn.Parameter(torch.empty(sharded_shape, **kwargs)), full_shape

    def _get_gathered_param(self, param, full_shape, parent_group):
        sharded_group = net.create_groups_from_world(group_count=-self.sharded_count, parent_group=parent_group).model_group
        return net.zero_gather(param, group=sharded_group).view(-1).narrow(0, 0, full_shape.numel()).view(full_shape)

    def __init__(self, model_dim, local_experts, sharded_count, my_config):
        super().__init__()
        self.sharded_count = sharded_count
        self.W, self.W_full_shape = self._create_sharded_param(local_experts, model_dim, model_dim)
        self.my_activation = torch.nn.functional.relu if my_config == 'relu' else None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
          self.W.normal_(0, 0.001)

    def forward(self, x, ctx):
        W_full = self._get_gathered_param(self.W, self.W_full_shape, ctx.group)
        y = torch.matmul(x, W_full)
        if self.my_activation is not None:
          y = self.my_activation(y)
        return y


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._moe_layer = tutel_moe.moe_layer(
            gate_type = {'type': 'top', 'k': top_value, 'fp32_gate': args.fp32_gate, 'capacity_factor': args.capacity_factor},
            experts = {'type': 'custom', 'module': CustomExpertDemo, 'count_per_node': num_local_experts, 'my_config': None},
            model_dim = model_dim,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds = (1, dist_rank + 1, 1),
            a2a_ffn_overlap_degree = a2a_ffn_overlap_degree,
            use_2dh=args.use_2dh,
        )

        # Summary of different parameter types: gate, local_experts
        local_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='local_experts')])
        shared_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='gate')])
        dist_print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_count, shared_count))

    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel().to(device)
dist_print(model)

if args.checkpoint_path:
    checkpoint_path = system.apply_rank_size_from_pattern(args.checkpoint_path, rank=parallel_env.global_rank, size=parallel_env.global_size)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print('Checkpoint not loaded: file `%s` is not found. Will train the model from start.' % checkpoint_path)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

torch.manual_seed(0)
x = torch.tensor(torch.randn([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach().numpy(), dtype=torch.get_default_dtype(), requires_grad=False, device=device)
y = torch.LongTensor(batch_size).random_(1).to(device)

tuples = (dist_world_size, args.dtype, model_dim, batch_size * num_tokens, num_local_experts, top_value, a2a_ffn_overlap_degree, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, device = `%s`' % tuples)

average_time, num_steps = 0, args.num_steps

if args.allreduce_degree == -1:
    params_for_all_reduce = []
else:
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

if args.checkpoint_path:
    torch.save(model.state_dict(), checkpoint_path)
