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

from fmoe import FMoE
from fmoe import FMoELinear
from fmoe import GShardGate

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_tokens', type=int, default=1024)
parser.add_argument('--model_dim', type=int, default=2048)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
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

    
class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, **kwargs)
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank, gate=GShardGate
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
    
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._moe_layer = FMoETransformerMLP(
            num_expert = num_local_experts,
            d_model = model_dim,
            d_hidden = hidden_size,
            top_k = top_value,
            activation = lambda x: F.relu(x),
            expert_rank = dist_rank
        ).to(device)

        for name, param in self._moe_layer.named_parameters():
            if '.experts.' in name:
                setattr(param, 'skip_allreduce', True)

        # Distinguish different parameter types: gate, local_experts
        local_count = sum([torch.numel(param) for name, param in self._moe_layer.named_parameters() if 'expert' in name])
        shared_count = sum([torch.numel(param) for name, param in self._moe_layer.named_parameters() if 'gate' in name])
        dist_print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_count, shared_count))

    def forward(self, input):
        result = self._moe_layer(input)
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
