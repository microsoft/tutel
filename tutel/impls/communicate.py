# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import time
import torch
from torch import Tensor
import torch.distributed as dist

from .jit_compiler import tutel_custom_kernel

def get_world_size(group):
    try:
        return dist.get_world_size(group)
    except:
        return 1

def get_world_rank(group):
    try:
        return dist.get_rank(group)
    except:
        return 0

class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        if not hasattr(AllToAll, '__prepared__'):
            AllToAll.__prepared__ = True
            if not hasattr(dist, 'all_to_all_single') and (AllToAll.a2a_type & 1) == 1:
                AllToAll.a2a_type ^= 3
            if (AllToAll.a2a_type & 2) == 2:
                host_unique_id = torch.zeros([256], dtype=torch.int32).cpu()
                if get_world_rank(group) == 0:
                    tutel_custom_kernel.external_all2all(host_unique_id, 0)
                host_unique_id = host_unique_id.to(input.device)
                dist.broadcast(host_unique_id, 0, group, async_op=True).wait()
                tutel_custom_kernel.external_all2all(host_unique_id.cpu(), 1)

        ctx.group = group
        ctx.world_size = get_world_size(group)
        if ctx.world_size <= 1 or AllToAll.a2a_type == 0:
            return input
        input = input.contiguous()
        if (AllToAll.a2a_type & 8) == 8:
            torch.cuda.synchronize(input.device)
            t_start = time.time()
        if AllToAll.a2a_type == 1:
          output = torch.empty_like(input)
          dist.all_to_all_single(output, input, group=group)
        else:
          output = tutel_custom_kernel.external_all2all(input, -1)
        if (AllToAll.a2a_type & 8) == 8:
            torch.cuda.synchronize(input.device)
            t_stop = time.time()
            if get_world_rank(group) == 0:
                print('[INFO] AllToAll on message size (%d x %s) costs %g sec.' % (torch.numel(input), input.dtype, t_stop - t_start))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return (None, AllToAll.apply(ctx.group, grad_output))

