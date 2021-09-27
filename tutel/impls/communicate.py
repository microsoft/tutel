# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import os
import re
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

def set_numa_affinity(group_rank):
    try:
        nodes = sorted([int(x[4:]) for x in os.listdir('/sys/devices/system/node') if re.match('node[0-9]+', x)])
        cpus = [sorted([int(x[3:]) for x in os.listdir('/sys/devices/system/node/node%d' % node_id) if re.match('cpu[0-9]+', x)]) for node_id in nodes]
        sel_node = group_rank % len(nodes)
        os.sched_setaffinity(0, cpus[sel_node])
    except Exception as ex:
        if group_rank == 0:
            print('[WARN] Failed to set NUMA status: %s' % ex)

class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        if not hasattr(AllToAll, '__prepared__'):
            AllToAll.__prepared__ = True
            if not hasattr(dist, 'all_to_all_single') and (AllToAll.a2a_type & 1) == 1:
                AllToAll.a2a_type ^= 3
            if (AllToAll.a2a_type & 2) == 2 and get_world_size(group) > 1:
                host_unique_id = torch.zeros([256], dtype=torch.int32).cpu()
                if get_world_rank(group) == 0:
                    tutel_custom_kernel.external_all2all(host_unique_id, 0)
                host_unique_id = host_unique_id.to(input.device)
                dist.broadcast(host_unique_id, 0, group, async_op=True).wait()
                tutel_custom_kernel.external_all2all(host_unique_id.cpu(), 1)
            if int(os.environ.get('AUTO_NUMA', '1')) != 0:
                set_numa_affinity(get_world_rank(group))

        ctx.group = group
        ctx.world_size = get_world_size(group)
        if ctx.world_size <= 1 or AllToAll.a2a_type == 0:
            return input
        input = input.contiguous()
        if (AllToAll.a2a_type & 8) == 8:
            torch.cuda.synchronize(input.device)
            t_start = time.time()
        if (AllToAll.a2a_type & 1) == 1:
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

# A2A_TYPE: 0 for skip AllToAll, 1 for standard Pytorch AllToAll, 9 for standard Pytorch AllToAll with Timing
AllToAll.a2a_type = int(os.environ.get('A2A_TYPE', '1'))
