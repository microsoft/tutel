# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import os
import re
import time
import torch
import logging 
from torch import Tensor
import torch.distributed as dist

from .jit_compiler import tutel_custom_kernel

def get_world_size(group=None):
    try:
        return dist.get_world_size(group)
    except:
        return 1

def get_world_rank(group=None):
    try:
        return dist.get_rank(group)
    except:
        return 0


TUTEL_GROUPING_CACHE = {}

def create_groups_from_world(group_count, include_init=None):
    backend = TUTEL_GROUPING_CACHE.get('', include_init)
    if include_init:
        assert backend == include_init, "Only 1 backend type is allowed, get: %s v.s. %s" % (backend, include_init)
        TUTEL_GROUPING_CACHE[''] = backend

    if group_count in TUTEL_GROUPING_CACHE:
        return TUTEL_GROUPING_CACHE[group_count]

    try:
      if ('LOCAL_RANK' not in os.environ) and ('OMPI_COMM_WORLD_SIZE' in os.environ):
          if include_init:
              dist.init_process_group(backend=backend,
                  init_method='tcp://%s:%s' % (os.environ['MASTER_ADDR'], os.environ.get('MASTER_PORT', '23456')),
                  rank=int(os.environ['OMPI_COMM_WORLD_RANK']), world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']))
          dist_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
      else:
          if include_init:
              dist.init_process_group(backend=backend)
          dist_local_rank = min(int(os.environ.get('LOCAL_RANK', 0)), torch.cuda.device_count() - 1)
      glob_world_size, glob_world_rank = dist.get_world_size(), dist.get_rank()
      is_distributed = True

      def dist_print(*args):
          if glob_world_rank == 0:
              print(*args)
    except ValueError:
        glob_world_size, glob_world_rank, dist_local_rank = 1, 0, 0
        is_distributed = False
        dist_print = print

    assert glob_world_size % group_count == 0, f"Expected to evenly divide devices into {group_count} groups, while the world size of current sesion is {glob_world_size}."

    dist_group_size = group_count
    dist_world_size = glob_world_size // dist_group_size
    dist_world_rank = glob_world_rank % dist_world_size
    dist_group_rank = glob_world_rank // dist_world_size

    if is_distributed:
        global_group = model_group = data_group = dist.group.WORLD

        if dist_group_size != glob_world_size:
            groups, inner_ranks = [], []
            for gr in range(dist_group_size):
                group_ranks = [x for x in range(gr * dist_world_size, (gr + 1) * dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                inner_ranks += [group_ranks]
            model_group = groups[dist_group_rank]

        if dist_world_size != glob_world_size:
            groups, outer_ranks = [], []
            for gr in range(dist_world_size):
                group_ranks = [x for x in range(gr, dist_world_size * dist_group_size, dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                outer_ranks += [group_ranks]
            data_group = groups[dist_world_rank]
    else:
        model_group, data_group, global_group = None, None, None

    class ParallelPropStorage:
        pass

    result = ParallelPropStorage()
    result.global_size = glob_world_size
    result.global_rank = glob_world_rank
    result.group_count = dist_group_size
    result.data_rank = dist_group_rank
    result.model_rank = dist_world_rank

    if backend == 'nccl':
        result.local_device = torch.device('cuda', dist_local_rank)
        torch.cuda.set_device(result.local_device)
    elif backend == 'gloo':
        result.local_device = torch.device('cpu')
    elif backend is None:
        result.local_device = None
    else:
        raise Exception('Unsupported backend type: %s' % backend)

    result.data_group = data_group
    result.model_group = model_group
    result.global_group = global_group

    result.is_distributed = is_distributed
    result.dist_print = dist_print

    TUTEL_GROUPING_CACHE[group_count] = result
    return result


class AllToAllStatus:
    initialized = False
    gather_tensor_shape = None
    scatter_tensor_shape = None
    num_split = 0
    split_dim = 0
    algo = os.environ.get('TUTEL_ALLTOALL_ALGO', '').upper()

    @staticmethod
    def init(group: dist.ProcessGroup, num_split: int, split_dim: int, gather_tensor_ref: Tensor) -> None:
        world_size = get_world_size(group)
        if world_size <= 1:
            return

        AllToAllStatus.num_split = num_split
        AllToAllStatus.split_dim = split_dim

        # Initialize NCCL
        if not AllToAllStatus.initialized:
            world_rank = get_world_rank(group)
            nccl_unique_id_size = tutel_custom_kernel.get_nccl_unique_id_size()
            nccl_unique_id = torch.zeros([nccl_unique_id_size], dtype=torch.int8).cpu()
            if world_rank == 0:
                tutel_custom_kernel.get_nccl_unique_id(nccl_unique_id)
            nccl_unique_id = nccl_unique_id.to(gather_tensor_ref.device)
            dist.broadcast(nccl_unique_id, 0, group)
            tutel_custom_kernel.init_nccl(
                nccl_unique_id.cpu(),
                world_size,
                world_rank,
                AllToAllStatus.num_split)
            AllToAllStatus.initialized = True


def all_to_all_single(input, group=None):
    world_size = get_world_size(group)
    if world_size <= 1:
        return input
    input = input.contiguous()
    output = torch.empty_like(input)
    if AllToAllStatus.algo:
        AllToAllStatus.init(group, 1, -1, input)
        tutel_custom_kernel.all_to_all_async(output, input, AllToAllStatus.algo)
    else:
        dist.all_to_all_single(output, input, group=group)
    return output

class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        ctx.group = group
        return all_to_all_single(input, group)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return (None, AllToAll.apply(ctx.group, grad_output))

class CurrentStreamRelease(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized:
            return input
        ctx.idx = idx
        input = input.contiguous()
        return tutel_custom_kernel.current_stream_release(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return (grad_output, None)
        return (tutel_custom_kernel.current_stream_acquire(grad_output, ctx.idx), None)

class CurrentStreamAcquire(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized:
            return input
        ctx.idx = idx
        return tutel_custom_kernel.current_stream_acquire(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return (grad_output, None)
        grad_output = grad_output.contiguous()
        return (tutel_custom_kernel.current_stream_release(grad_output, ctx.idx), None)

class AllToAllScatterAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tuple[Tensor]:
        if not AllToAllStatus.initialized:
            return (input,)
        ctx.input_shape = input.shape
        output_shape = torch.Size([
            x if i != AllToAllStatus.split_dim else x // AllToAllStatus.num_split
            for i, x in enumerate(ctx.input_shape)
        ])
        ctx.num_slices_per_split = ctx.input_shape[:AllToAllStatus.split_dim].numel()
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(input, output_shape, ctx.num_slices_per_split, False))

    @staticmethod
    def backward(ctx: Any, *grad_output) -> Tensor:
        if not AllToAllStatus.initialized:
            return grad_output[0]
        return tutel_custom_kernel.nccl_all_to_all_gather_async(grad_output, ctx.input_shape, ctx.num_slices_per_split, True)

class AllToAllGatherAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *input) -> Tensor:
        if not AllToAllStatus.initialized:
            return input[0]
        ctx.input_shape = input[0].shape
        output_shape = torch.Size([
            x if i != AllToAllStatus.split_dim else x * AllToAllStatus.num_split
            for i, x in enumerate(ctx.input_shape)
        ])
        ctx.num_slices_per_split = ctx.input_shape[:AllToAllStatus.split_dim].numel()
        return tutel_custom_kernel.nccl_all_to_all_gather_async(input, output_shape, ctx.num_slices_per_split, False)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        if not AllToAllStatus.initialized:
            return (grad_output,)
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(grad_output, ctx.input_shape, ctx.num_slices_per_split, True))


class BwdAllreduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx, doutput):
        dinput = torch.clone(doutput).contiguous()
        dist.all_reduce(dinput, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return (None, dinput)


class PreAllreduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        ctx.num_nodes = get_world_size(ctx.group)
        if ctx.num_nodes <= 1:
            return input
        ctx.input_shape = input.shape
        output = torch.empty([ctx.num_nodes, input.numel()], device=input.device, dtype=input.dtype)
        tensor_list = [x.contiguous() for x in torch.chunk(output, chunks=ctx.num_nodes, dim=0)]
        dist.all_gather(tensor_list=tensor_list, tensor=input.contiguous(), group=group)
        output = output.view([input.shape[0] * ctx.num_nodes] + list(input.shape[1:]))
        return output
    @staticmethod
    def backward(ctx, doutput):
        if ctx.num_nodes <= 1:
            return (None, doutput)
        dinput = torch.empty(ctx.input_shape, device=doutput.device, dtype=doutput.dtype)
        chunks = [x.contiguous() for x in torch.chunk(doutput.view(ctx.num_nodes, -1), chunks=ctx.num_nodes, dim=0)]
        dist.reduce_scatter(output=dinput, input_list=chunks, group=ctx.group)
        return (None, dinput, None)

class PostAllreduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        ctx.num_nodes = get_world_size(ctx.group)
        if ctx.num_nodes <= 1:
            return input
        ctx.input_shape = input.shape
        ctx.leading_dim = 0
        chunks = [x.contiguous() for x in torch.chunk(input, chunks=ctx.num_nodes, dim=ctx.leading_dim)]
        assert len(chunks) == ctx.num_nodes
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output=output, input_list=list(chunks), group=group)
        return output
    @staticmethod
    def backward(ctx, doutput):
        if ctx.num_nodes <= 1:
            return (None, doutput)
        dinput = torch.empty(ctx.input_shape, device=doutput.device, dtype=doutput.dtype)
        tensor_list = [x.contiguous() for x in torch.chunk(dinput, chunks=ctx.num_nodes, dim=ctx.leading_dim)]
        dist.all_gather(tensor_list=tensor_list, tensor=doutput, group=ctx.group)
        return (None, dinput)

