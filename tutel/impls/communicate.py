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

    result.model_size = dist_world_size
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

def swap_axis(t, x, y):
    return t if x == y else t.swapaxes(x, y)

def simple_all_reduce(input, group=None, op=torch.distributed.ReduceOp.SUM):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    output = torch.clone(input, memory_format=torch.contiguous_format)
    dist.all_reduce(output, op=op, group=group)
    return output

def simple_all_to_all(input, group=None, background=False):
    world_size = get_world_size(group)
    if world_size == 1:
        return input if not background else (input, lambda *args: None)
    simple_all_to_all._use_builtins = True
    input = input.contiguous()
    output = torch.empty_like(input)
    if background:
        future_op = dist.all_to_all_single(output, input, group=group, async_op=True)
        return output, future_op.wait
    dist.all_to_all_single(output, input, group=group)
    return output

def simple_split(input, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    assert input.size(0) % world_size == 0, "Cannot evenly devide dim length %s into %s slices" % (input.size(0), world_size)
    input = input.contiguous()
    return input.chunk(chunks=world_size, dim=0)[get_world_rank(group)]

def simple_reduce_scatter(input, group=None, op=torch.distributed.ReduceOp.SUM):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    input = input.contiguous()
    assert input.size(0) % world_size == 0, "Cannot evenly devide dim length %s into %s slices" % (input.size(0), world_size)
    if not input.is_cuda:
      return simple_split(simple_all_reduce(input, group, op=op))
    chunks = list(input.chunk(chunks=world_size, dim=0))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output=output, input_list=chunks, group=group, op=op)
    return output

def simple_all_gather(input, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    input = input.contiguous()
    output = torch.empty([world_size, input.numel()], device=input.device, dtype=input.dtype)
    tensor_list = list(torch.chunk(output, chunks=world_size, dim=0))
    dist.all_gather(tensor_list=tensor_list, tensor=input, group=group)
    return output.view([-1,] + list(input.shape[1:]))

class AllToAllStatus:
    initialized = False
    num_split = 0
    split_dim = 0

    @staticmethod
    def init(group: dist.ProcessGroup, num_split: int, split_dim: int) -> None:
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
            nccl_unique_id = nccl_unique_id.cuda()
            dist.broadcast(nccl_unique_id, 0, group)
            tutel_custom_kernel.init_nccl(
                nccl_unique_id.cpu(),
                world_size,
                world_rank,
                AllToAllStatus.num_split)
            AllToAllStatus.initialized = True

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

class NcclStreamRelease(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized:
            return input
        ctx.idx = idx
        return tutel_custom_kernel.nccl_stream_release(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return (grad_output, None)
        return (tutel_custom_kernel.nccl_stream_acquire(grad_output, ctx.idx), None)

class NcclStreamAcquire(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized:
            return input
        ctx.idx = idx
        return tutel_custom_kernel.nccl_stream_acquire(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return (grad_output, None)
        return (tutel_custom_kernel.nccl_stream_release(grad_output, ctx.idx), None)

class AllToAll2DAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return input
        return tutel_custom_kernel.nccl_all_to_all_2d_async(input)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized:
            return (grad_output, None)
        return tutel_custom_kernel.nccl_all_to_all_2d_async(grad_output)

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


class RestoreBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input, output, group=None):
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx: Any, grad_output):
        grad_output = simple_all_to_all(grad_output, group=ctx.group)
        return (None, grad_output)


class PrimAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return simple_all_to_all(input, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (PrimAllToAll.apply(grad_output, ctx.group), None)

    @staticmethod
    def single(input, group=None):
        return PrimAllToAll.apply(input, group)

    @staticmethod
    def transform(input, input_dim, output_dim, group=None, background=False):
        """
          [HY] X LY Z -> [HX] HY LX LY Z
        """
        if background:
            world_size = get_world_size(group)
            if input_dim == output_dim or world_size == 1:
                return lambda *args: input

            if input_dim == 0:
                reshaped_input = input.view(list(input.shape[:output_dim]) + [world_size, -1] + list(input.shape[output_dim + 1:]))
                reshaped_input = reshaped_input.permute([output_dim] + list(range(output_dim)) + list(range(output_dim + 1, reshaped_input.dim())))
                output, f_wait = simple_all_to_all(reshaped_input, group, background=True)

                def f_async():
                    f_wait()
                    local_input = RestoreBackward.apply(output, reshaped_input)
                    local_input = local_input.view([-1] + list(local_input.shape[2:]))
                    return local_input

                return f_async
            elif output_dim == 0:
                reshaped_input = input
                output, f_wait = simple_all_to_all(reshaped_input, group, background=True)

                def f_async():
                    f_wait()
                    local_input = RestoreBackward.apply(output, reshaped_input)
                    local_input = local_input.view([world_size, -1] + list(local_input.shape[1:]))
                    local_input = local_input.permute(list(range(1, input_dim + 1)) + [0] + list(range(input_dim + 1, local_input.dim())))
                    local_input = local_input.contiguous().view(list(local_input.shape[:input_dim]) + [-1] + list(local_input.shape[input_dim + 2:]))
                    return local_input

                return f_async
            else:
                raise Exception('Unhandle async branch case for flexible all_to_all()')

        if input_dim == output_dim:
            return input

        world_size = get_world_size(group)
        if world_size == 1:
            return input

        if input_dim == 0:
            return all_to_all(input, input_dim, output_dim, group=group, background=True)()
        elif output_dim == 0:
            return all_to_all(input, input_dim, output_dim, group=group, background=True)()
        else:
            reshaped_input = swap_axis(input, 0, output_dim)
            reshaped_input = PrimAllToAll.transform(reshaped_input, input_dim, 0, group)
            reshaped_input = swap_axis(reshaped_input, 0, output_dim).contiguous()
        return reshaped_input

class PrimBwdAllreduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, op=torch.distributed.ReduceOp.SUM, group=None):
        ctx.group = group
        ctx.op = op
        return input
    @staticmethod
    def backward(ctx, doutput):
        return (simple_all_reduce(doutput, group=ctx.group, op=ctx.op), None, None)
    @staticmethod
    def transform(input, op=torch.distributed.ReduceOp.SUM, group=None):
        return PrimBwdAllreduce.apply(input, op, group)

class PrimFwdAllreduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, op=torch.distributed.ReduceOp.SUM, group=None):
        return simple_all_reduce(input, group=group)
    @staticmethod
    def backward(ctx, doutput):
        return (doutput, None, None)
    @staticmethod
    def transform(input, op=torch.distributed.ReduceOp.SUM, group=None):
        return PrimFwdAllreduce.apply(input, op, group)

class PrimReducescatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, op=torch.distributed.ReduceOp.SUM, group=None):
        ctx.group = group
        return simple_reduce_scatter(input, group, op=op)

    @staticmethod
    def backward(ctx, doutput):
        return (simple_all_gather(doutput, ctx.group), None, None)

    @staticmethod
    def transform(input, dim, group=None):
        input = swap_axis(input, 0, dim)
        input = PrimReducescatter.apply(input, group)
        input = swap_axis(input, 0, dim)
        return input

class PrimAllgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fused=False, group=None):
        ctx.group = group
        ctx.fused = fused
        return simple_all_gather(input, group)

    @staticmethod
    def backward(ctx, doutput):
        if ctx.fused:
            return (simple_reduce_scatter(doutput, ctx.group), None, None)
        return (simple_split(doutput, ctx.group), None, None)

    @staticmethod
    def transform(input, dim, group=None):
        input = swap_axis(input, 0, dim)
        input = PrimAllgather.apply(input, False, group)
        input = swap_axis(input, 0, dim)
        return input

    @staticmethod
    def zero_gather(input, full_shape=None, group=None):
        if not full_shape:
            full_shape = [x for x in input.shape]
            full_shape[0] *= get_world_size(group)
        numel = 1
        for x in full_shape:
            numel *= int(x)
        input = PrimAllgather.apply(input, True, group)
        return input.view(-1)[:numel].view(full_shape)


class PrimSpatialSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return simple_split(input, ctx.group)

    @staticmethod
    def backward(ctx, doutput):
        return (simple_all_gather(doutput, ctx.group), None)

    @staticmethod
    def transform(input, dim, group=None):
        input = swap_axis(input, 0, dim)
        input = PrimSpatialSplit.apply(input, group)
        input = swap_axis(input, 0, dim)
        return input

all_to_all = PrimAllToAll.transform
all_to_all_single = PrimAllToAll.single
zero_gather = PrimAllgather.zero_gather
all_gather = PrimAllgather.transform
spatial_split = PrimSpatialSplit.transform
reduce_scatter = PrimReducescatter.transform
allreduce_forward = PrimFwdAllreduce.transform
allreduce_backward = PrimBwdAllreduce.transform
