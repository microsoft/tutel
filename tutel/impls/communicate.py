# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import os
import re
import time
import torch
import logging
import datetime

from torch import Tensor
import torch.distributed as dist

from .jit_compiler import tutel_custom_kernel

TUTEL_GLOBAL_TIMEOUT_SEC = int(os.environ.get('TUTEL_GLOBAL_TIMEOUT_SEC', 86400))

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

def barrier(group=None):
    if get_world_size(group) == 1:
        return
    dist.barrier(group=group)


TUTEL_GROUPING_CACHE = {}
TUTEL_SHARED_NCCL = False
TUTEL_SKIP_A2A = int(os.environ.get('SKIP_A2A', 0)) > 0
TUTEL_SUBGROUP_CACHE = {}

def create_standalone_group():
    try:
        return dist.new_group(ranks=[get_world_rank()])
    except:
        return None

def create_groups_from_world(group_count, include_init=None, parent_group=None):
    parent_size, world_size = get_world_size(parent_group), get_world_size()

    class DistributedProperties:
        pass

    if parent_size > 1 and parent_size < world_size:
        assert include_init is None, 'Torch distributed environment had been initialized.'
        raise Exception('Splitting nesting groups from a subgroup is yet not allowed, please report an issue for this requirement.')

    backend = TUTEL_GROUPING_CACHE.get('', include_init)
    if include_init:
        assert backend == include_init, "Only 1 backend type is allowed, get: %s v.s. %s" % (backend, include_init)
        TUTEL_GROUPING_CACHE[''] = backend

    if group_count in TUTEL_GROUPING_CACHE:
        return TUTEL_GROUPING_CACHE[group_count]

    try:
      if ('LOCAL_RANK' not in os.environ) and ('OMPI_COMM_WORLD_SIZE' in os.environ):
          if include_init:
              dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=TUTEL_GLOBAL_TIMEOUT_SEC),
                  init_method='tcp://%s:%s' % (os.environ['MASTER_ADDR'], os.environ.get('MASTER_PORT', '23456')),
                  rank=int(os.environ['OMPI_COMM_WORLD_RANK']), world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']))
          dist_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
      else:
          if include_init:
              dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=TUTEL_GLOBAL_TIMEOUT_SEC))
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

    original_group_count = group_count
    if group_count < 0:
        group_count = glob_world_size // -group_count
    assert group_count > 0 and glob_world_size % group_count == 0, f"Expected to evenly divide devices into {group_count} groups, while the world size of current sesion is {glob_world_size}."

    dist_group_size = group_count
    dist_world_size = glob_world_size // dist_group_size
    dist_world_rank = glob_world_rank % dist_world_size
    dist_group_rank = glob_world_rank // dist_world_size

    if is_distributed:
        global_group = model_group = data_group = dist.group.WORLD

        if dist_world_size != glob_world_size:
            groups, inner_ranks = [], []
            for gr in range(dist_group_size):
                group_ranks = [x for x in range(gr * dist_world_size, (gr + 1) * dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks, timeout=datetime.timedelta(seconds=TUTEL_GLOBAL_TIMEOUT_SEC))]
                inner_ranks += [group_ranks]
            model_group = groups[dist_group_rank]

        if dist_group_size != glob_world_size:
            groups, outer_ranks = [], []
            for gr in range(dist_world_size):
                group_ranks = [x for x in range(gr, dist_world_size * dist_group_size, dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks, timeout=datetime.timedelta(seconds=TUTEL_GLOBAL_TIMEOUT_SEC))]
                outer_ranks += [group_ranks]
            data_group = groups[dist_world_rank]
    else:
        model_group, data_group, global_group = None, None, None

    result = DistributedProperties()

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

    global TUTEL_SHARED_NCCL
    if is_distributed and not TUTEL_SHARED_NCCL and backend == 'nccl':
        try:
            world_size = get_world_size()
            world_rank = get_world_rank()
            nccl_unique_id_size = tutel_custom_kernel.get_nccl_unique_id_size()
            nccl_unique_id = torch.zeros([nccl_unique_id_size], dtype=torch.int8).cpu()
            if world_rank == 0:
                tutel_custom_kernel.get_nccl_unique_id(nccl_unique_id)
            nccl_unique_id = nccl_unique_id.cuda()
            dist.broadcast(nccl_unique_id, 0, None)
            tutel_custom_kernel.init_shared_nccl(
                nccl_unique_id.cpu(),
                world_size,
                world_rank)
            TUTEL_SHARED_NCCL = True
        except:
            pass

    TUTEL_GROUPING_CACHE[original_group_count] = result
    return result

def swap_axis(t, x, y):
    return t if x == y else t.swapaxes(x, y)

def simple_all_reduce(input, group=None, op=torch.distributed.ReduceOp.SUM, inplace=False):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    output = input if inplace else torch.clone(input, memory_format=torch.contiguous_format)
    dist.all_reduce(output, op=op, group=group)
    return output

def simple_all_to_all(input, group=None, background=False):
    world_size = get_world_size(group)
    input = input.contiguous()
    if world_size == 1 or TUTEL_SKIP_A2A:
        return input if not background else (input, lambda *args: None)
    simple_all_to_all._use_builtins = True
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
    assert input.size(0) % world_size == 0, "Cannot evenly divide dim length %s into %s slices" % (input.size(0), world_size)
    input = input.contiguous()
    return input.chunk(chunks=world_size, dim=0)[get_world_rank(group)]

def simple_reduce_scatter(input, group=None, op=torch.distributed.ReduceOp.SUM):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    input = input.contiguous()
    assert input.size(0) % world_size == 0, "Cannot evenly divide dim length %s into %s slices" % (input.size(0), world_size)
    if not input.is_cuda:
      return simple_split(simple_all_reduce(input, group, op=op), group=group)
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
    dist.all_gather(tensor_list=tensor_list, tensor=input.view(1, -1), group=group)
    return output.view([-1,] + list(input.shape[1:]))

def batch_all_to_all_v(datas, partition_sizes, group=None):
    assert group is None, "batched_all_to_all_v() with non-default group is not implemented in this version."
    assert type(datas) in (tuple, list), "data type for batch_all_to_all_v() is not a list of tensors"
    in_sizes = partition_sizes
    if type(in_sizes) != torch.Tensor:
        in_sizes = torch.tensor(in_sizes, dtype=torch.int64, device=datas[0].device)
    else:
        in_sizes = in_sizes.to(torch.int64)
    world_size = get_world_size(group)
    assert in_sizes.numel() == world_size
    if world_size == 1:
        return list(datas), in_sizes
    out_sizes = simple_all_to_all(in_sizes, group=group)
    datas = [data.contiguous().view(-1).cuda() for data in datas]
    outputs = [torch.empty([out_sizes.sum()], dtype=data.dtype, device=data.device) for data in datas]
    tutel_custom_kernel.batch_all_to_all_v(datas, outputs, in_sizes, out_sizes)
    return outputs, out_sizes

def batch_all_gather_v(datas, group=None):
    assert group is None, "batch_all_gather_v() with non-default group is not implemented in this version."
    assert type(datas) in (tuple, list), "data type for batch_all_gather_v() is not a list of tensors"
    datas = [data.contiguous().view(-1).cuda() for data in datas]
    input_size = torch.tensor([int(datas[0].numel())], dtype=torch.int64, device=datas[0].device)
    world_size = get_world_size(group)
    if world_size == 1:
        return list(datas), input_size
    output_sizes = simple_all_gather(input_size)
    size_int = int(output_sizes.sum())
    outputs = [torch.empty([size_int], dtype=data.dtype, device=data.device) for i, data in enumerate(datas)]
    tutel_custom_kernel.batch_all_gather_v(datas, outputs, output_sizes)
    return outputs, output_sizes

class AllToAllStatus:
    initialized = False
    num_split = 0
    split_dim = 0
    max_num_split = 32

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
                AllToAllStatus.max_num_split)
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
        ctx.num_split = AllToAllStatus.num_split
        ctx.num_slices_per_split = ctx.input_shape[:AllToAllStatus.split_dim].numel()
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(input, output_shape, ctx.num_split, ctx.num_slices_per_split, False))

    @staticmethod
    def backward(ctx: Any, *grad_output) -> Tensor:
        if not AllToAllStatus.initialized:
            return grad_output[0]
        return tutel_custom_kernel.nccl_all_to_all_gather_async(grad_output, ctx.input_shape, ctx.num_split, ctx.num_slices_per_split, True)

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
        ctx.num_split = AllToAllStatus.num_split
        ctx.num_slices_per_split = ctx.input_shape[:AllToAllStatus.split_dim].numel()
        return tutel_custom_kernel.nccl_all_to_all_gather_async(input, output_shape, ctx.num_split, ctx.num_slices_per_split, False)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        if not AllToAllStatus.initialized:
            return (grad_output,)
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(grad_output, ctx.input_shape, ctx.num_split, ctx.num_slices_per_split, True))


class RestoreBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input, output, group=None):
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx: Any, grad_output):
        grad_output = simple_all_to_all(grad_output, group=ctx.group)
        return (None, grad_output)


class PrimAllToAll2D(torch.autograd.Function):
    LOCAL_SIZE = 0

    @staticmethod
    def forward(ctx, x, input_dim, output_dim):
        if PrimAllToAll2D.LOCAL_SIZE == 0:
            PrimAllToAll2D.LOCAL_SIZE = int(os.environ.get('LOCAL_SIZE', 1))
            if PrimAllToAll2D.LOCAL_SIZE == 1:
                logging.warning("LOCAL_SIZE (> 1) for AllToAll 2DH is not set, please set the correct LOCAL_SIZE variable, or using mpiexec to launch tutel.net")
        ctx.input_dim = input_dim
        ctx.output_dim = output_dim
        dist = create_groups_from_world(-PrimAllToAll2D.LOCAL_SIZE)
        y = all_to_all(x, input_dim, output_dim, group=dist.data_group)
        y = all_to_all(y, input_dim, output_dim, group=dist.model_group)
        y = y.view(list(y.shape[:input_dim]) + [get_world_size(dist.model_group), get_world_size(dist.data_group), -1] + list(y.shape[input_dim + 1:])).swapaxes(input_dim, input_dim + 1).contiguous().view(y.shape)
        return y
    @staticmethod
    def backward(ctx, dy):
        return (PrimAllToAll2D.apply(dy, ctx.output_dim, ctx.input_dim), None, None)

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
    def transform(input, input_dim, output_dim, group=None, background=False, use_2dh=False):
        """
          [HY] X LY Z -> [HX] HY LX LY Z
        """
        if use_2dh:
            assert background == False, "Background mode for AllToAll 2DH is not implemented."
            return PrimAllToAll2D.apply(input, input_dim, output_dim)

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
        input = PrimReducescatter.apply(input, torch.distributed.ReduceOp.SUM, group)
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
    def transform(input, dim, fused=False, group=None):
        input = swap_axis(input, 0, dim)
        input = PrimAllgather.apply(input, fused, group)
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

    @staticmethod
    def zero_scatter(input, scatter_fn, group=None):
        group_size = get_world_size(group)
        full_size = input.numel()
        if full_size % group_size == 0:
            data = input.reshape(-1)
        else:
            data = torch.zeros([(full_size + group_size - 1) // group_size * group_size], device=input.device, dtype=input.dtype)
            data[:full_size] = input.reshape(-1)
        return scatter_fn(data, group=group), input.shape


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

def pre_expert_permute(input, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    input = input.view([world_size, -1] + list(input.shape[1:]))
    input = input.permute([1, 0] + list(range(2, input.dim())))
    input = input.contiguous().view([input.shape[0], -1] + list(input.shape[3:]))
    return input

def post_expert_permute(input, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    input = input.view([input.shape[0], world_size, -1] + list(input.shape[2:]))
    input = input.permute([1, 0] + list(range(2, input.dim())))
    input = input.contiguous().view([-1] + list(input.shape[2:]))
    return input

all_to_all = PrimAllToAll.transform
all_to_all_single = PrimAllToAll.single
zero_gather = PrimAllgather.zero_gather
zero_scatter = PrimAllgather.zero_scatter
all_gather = PrimAllgather.transform
spatial_split = PrimSpatialSplit.transform
reduce_scatter = PrimReducescatter.transform
allreduce_forward = PrimFwdAllreduce.transform
allreduce_backward = PrimBwdAllreduce.transform
