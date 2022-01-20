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


class AllToAllStatus:
    initialized = False
    gather_tensor = None
    scatter_tensors = []
    world_size = 0

    @staticmethod
    def init(group: dist.ProcessGroup, num_split: int, split_dim: int, gather_tensor_ref: Tensor) -> None:
        AllToAllStatus.world_size = get_world_size(group)
        if AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return

        # Make sure addresses of tensors allocated are not in use
        torch.cuda.synchronize()

        # Initialize NCCL
        world_rank = get_world_rank(group)
        nccl_unique_id_size = tutel_custom_kernel.get_nccl_unique_id_size()
        nccl_unique_id = torch.zeros([nccl_unique_id_size], dtype=torch.int8).cpu()
        if world_rank == 0:
            tutel_custom_kernel.get_nccl_unique_id(nccl_unique_id)
        nccl_unique_id = nccl_unique_id.to(gather_tensor_ref.device)
        dist.broadcast(nccl_unique_id, 0, group)
        tutel_custom_kernel.init_nccl(
            nccl_unique_id.cpu(),
            AllToAllStatus.world_size,
            world_rank,
            num_split,
            gather_tensor_ref.shape[:split_dim].numel())

        # Initialize dedicate buffers for scatter/gather
        AllToAllStatus.gather_tensor = torch.empty_like(gather_tensor_ref).contiguous()
        scatter_tensors_ref = gather_tensor_ref.split(
            gather_tensor_ref.shape[split_dim] // num_split,
            dim=split_dim)
        AllToAllStatus.scatter_tensors = [
            torch.empty_like(x).contiguous() for x in scatter_tensors_ref]

        AllToAllStatus.initialized = True

class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        ctx.group = group
        world_size = get_world_size(group)
        if world_size <= 1:
            return input
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return (None, AllToAll.apply(ctx.group, grad_output))

class CurrentStreamRelease(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return input
        ctx.idx = idx
        input = input.contiguous()
        return tutel_custom_kernel.current_stream_release(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return (grad_output, None)
        return (tutel_custom_kernel.current_stream_acquire(grad_output, ctx.idx), None)

class CurrentStreamAcquire(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, idx: int) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return input
        ctx.idx = idx
        return tutel_custom_kernel.current_stream_acquire(input, idx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return (grad_output, None)
        grad_output = grad_output.contiguous()
        return (tutel_custom_kernel.current_stream_release(grad_output, ctx.idx), None)

class AllToAllScatterAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tuple[Tensor]:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return (input,)
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(
            input,
            AllToAllStatus.scatter_tensors,
            False))

    @staticmethod
    def backward(ctx: Any, *grad_output) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return grad_output[0]
        return tutel_custom_kernel.nccl_all_to_all_gather_async(
            grad_output,
            AllToAllStatus.gather_tensor,
            True)

class AllToAllGatherAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *input) -> Tensor:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return input[0]
        return tutel_custom_kernel.nccl_all_to_all_gather_async(
            input,
            AllToAllStatus.gather_tensor,
            False)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        if not AllToAllStatus.initialized or AllToAllStatus.world_size <= 1:
            return (grad_output,)
        return tuple(tutel_custom_kernel.nccl_all_to_all_scatter_async(
            grad_output,
            AllToAllStatus.scatter_tensors,
            True))


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
        dist.all_gather(tensor_list=tensor_list, tensor=input.contiguous())
        output = output.view(list(input.shape[:0]) + [input.shape[0] * ctx.num_nodes] + list(input.shape[1:]))
        return output
    @staticmethod
    def backward(ctx, doutput):
        if get_world_size(ctx.group) <= 1:
            return (None, doutput)
        dinput = torch.empty(ctx.input_shape, device=doutput.device, dtype=doutput.dtype)
        chunks = [x.contiguous() for x in torch.chunk(doutput.view(ctx.num_nodes, -1), chunks=ctx.num_nodes, dim=0)]
        dist.reduce_scatter(output=dinput, input_list=chunks)
        return (None, dinput)

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
        dist.reduce_scatter(output=output, input_list=list(chunks))
        return output
    @staticmethod
    def backward(ctx, doutput):
        if ctx.num_nodes <= 1:
            return (None, doutput)
        dinput = torch.empty(ctx.input_shape, device=doutput.device, dtype=doutput.dtype)
        tensor_list = [x.contiguous() for x in torch.chunk(dinput, chunks=ctx.num_nodes, dim=ctx.leading_dim)]
        dist.all_gather(tensor_list=tensor_list, tensor=doutput)
        return (None, dinput)

