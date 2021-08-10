# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList

import custom_kernel

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

def get_world_size():
    try:
        world_size = dist.get_world_size(self.expert_group)
    except:
        # FIXME: dist.get_world_size(self.expert_group)
        world_size = 1
    return world_size

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        return input # FIXME: no implementation

        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, *grad_output) # FIXME: no implementation

        return (None, _AllToAll.apply(ctx.group, *grad_output))



class _CustomEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, reshaped_input: Tensor) -> Tensor:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache

        ctx.reshaped_input = reshaped_input
        ctx.gates1_s = gates1_s

        dispatched_input = torch.zeros([num_experts * capacity, reshaped_input.size(1)], dtype=reshaped_input.dtype, device=reshaped_input.device)
        func_fwd(gates1_s, indices1_s, locations1_s, reshaped_input, dispatched_input)
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        grad_gates1_s = None
        # grad_gates1_s = torch.empty([ctx.reshaped_input.size(0)], dtype=dispatched_input.dtype, device=dispatched_input.device)
        # func_bwd_gate(dispatched_input, indices1_s, locations1_s, ctx.reshaped_input, grad_gates1_s)

        grad_reshaped_input = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
        func_bwd_data(dispatched_input, ctx.gates1_s, indices1_s, locations1_s, grad_reshaped_input)
        return (grad_gates1_s, grad_reshaped_input)

class _CustomDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, expert_output: Tensor) -> Tensor:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        ctx.expert_output = expert_output
        ctx.gates1_s = gates1_s

        combined_output = torch.empty([gates1_s.size(0), expert_output.size(1)], dtype=gates1_s.dtype, device=gates1_s.device)
        func_bwd_data(expert_output, ctx.gates1_s, indices1_s, locations1_s, combined_output)
        return combined_output
        

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        grad_gates1_s = torch.empty(indices1_s.shape, dtype=combined_output.dtype, device=combined_output.device)
        func_bwd_gate(ctx.expert_output, indices1_s, locations1_s, combined_output, grad_gates1_s)

        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        func_fwd(ctx.gates1_s, indices1_s, locations1_s, combined_output, grad_expert_output)
        return (grad_gates1_s, grad_expert_output)


class _DebugFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputTensor: Tensor) -> Tensor:
        return inputTensor
    @staticmethod
    def backward(ctx: Any, outputTensor: Tensor) -> Tensor:
        print(outputTensor)
        return outputTensor

class JitKernel:
    @staticmethod
    def create(source):
        if not hasattr(JitKernel, '__CTX__'):
            torch.cuda.init()
            JitKernel.__CTX__ = 0
        __ctx__ = JitKernel.__CTX__
        JitKernel.__CTX__ += 1
        with open(f'/tmp/{__ctx__}.cu', 'w') as fp:
            fp.write(source)

        def func(*inputs):
            custom_kernel.invoke(inputs, __ctx__)
        return func

func_fwd = JitKernel.create('''
extern "C" __global__ __launch_bounds__(64) void forward_dispatched_input(float* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 64
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 8
  // [thread_extent] threadIdx.y = 64
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[(((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))])) : 0.000000e+00f);
  }
  for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[(((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32768))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32832))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32896))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32960))])) : 0.000000e+00f);
  }
}
''')

func_bwd_gate = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void backward_gates1_s(float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.x = 32
  float grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = 0.000000e+00f;
  for (int HIDDEN_outer = 0; HIDDEN_outer < 64; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((int)blockIdx.x))] < 1024) ? (dispatched_input[(((((indices1_s[(((int)blockIdx.x))] * 2097152) + (locations1_s[(((int)blockIdx.x))] * 2048)) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))] * reshaped_input[((((((int)blockIdx.x) * 2048) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))]) : 0.000000e+00f));
  }
  float red_buf0[1];
  unsigned mask[1];
  float t0[1];
  red_buf0[(0)] = grad_gates1_s_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    grad_gates1_s[(((int)blockIdx.x))] = red_buf0[(0)];
  }
}
''')

func_bwd_data = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void backward_reshaped_input(float* __restrict__ dispatched_input, float* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 256
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 32
  // [thread_extent] threadIdx.y = 32
  grad_reshaped_input[((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))] = ((locations1_s[((((int)blockIdx.x) * 8))] < 1024) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[(((((indices1_s[((((int)blockIdx.x) * 8))] * 2097152) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))] = ((locations1_s[((((int)blockIdx.x) * 8))] < 1024) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[((((((indices1_s[((((int)blockIdx.x) * 8))] * 2097152) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2048))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2080))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4096))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4128))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6144))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6176))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8192))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8224))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10240))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10272))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12288))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12320))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14336))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14368))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
}
''')

class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate_type, experts: Union[Module, ModuleList], model_dim: int, group: Optional[Any] = None) -> None:
        super().__init__()
        self.world_size = get_world_size()

        if gate_type == 'Top1Gate':
            from .top1gate import Top1Gate as gating
        elif gate_type == 'Top2Gate':
            from .top2gate import Top2Gate as gating
        else:
            raise Exception(f"Unrecognized gate_type: {gate_type}")

        self.gate = gating(model_dim=model_dim, num_experts=self.world_size * len(experts))

        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore
        self.num_local_experts = len(self.experts)
        self.in_generation = False

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)

        if not hasattr(self, 'expected_bsz'):
            self.expected_bsz = input.shape[0]

        expected_bsz = getattr(self, 'expected_bsz')
        assert expected_bsz > 0 and expected_bsz == input.shape[0], f"Current batch_size {input.shape[0]} is changed or illegal to perform pre-designed load balance, expect: {expected_bsz}"

        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
            print(f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})")
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:input_shape[0], :, :] = input
            input = padded_input

        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = int(distributed_utils.all_reduce(  # FIXME: no package for `distributed_utils`
                reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

        l_aux, _CustomEncoder._cache = self.gate(reshaped_input)

        # dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        # E, C, S = dispatch_mask.size()

        S, M = reshaped_input.size(0), reshaped_input.size(1) 
        E, C = _CustomEncoder._cache[4], _CustomEncoder._cache[1]
        
        # custom calculation of dispatched_input
        if not hasattr(self, 'ones_helper'):
            self.ones_helper = torch.ones(_CustomEncoder._cache[3].size(), dtype=_CustomEncoder._cache[3].dtype, device=_CustomEncoder._cache[3].device)

        dispatched_input = _CustomEncoder.apply(self.ones_helper, reshaped_input)
        dispatched_input = _AllToAll.apply(self.expert_group, dispatched_input)
        
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)

        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.expert_group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)

        # einsum("sec,ecm->sm")
        combined_output = _CustomDecoder.apply(_CustomEncoder._cache[3], expert_output.view(E*C, M))
        
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]
        combined_output.l_aux = l_aux
        return combined_output

    def prepare_for_inference_(self):
        self.in_generation = True
