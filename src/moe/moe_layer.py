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
        JitKernels.func_fwd(gates1_s, indices1_s, locations1_s, reshaped_input, dispatched_input)
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        grad_gates1_s = None
        # grad_gates1_s = torch.empty([ctx.reshaped_input.size(0)], dtype=dispatched_input.dtype, device=dispatched_input.device)
        # JitKernels.func_bwd_gate(dispatched_input, indices1_s, locations1_s, ctx.reshaped_input, grad_gates1_s)

        grad_reshaped_input = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
        JitKernels.func_bwd_data(dispatched_input, ctx.gates1_s, indices1_s, locations1_s, grad_reshaped_input)
        return (grad_gates1_s, grad_reshaped_input)

class _CustomDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, expert_output: Tensor) -> Tensor:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        ctx.expert_output = expert_output
        ctx.gates1_s = gates1_s

        combined_output = torch.empty([gates1_s.size(0), expert_output.size(1)], dtype=gates1_s.dtype, device=gates1_s.device)
        JitKernels.func_bwd_data(expert_output, ctx.gates1_s, indices1_s, locations1_s, combined_output)
        return combined_output
        

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomEncoder._cache
        
        grad_gates1_s = torch.empty(indices1_s.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_bwd_gate(ctx.expert_output, indices1_s, locations1_s, combined_output, grad_gates1_s)

        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_fwd(ctx.gates1_s, indices1_s, locations1_s, combined_output, grad_expert_output)
        return (grad_gates1_s, grad_expert_output)


class _DebugFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputTensor: Tensor) -> Tensor:
        return inputTensor
    @staticmethod
    def backward(ctx: Any, outputTensor: Tensor) -> Tensor:
        print(outputTensor)
        return outputTensor

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

        global JitKernels
        if input.dtype == torch.float16:
            from . import jit_kernel_fp16 as jit_kernel
        else:
            from . import jit_kernel as jit_kernel
        JitKernels = jit_kernel

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
