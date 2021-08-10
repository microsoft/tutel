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

import my_custom_kernel_3

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



class _CustomKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, reshaped_input: Tensor) -> Tensor:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomKernel._cache

        # data type cast
        gates1_s_dtype = gates1_s.dtype
        indices1_s_dtype = indices1_s.dtype
        locations1_s_dtype = locations1_s.dtype
        reshaped_input_dtype = reshaped_input.dtype
        if gates1_s_dtype is not torch.float32:
            gates1_s = gates1_s.to(torch.float32)
        if indices1_s_dtype is not torch.int32:
            indices1_s = indices1_s.to(torch.int32)
        if locations1_s_dtype is not torch.int32:
            locations1_s = locations1_s.to(torch.int32)
        if reshaped_input_dtype is not torch.float32:
            reshaped_input = reshaped_input.to(torch.float32)
        
        ctx.reshaped_input = reshaped_input
        ctx.gates1_s = gates1_s

        dispatched_input = my_custom_kernel_3.forward(
            indices1_s,
            locations1_s,
            gates1_s,
            reshaped_input
        )
        
        if reshaped_input_dtype is not torch.float32:
            dispatched_input = dispatched_input.to(reshaped_input_dtype)
        
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomKernel._cache
        
        # data type cast
        dispatched_input_dtype = dispatched_input.dtype
        indices1_s_dtype = indices1_s.dtype
        locations1_s_dtype = locations1_s.dtype
        if dispatched_input_dtype is not torch.float32:
            dispatched_input = dispatched_input.to(torch.float32)
        if indices1_s_dtype is not torch.int32:
            indices1_s = indices1_s.to(torch.int32)
        if locations1_s_dtype is not torch.int32:
            locations1_s = locations1_s.to(torch.int32)

        grad_gates1_s = my_custom_kernel_3.backward_gates1_s(
            dispatched_input,
            indices1_s,
            locations1_s,
            ctx.reshaped_input
        )
        grad_reshaped_input = my_custom_kernel_3.backward_reshaped_input(
            dispatched_input,
            ctx.gates1_s,
            indices1_s,
            locations1_s
        )

        if dispatched_input_dtype is not torch.float32:
            grad_reshaped_input = grad_reshaped_input.to(dispatched_input_dtype)

        return (grad_gates1_s, grad_reshaped_input)

class _CustomDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, expert_output: Tensor) -> Tensor:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomKernel._cache
        
        # data type cast
        gates1_s_dtype = gates1_s.dtype
        indices1_s_dtype = indices1_s.dtype
        locations1_s_dtype = locations1_s.dtype
        expert_output_dtype = expert_output.dtype
        
        if gates1_s_dtype is not torch.float32:
            gates1_s = gates1_s.to(torch.float32)
        if indices1_s_dtype is not torch.int32:
            indices1_s = indices1_s.to(torch.int32)
        if locations1_s_dtype is not torch.int32:     
            locations1_s = locations1_s.to(torch.int32)
        if expert_output_dtype is not torch.float32:    
            expert_output = expert_output.to(torch.float32)
            
        ctx.expert_output = expert_output
        ctx.gates1_s = gates1_s

        combined_output = my_custom_kernel_3.backward_reshaped_input(
            expert_output,
            gates1_s,
            indices1_s,
            locations1_s
        )
        if expert_output_dtype is not torch.float32:
            combined_output = combined_output.to(expert_output_dtype)
        return combined_output
        

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor) -> Tuple[Tensor, Tensor]:
        [indices1_s, capacity, locations1_s, tmp, num_experts] = _CustomKernel._cache
        
        # data type cast
        combined_output_dtype = combined_output.dtype
        indices1_s_dtype = indices1_s.dtype
        locations1_s_dtype = locations1_s.dtype
        
        if combined_output_dtype is not torch.float32:
            combined_output = combined_output.to(torch.float32)
        if indices1_s_dtype is not torch.int32:
            indices1_s = indices1_s.to(torch.int32)
        if locations1_s_dtype is not torch.int32:
            locations1_s = locations1_s.to(torch.int32)
        
        grad_gates1_s = my_custom_kernel_3.backward_gates1_s(
            ctx.expert_output,
            indices1_s,
            locations1_s,
            combined_output
        )
        grad_expert_output = my_custom_kernel_3.forward(
            indices1_s,
            locations1_s,
            ctx.gates1_s,
            combined_output
        )

        if combined_output_dtype is not torch.float32:
            grad_expert_output = grad_expert_output.to(combined_output_dtype)
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

        self.gate = gating(model_dim=model_dim, num_experts=self.world_size * len(experts), use_fp32=True)

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

        l_aux, _CustomKernel._cache = self.gate(reshaped_input)

        # dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        # E, C, S = dispatch_mask.size()

        S, M = reshaped_input.size(0), reshaped_input.size(1) 
        E, C = _CustomKernel._cache[4], _CustomKernel._cache[1]
        
        # custom calculation of dispatched_input
        t = torch.ones(_CustomKernel._cache[3].size(), dtype=_CustomKernel._cache[3].dtype, device='cuda')
        dispatched_input = _CustomKernel.apply(t, reshaped_input)
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
        combined_output = _CustomDecoder.apply(_CustomKernel._cache[3], expert_output.view(E*C, M))
        
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]
        combined_output.l_aux = l_aux
        return combined_output

    def prepare_for_inference_(self):
        self.in_generation = True
