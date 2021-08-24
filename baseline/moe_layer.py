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

def get_world_size(group):
    try:
        return dist.get_world_size(group)
    except:
        return 1

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        ctx.group = group
        ctx.world_size = get_world_size(group)
        if ctx.world_size <= 1:
            return input
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        if ctx.world_size <= 1:
            return (None, grad_output)
        return (None, _AllToAll.apply(ctx.group, grad_output))


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate_type, model_dim: int, external_experts: Union[Module, ModuleList], fp32_gate = False, group: Optional[Any] = None, result_func = None) -> None:
        super().__init__()

        experts = external_experts
        self.expert_group = group if group is not None else dist.group.WORLD
        self.world_size = get_world_size(self.expert_group)
        self.result_func = result_func

        if gate_type == 'Top1Gate':
            from .top1gate import Top1Gate as gating
        elif gate_type == 'Top2Gate':
            from .top2gate import Top2Gate as gating
        else:
            raise Exception("Unrecognized gate_type: %s" % gate_type)

        self.gate = gating(model_dim=model_dim, num_experts=self.world_size * len(experts), use_fp32=fp32_gate)

        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList(experts)
        self.expert_group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in expert.parameters():
                p.expert = True  # type: ignore
        self.num_local_experts = len(self.experts)
        self.in_generation = False

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gate.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

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
        assert expected_bsz > 0 and expected_bsz == input.shape[0], "Current batch_size %s is changed or illegal to perform pre-designed load balance, expect: %s" % (input.shape[0], expected_bsz)

        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
            print("padding batch with unexpected size %s (expected: %s)" % (input_shape[0], expected_bsz))
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
            expected_dim = int(distributed_utils.all_reduce(  # FIXME: add package for `distributed_utils`
                reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

        l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(reshaped_input)

        dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        M = reshaped_input.size(1)
        assert reshaped_input.size() == (S, M)
        # einsum("sec,sm->ecm")
        dispatched_input = torch.mm(dispatch_mask.view(E*C, S), reshaped_input)  # -> (E*C),M

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
        combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, M))

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]
        combined_output.l_aux = l_aux
        result_output = combined_output
        return self.result_func(result_output) if self.result_func is not None else result_output

    def prepare_for_inference_(self):
        self.in_generation = True
