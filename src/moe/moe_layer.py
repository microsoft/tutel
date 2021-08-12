from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F


def get_world_size():
    try:
        world_size = dist.get_world_size(self.expert_group)
    except:
        # FIXME: dist.get_world_size(self.expert_group)
        world_size = 1
    return world_size


def load_kernels(dtype):
    global JitKernels
    try:
        return JitKernels
    except:
      if dtype == torch.float16:
          from . import jit_kernel_fp16 as jit_kernel
      else:
          from . import jit_kernel as jit_kernel
      JitKernels = jit_kernel
      return JitKernels


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        return input # FIXME: not implemented

        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        return (None, *grad_output) # FIXME: not implemented

        return (None, _AllToAll.apply(ctx.group, *grad_output))


class Top1Gate(torch.nn.Module):

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        input_noise_type=None,
        capacity_factor=1.0,
    ):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor

    def forward(self, input: torch.Tensor):
        logits = self.wg(input)
        num_tokens, num_experts = logits.shape[0], logits.shape[1]

        EVAL_CAPACITY_TOKEN_FRACTION = 0.25
        capacity = int(EVAL_CAPACITY_TOKEN_FRACTION * num_tokens) if not self.training else int(self.capacity_factor * ((num_tokens + num_experts - 1) // num_experts))

        indices1_s = torch.argmax(logits, dim=1)

        gates = F.softmax(logits, dim=1)
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)
        gates1_s = (gates * mask1).sum(dim=1)

        jit_kernel = load_kernels(input.dtype)

        locations1_s = torch.empty([num_tokens,], dtype=torch.int32, device=logits.device)
        jit_kernel.fwd_cumsum(indices1_s.to(torch.int32), locations1_s)

        # Compute l_aux
        if gates.dtype == torch.float32:
            me = torch.sum(gates, dim=0)
            ce = torch.sum(mask1.to(gates.dtype), dim=0)
            l_aux = torch.sum(me * ce) * (num_experts / (gates.size(0) * gates.size(0)))
        else:
            # Avoid data overflow in float16 mode
            me = torch.mean(gates, dim=0)
            ce = torch.mean(mask1.to(gates.dtype), dim=0)
            l_aux = torch.sum(me * ce) * num_experts

        return l_aux, (indices1_s.to(torch.int32), capacity, locations1_s.to(torch.int32), gates1_s, num_experts)


class _CustomEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, reshaped_input: Tensor):
        [indices1_s, capacity, locations1_s, _, num_experts] = _CustomEncoder._cache

        ctx.reshaped_input = reshaped_input
        ctx.gates1_s = gates1_s

        dispatched_input = torch.zeros([num_experts * capacity, reshaped_input.size(1)], dtype=reshaped_input.dtype, device=reshaped_input.device)
        JitKernels.func_fwd(gates1_s, indices1_s, locations1_s, reshaped_input, dispatched_input)
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor):
        [indices1_s, capacity, locations1_s, gates1_s, num_experts] = _CustomEncoder._cache
        
        # grad_gates1_s = None
        # grad_gates1_s = torch.empty([ctx.reshaped_input.size(0)], dtype=dispatched_input.dtype, device=dispatched_input.device)
        # JitKernels.func_bwd_gate(dispatched_input, indices1_s, locations1_s, ctx.reshaped_input, grad_gates1_s)

        grad_reshaped_input = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
        JitKernels.func_bwd_data(dispatched_input, ctx.gates1_s, indices1_s, locations1_s, grad_reshaped_input)
        return (None, grad_reshaped_input)


class _CustomDecoder(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, expert_output: Tensor):
        [indices1_s, capacity, locations1_s, _, num_experts] = _CustomEncoder._cache
        
        ctx.expert_output = expert_output
        ctx.gates1_s = gates1_s

        combined_output = torch.empty([gates1_s.size(0), expert_output.size(1)], dtype=gates1_s.dtype, device=gates1_s.device)
        JitKernels.func_bwd_data(expert_output, ctx.gates1_s, indices1_s, locations1_s, combined_output)
        return combined_output
        
    @staticmethod
    def backward(ctx: Any, combined_output: Tensor):
        [indices1_s, capacity, locations1_s, gates1_s, num_experts] = _CustomEncoder._cache
        
        grad_gates1_s = torch.empty(indices1_s.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_bwd_gate(ctx.expert_output, indices1_s, locations1_s, combined_output, grad_gates1_s)

        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_fwd(ctx.gates1_s, indices1_s, locations1_s, combined_output, grad_expert_output)
        return (grad_gates1_s, grad_expert_output)


class MOELayer(torch.nn.Module):

    def __init__(self, gate_type, experts: Union[torch.nn.Module, ModuleList], model_dim: int, group: Optional[Any] = None):
        super().__init__()
        self.world_size = get_world_size()

        if gate_type == 'Top1Gate':
            gating = Top1Gate
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

    def forward(self, input: Tensor, **kwargs: Any):
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"

        load_kernels(input.dtype)

        model_dim = input.shape[2]
        reshaped_input = input.reshape(-1, model_dim)

        l_aux, _CustomEncoder._cache = self.gate(reshaped_input)

        S, M = reshaped_input.size(0), reshaped_input.size(1) 
        E, C = _CustomEncoder._cache[4], _CustomEncoder._cache[1]

        if not hasattr(self, 'ones_helper'):
            self.ones_helper = torch.ones(_CustomEncoder._cache[3].size(), dtype=_CustomEncoder._cache[3].dtype, device=_CustomEncoder._cache[3].device)
        else:
            assert self.ones_helper.size() == _CustomEncoder._cache[3].size()

        dispatched_input = _CustomEncoder.apply(self.ones_helper, reshaped_input)
        dispatched_input = _AllToAll.apply(self.expert_group, dispatched_input)
        
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, model_dim)

        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.expert_group, expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, model_dim)

        # einsum("sec,ecm->sm")
        combined_output = _CustomDecoder.apply(_CustomEncoder._cache[3], expert_output.view(E*C, M))
        
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input.shape[0], :, :]
        combined_output.l_aux = l_aux
        return combined_output

    def prepare_for_inference_(self):
        self.in_generation = True
