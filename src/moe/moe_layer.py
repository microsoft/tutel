from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

def shared_data():
    pass

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

def load_kernels():
    global JitKernels
    try:
        return JitKernels
    except:
      if shared_data.message_dtype == torch.float16:
          from .jit_kernels import sparse_fp16 as jit_kernel
      else:
          from .jit_kernels import sparse_fp32 as jit_kernel
      JitKernels = jit_kernel
      return JitKernels


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


class Top1Gate(torch.nn.Module):

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        capacity_factor=1.0,
        allow_approximation=False,
    ):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.allow_approximation = allow_approximation

    def forward(self, input: torch.Tensor):
        logits = self.wg(input)
        num_tokens, num_experts = logits.shape[0], logits.shape[1]

        if not hasattr(self, 'gating_kernel'):
            from .jit_kernels.gating import get_gating_kenel
            self.gating_kernel = get_gating_kenel(num_tokens, num_experts)

        capacity = int(self.capacity_factor * ((num_tokens + num_experts - 1) // num_experts))

        indices1_s = torch.argmax(logits, dim=1)
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        gates = F.softmax(logits, dim=1)
        gates1_s = (gates * mask1).sum(dim=1)

        locations1_s = torch.empty([num_tokens,], dtype=torch.int32, device=logits.device).contiguous()
        self.gating_kernel(indices1_s.to(torch.int32).contiguous(), locations1_s)

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

        return l_aux, indices1_s.to(torch.int32), capacity, locations1_s.to(torch.int32), gates1_s, num_experts


class Top2Gate(torch.nn.Module):
 
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        capacity_factor=1.0,
        allow_approximation=False,
    ):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.allow_approximation = allow_approximation

    def forward(self, input: torch.Tensor):
        raise Exception('Not implemented')  ## FIXME: not implemented


class _CustomEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, reshaped_input: Tensor):
        ctx.reshaped_input = reshaped_input
        ctx.gates1_s = gates1_s

        sc_size = shared_data.num_experts * shared_data.capacity
        assert sc_size == 2048
        assert reshaped_input.size(1) == 2048

        dispatched_input = torch.zeros([sc_size, reshaped_input.size(1)], dtype=reshaped_input.dtype, device=reshaped_input.device)
        JitKernels.func_fwd(gates1_s, shared_data.indices1_s, shared_data.locations1_s, reshaped_input, dispatched_input)
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor):
        # grad_gates1_s = None
        # grad_gates1_s = torch.empty([ctx.reshaped_input.size(0)], dtype=dispatched_input.dtype, device=dispatched_input.device)
        # JitKernels.func_bwd_gate(dispatched_input, shared_data.indices1_s, shared_data.locations1_s, ctx.reshaped_input, grad_gates1_s)

        assert len(ctx.reshaped_input.shape) == 2
        assert ctx.reshaped_input.shape[0] == 2048
        assert ctx.reshaped_input.shape[1] == 2048

        grad_reshaped_input = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
        JitKernels.func_bwd_data(dispatched_input, ctx.gates1_s, shared_data.indices1_s, shared_data.locations1_s, grad_reshaped_input)
        return (None, grad_reshaped_input)


class _CustomDecoder(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, gates1_s: Tensor, expert_output: Tensor):
        ctx.expert_output = expert_output
        ctx.gates1_s = gates1_s

        assert gates1_s.size(0) == 2048
        assert expert_output.size(1) == 2048

        combined_output = torch.empty([gates1_s.size(0), expert_output.size(1)], dtype=gates1_s.dtype, device=gates1_s.device)
        JitKernels.func_bwd_data(expert_output, ctx.gates1_s, shared_data.indices1_s, shared_data.locations1_s, combined_output)
        return combined_output
        
    @staticmethod
    def backward(ctx: Any, combined_output: Tensor):
        assert len(shared_data.indices1_s.shape) == 1
        assert shared_data.indices1_s.shape[0] == 2048
        grad_gates1_s = torch.empty(shared_data.indices1_s.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_bwd_gate(ctx.expert_output, shared_data.indices1_s, shared_data.locations1_s, combined_output, grad_gates1_s)

        assert len(ctx.expert_output.shape) == 2
        assert ctx.expert_output.shape[0] == 2048
        assert ctx.expert_output.shape[1] == 2048
        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        JitKernels.func_fwd(ctx.gates1_s, shared_data.indices1_s, shared_data.locations1_s, combined_output, grad_expert_output)
        return (grad_gates1_s, grad_expert_output)


class MOELayer(torch.nn.Module):

    def __init__(self, gate_type, model_dim: int, builtin_experts = None, external_experts = None, allow_approximation = False, group: Optional[Any] = None):
        super().__init__()

        self.expert_group = group = group if group is not None else dist.group.WORLD
        self.world_size = get_world_size(self.expert_group)

        if gate_type == 'Top1Gate':
            gating = Top1Gate
        elif gate_type == 'Top2Gate':
            gating = Top2Gate
        else:
            raise Exception(f"Unrecognized gate_type: {gate_type}")

        if external_experts is not None:
            self.experts = cast(ModuleList, external_experts) if type(external_experts) == ModuleList else ModuleList(external_experts)
            self.num_local_experts = len(self.experts)
        elif builtin_experts is not None:
            network_type = builtin_experts['type']
            if network_type == 'ffn':
                ''' << Fused FFN Experts >>

                    hidden[W, E, C, V] +=! input[W, E, C, M] x expert_fc1[0, E, M, V]
                    hidden[W, E, C, V]  =  hidden[W, E, C, V] + bias_fc1[E, V]
                    hidden[W, E, C, V]  =  activation_fn(hidden[W, E, C, V])
                    hidden[W, E, C, M] +=! hidden[W, E, C, V] x expert_fc2[0, E, V, M]
                    output[W, E, C, M]  =  hidden[W, E, C, M] + bias_fc2[E, M]
                '''

                self.num_local_experts = builtin_experts.get('count_per_node', 1)
                activation_fn = builtin_experts.get('activation_fn', lambda x: x)

                class FusedExpertsNetwork(torch.nn.Module):
                    def __init__(self, model_dim, hidden_size, local_experts):
                        super().__init__()

                        fc1_weight = torch.empty(1, local_experts, model_dim, hidden_size)
                        fc2_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
                        fc2_bias = torch.empty(1, local_experts, 1, model_dim)

                        for i in range(local_experts):
                            fc1 = torch.nn.Linear(model_dim, hidden_size)
                            fc2 = torch.nn.Linear(hidden_size, model_dim)
                            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = fc1.weight.t(), fc1.bias
                            fc2_weight[0, i, :, :], fc2_bias[0, i, :, :] = fc2.weight.t(), fc2.bias

                        self.register_parameter(name='fc1_weight', param=torch.nn.Parameter(fc1_weight))
                        self.register_parameter(name='fc2_weight', param=torch.nn.Parameter(fc2_weight))
                        self.register_parameter(name='fc1_bias', param=torch.nn.Parameter(fc1_bias))
                        self.register_parameter(name='fc2_bias', param=torch.nn.Parameter(fc2_bias))

                    def forward(self, x):
                        x = torch.matmul(x, self.fc1_weight.repeat(x.shape[0], 1, 1, 1) if x.shape[0] > 1 else self.fc1_weight) + self.fc1_bias
                        x = activation_fn(x)
                        x = torch.matmul(x, self.fc2_weight.repeat(x.shape[0], 1, 1, 1) if x.shape[0] > 1 else self.fc2_weight) + self.fc2_bias
                        return x

                    def to(self, *args, **kwargs):
                        self = super().to(*args, **kwargs)
                        self.fc1_weight = self.fc1_weight.to(*args, **kwargs)
                        self.fc2_weight = self.fc2_weight.to(*args, **kwargs)
                        self.fc1_bias = self.fc1_bias.to(*args, **kwargs)
                        self.fc2_bias = self.fc2_bias.to(*args, **kwargs)
                        return self

                self.experts = ModuleList([FusedExpertsNetwork(model_dim, builtin_experts['hidden_size_per_expert'], self.num_local_experts)])
            else:
                raise Exception(f'Builtin expert type is not recognized: {network_type}')

        else:
            raise Exception("You must specify either `builtin_experts` or `external_experts` for MoE layer.")

        for expert in self.experts:
            for p in expert.parameters():
                p.expert = True

        self.gate = gating(model_dim=model_dim, num_experts=self.world_size * self.num_local_experts, allow_approximation=allow_approximation)
        self.in_generation = False

    def forward(self, input: Tensor, **kwargs: Any):
        original_shape  = input.shape
        if input.shape == 2:
            input = input.view(input.shape[0], 1, input.shape[1])
        assert len(input.shape) == 3, "input Tensor must be 2D/3D format: (s)equence, (t)oken [Optional], (m)odel"
        reshaped_input = input.reshape(-1, input.shape[2])

        if not hasattr(self, 'ones_gates1_s'):
            self.ones_gates1_s = torch.ones([reshaped_input.size(0),], dtype=input.dtype, device=input.device)
        else:
            assert self.ones_gates1_s.size(0) == reshaped_input.size(0), f"Did you have changed the batch_size of input? Expect {self.ones_gates1_s.size(0)}, get {reshaped_input.size(0)}. Please do padding to keep it constantly within one session."

        l_aux, shared_data.indices1_s, shared_data.capacity, shared_data.locations1_s, shared_data.gates1_s, shared_data.num_experts = self.gate(reshaped_input)
        shared_data.message_dtype = input.dtype

        load_kernels()

        S, M = reshaped_input.size(0), reshaped_input.size(1) 
        E, C = shared_data.num_experts, shared_data.capacity

        dispatched_input = _CustomEncoder.apply(self.ones_gates1_s, reshaped_input)
        dispatched_input = _AllToAll.apply(self.expert_group, dispatched_input)
        
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, M)

        if len(self.experts) == 1:
            expert_output = self.experts[0](dispatched_input)
        else:
            chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
            expert_outputs = [expert(chunk) for chunk, expert in zip(chunks, self.experts)]
            expert_output = torch.cat(expert_outputs, dim=1)

        expert_output = _AllToAll.apply(self.expert_group, expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, M)

        # einsum("sec,ecm->sm")
        combined_output = _CustomDecoder.apply(shared_data.gates1_s, expert_output.view(E*C, M))
        
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input.shape[0], :, :]
        combined_output = combined_output.view(original_shape)
        combined_output.l_aux = l_aux
        return combined_output

    def prepare_for_inference_(self):
        self.in_generation = True
