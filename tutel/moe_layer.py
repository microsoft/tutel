# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

try:
    import tutel_custom_kernel
except:
    raise Exception("Cannot import JIT optimized kernels. Did you forget to install Custom Kernel Extension?")


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

def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor):
        ctx.group = group
        ctx.world_size = get_world_size(group)
        if ctx.world_size <= 1 or AllToAll.skip_a2a:
            return input
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        if ctx.world_size <= 1 or AllToAll.skip_a2a:
            return (None, grad_output)
        return (None, AllToAll.apply(ctx.group, grad_output))


class ExpertsGemm(torch.autograd.Function):
    # output[W, E, C, V] +=! data[W, E, C, M] x weight[0, E, M, V]
    @staticmethod
    def forward(ctx: Any, data: Tensor, weight: Tensor):
        ctx.data = data
        ctx.weight = weight

        output = torch.empty([data.size(0), data.size(1), data.size(2), weight.size(3)], dtype=data.dtype, device=data.device)
        tutel_custom_kernel.experts_gemm_forward([data, weight, output], 1)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        grad_weight = torch.empty(ctx.weight.size(), dtype=grad_output.dtype, device=grad_output.device)
        tutel_custom_kernel.experts_gemm_backward_weight([ctx.data, grad_output, grad_weight], 1)

        grad_data = torch.empty(ctx.data.size(), dtype=grad_output.dtype, device=grad_output.device)
        tutel_custom_kernel.experts_gemm_backward_data([grad_output, ctx.weight, grad_data], 1)
        return (grad_data, grad_weight)

def load_balance(gates, mask1, num_global_experts, use_fp32):
    if gates.dtype == torch.float32 or use_fp32:
        me = torch.sum(gates.float(), dim=0)
        ce = torch.sum(mask1.to(me.dtype), dim=0)
        l_loss = torch.sum(me * ce) * (num_global_experts / (gates.size(0) * gates.size(0)))
    else:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.to(gates.dtype), dim=0)
        l_loss = torch.sum(me * ce) * num_global_experts
    return l_loss


class Top1Gate(torch.nn.Module):

    def __init__(
        self,
        model_dim,
        num_global_experts,
        capacity_factor=1.0,
        use_fp32=False,
    ):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.use_fp32 = use_fp32
        self.num_global_experts = num_global_experts

    def capacity(self, expected_sample_size):
        if not hasattr(self, 'capacity_int'):
            self.capacity_int = int(self.capacity_factor * ((expected_sample_size + self.num_global_experts - 1) // self.num_global_experts))
        return self.capacity_int

    def forward(self, input: torch.Tensor):
        logits = self.wg(input)

        if not hasattr(self, 'tutel_cumsum'):
            from .jit_kernels.gating import get_cumsum_kernel
            self.tutel_cumsum = get_cumsum_kernel(logits.size(0), self.num_global_experts)

        indices1_s = torch.argmax(logits, dim=1)
        mask1 = one_hot_with_dtype(indices1_s, num_classes=self.num_global_experts, dtype=indices1_s.dtype)

        mask1_ = mask1.to(logits.dtype)
        gates = F.softmax(logits, dim=1)
        gates1_s = (gates * mask1_).sum(dim=1)
        l_loss = load_balance(gates, mask1_, self.num_global_experts, self.use_fp32)

        locations1 = self.tutel_cumsum(mask1)
        locations1_s = torch.sum(locations1 * mask1, dim=1).to(torch.int32)

        return l_loss, [gates1_s, ], [indices1_s.to(torch.int32), ], [locations1_s.to(torch.int32), ]


class Top2Gate(torch.nn.Module):
 
    def __init__(
        self,
        model_dim,
        num_global_experts,
        capacity_factor=1.0,
        use_fp32=False,
    ):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.use_fp32 = use_fp32
        self.num_global_experts = num_global_experts
        assert self.num_global_experts >= 2, "You have only 1 expert, while you are using a top-2 gate."

    def capacity(self, expected_sample_size):
        if not hasattr(self, 'capacity_int'):
            self.capacity_int = 2 * int(self.capacity_factor * ((expected_sample_size + self.num_global_experts - 1) // self.num_global_experts))
        return self.capacity_int

    def forward(self, input: torch.Tensor):
        logits = self.wg(input)

        if not hasattr(self, 'tutel_cumsum'):
            from .jit_kernels.gating import get_cumsum_kernel
            self.tutel_cumsum = get_cumsum_kernel(logits.size(0), self.num_global_experts)

        top2_indices = torch.topk(logits, 2, dim=1).indices
        indices1_s, indices2_s = top2_indices.chunk(2, dim=1)
        indices1_s, indices2_s = indices1_s.view(-1), indices2_s.view(-1)

        mask1 = one_hot_with_dtype(indices1_s, num_classes=self.num_global_experts, dtype=indices1_s.dtype)
        mask2 = one_hot_with_dtype(indices2_s, num_classes=self.num_global_experts, dtype=indices2_s.dtype)

        gates = F.softmax(logits, dim=1)
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        l_loss = load_balance(gates, mask1, self.num_global_experts, self.use_fp32)

        locations1 = self.tutel_cumsum(mask1)
        locations1_s = torch.sum(locations1 * mask1, dim=1).to(torch.int32)

        locations2 = self.tutel_cumsum(mask2)
        locations2 += torch.sum(mask1, dim=0, keepdim=True)
        locations2_s = torch.sum(locations2 * mask2, dim=1)

        # Normalize Gate
        denom_s = torch.clamp(gates1_s + gates2_s, min=torch.finfo(gates2_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

        return l_loss, [gates1_s, gates2_s], [indices1_s.to(torch.int32), indices2_s.to(torch.int32)], [locations1_s.to(torch.int32), locations2_s.to(torch.int32)]


class GatingEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, reshaped_input: Tensor):
        ctx.reshaped_input = reshaped_input
        ctx.config = config

        dispatched_input = torch.zeros([ctx.config.num_global_experts * ctx.config.capacity, ctx.config.model_dim], dtype=reshaped_input.dtype, device=reshaped_input.device)
        for i in range(len(ctx.config.indices_)):
          ctx.config.func_fwd(ctx.config.ones_helper, ctx.config.indices_[i], ctx.config.locations_[i], reshaped_input, dispatched_input)
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor):
        last_result = None
        for i in range(len(ctx.config.indices_)):
          grad_data = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
          ctx.config.func_bwd_data(ctx.config.ones_helper, dispatched_input, ctx.config.indices_[i], ctx.config.locations_[i], grad_data)
          last_result = grad_data if last_result is None else last_result + grad_data
        return (None, last_result)


class GatingDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, expert_output: Tensor, *gates_: Tensor):
        ctx.expert_output = expert_output
        ctx.gates_h2 = [x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x for x in gates_]
        ctx.config = config

        last_result = None
        for i in range(len(config.indices_)):
          single_output = torch.empty([config.expected_sample_size, config.model_dim], dtype=expert_output.dtype, device=expert_output.device)
          config.func_bwd_data(ctx.gates_h2[i], expert_output, config.indices_[i], config.locations_[i], single_output)
          last_result = single_output if last_result is None else last_result + single_output
        return last_result

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor):
        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        for i in range(len(ctx.config.indices_)):
          ctx.config.func_fwd(ctx.gates_h2[i], ctx.config.indices_[i], ctx.config.locations_[i], combined_output, grad_expert_output)

        grad_gates = []
        for i in range(len(ctx.config.indices_)):
          grad_gates1_s = torch.empty([ctx.config.expected_sample_size,], dtype=combined_output.dtype, device=combined_output.device)
          ctx.config.func_bwd_gate(ctx.expert_output, ctx.config.indices_[i], ctx.config.locations_[i], combined_output, grad_gates1_s)
          grad_gates.append(grad_gates1_s)
        return (None, grad_expert_output, *grad_gates)


class MOELayer(torch.nn.Module):

    def __init__(self, gate_type, model_dim: int, builtin_experts = None, external_experts = None, fp32_gate = False, scan_experts = None, result_func = None, group: Optional[Any] = None):
        super().__init__()

        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        self.expert_group = group = group if group is not None else dist.group.WORLD
        self.world_size = get_world_size(self.expert_group)
        self.result_func = result_func

        import os
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)
        AllToAll.skip_a2a = (int(os.environ.get('SKIP_A2A', '0')) != 0)

        if external_experts is not None:
            self.experts = cast(ModuleList, external_experts) if type(external_experts) == ModuleList else ModuleList(external_experts)
            self.num_local_experts = len(self.experts)
        elif builtin_experts is not None:
            network_type = builtin_experts['type']
            if network_type == 'ffn':
                ''' << Fused FFN Experts V1 >> (kernels = 5)

                    hidden[W, E, C, V] +=! input[W, E, C, M] x expert_fc1[0, E, M, V]
                    hidden[W, E, C, V]  =  hidden[W, E, C, V] + bias_fc1[E, V]
                    hidden[W, E, C, V]  =  activation_fn(hidden[W, E, C, V])
                    hidden[W, E, C, M] +=! hidden[W, E, C, V] x expert_fc2[0, E, V, M]
                    output[W, E, C, M]  =  hidden[W, E, C, M] + bias_fc2[E, M]

                    << Fused FFN Experts V2 >> (kernels = 7)

                    hidden[E, W, C, M]  =   input[W, E, C, M]
                    hidden[E, W, C, V] +=! hidden[E, W, C, M] x expert_fc1[0, E, M, V]
                    hidden[E, W, C, V]  =  hidden[E, W, C, V] + bias_fc1[E, V]
                    hidden[E, W, C, V]  =  activation_fn(hidden[E, W, C, V])
                    hidden[E, W, C, M] +=! hidden[E, W, C, V] x expert_fc2[0, E, V, M]
                    hidden[E, W, C, M]  =  hidden[E, W, C, M] + bias_fc2[E, M]
                    output[E, W, C, M]  =  hidden[E, W, C, M]
                '''

                self.num_local_experts = builtin_experts.get('count_per_node', 1)
                fused_custom_fn = builtin_experts.get('fused_custom_fn')
                if fused_custom_fn is None:
                    activation_fn = builtin_experts.get('activation_fn', lambda x: x)

                class FusedExpertsNetwork(torch.nn.Module):
                    def __init__(self, model_dim, hidden_size, local_experts):
                        super().__init__()
                        self.skip_expert = (int(os.environ.get('SKIP_EXPERT', '0')) != 0)

                        fc1_weight = torch.empty(1, local_experts, model_dim, hidden_size)
                        fc2_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
                        fc2_bias = torch.empty(1, local_experts, 1, model_dim)

                        for i in range(local_experts):
                            fc1 = torch.nn.Linear(model_dim, hidden_size)
                            fc2 = torch.nn.Linear(hidden_size, model_dim)
                            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = fc1.weight.t(), fc1.bias
                            fc2_weight[0, i, :, :], fc2_bias[0, i, :, :] = fc2.weight.t(), fc2.bias

                        self.model_dim, self.hidden_size, self.local_experts = model_dim, hidden_size, local_experts
                        self.expert_gemm_algo = int(os.environ.get('EXPERT_ALGO', '0'))

                        if self.local_experts == 1:
                            fc1_weight = fc1_weight.view(self.model_dim, self.hidden_size)
                            fc2_weight = fc2_weight.view(self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(-1, self.hidden_size)
                            fc2_bias = fc2_bias.view(-1, self.model_dim)
                        elif self.expert_gemm_algo == 0:
                            fc1_weight = fc1_weight.view(self.local_experts, self.model_dim, self.hidden_size)
                            fc2_weight = fc2_weight.view(self.local_experts, self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.local_experts, 1, self.hidden_size)
                            fc2_bias = fc2_bias.view(self.local_experts, 1, self.model_dim)

                        self.register_parameter(name='fc1_weight', param=torch.nn.Parameter(fc1_weight))
                        self.register_parameter(name='fc2_weight', param=torch.nn.Parameter(fc2_weight))
                        self.register_parameter(name='fc1_bias', param=torch.nn.Parameter(fc1_bias))
                        self.register_parameter(name='fc2_bias', param=torch.nn.Parameter(fc2_bias))

                    def extra_repr(self):
                        return 'model_dim=%d, hidden_size=%d, local_experts=%d' % (self.model_dim, self.hidden_size, self.local_experts)

                    def forward(self, x):
                        if self.skip_expert:
                            return x
                        if fused_custom_fn is not None:
                            x = fused_custom_fn(self, x)
                        elif self.local_experts == 1:
                            original_shape, x = x.shape, x.view(-1, self.model_dim)
                            x = torch.addmm(self.fc1_bias, x, self.fc1_weight)
                            x = activation_fn(x)
                            x = torch.addmm(self.fc2_bias, x, self.fc2_weight)
                            x = x.view(original_shape)
                        elif self.expert_gemm_algo == 0:
                            x = x.permute(1, 0, 2, 3)
                            original_shape, x = x.shape, x.reshape(self.local_experts, -1, self.model_dim)
                            x = torch.matmul(x, self.fc1_weight) + self.fc1_bias
                            x = activation_fn(x)
                            x = torch.matmul(x, self.fc2_weight) + self.fc2_bias
                            x = x.reshape(self.local_experts, original_shape[1], original_shape[2], self.model_dim)
                            x = x.permute(1, 0, 2, 3)
                        else:
                            x = ExpertsGemm.apply(x, self.fc1_weight) + self.fc1_bias
                            x = activation_fn(x)
                            x = ExpertsGemm.apply(x, self.fc2_weight) + self.fc2_bias
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
                raise Exception('Builtin expert type is not recognized: %s' % network_type)

        else:
            raise Exception("You must specify either `builtin_experts` or `external_experts` for MoE layer.")

        if scan_experts is not None:
            for expert in self.experts:
                for n, p in expert.named_parameters():
                    scan_experts(n, p)

        self.num_global_experts = self.world_size * self.num_local_experts
        self.model_dim = model_dim

        if gate_type == 'Top1Gate' or (gate_type == 'Top2Gate' and self.num_global_experts == 1):
            gating = Top1Gate
        elif gate_type == 'Top2Gate':
            gating = Top2Gate
        else:
            raise Exception("Unrecognized gate_type: %s" % gate_type)
        self.gate = gating(model_dim=model_dim, num_global_experts=self.num_global_experts, use_fp32=fp32_gate)

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gate.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def forward(self, input: Tensor, **kwargs: Any):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(input.shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        reshaped_input = input.reshape(-1, input.shape[-1])
        reshaped_input_samples = reshaped_input.shape[0]

        if not hasattr(self, 'expected_sample_size'):
            self._create_jit(reshaped_input.size(0))
        elif reshaped_input.size(0) != self.expected_sample_size:
            if reshaped_input.size(0) > self.expected_sample_size:
                raise Exception('MoE expects to keep working on sample size = %s, while receiving sample size = %s (> %s)' % (self.expected_sample_size, reshaped_input.size(0), self.expected_sample_size))
            else:
                print('MoE is initialized to keep working on sample size = %s, while receiving sample size = %s (will slow down this forward step)' % (self.expected_sample_size, reshaped_input.size(0)))
                pad_input = torch.zeros([self.expected_sample_size, self.model_dim], dtype=reshaped_input.dtype, layout=reshaped_input.layout, device=reshaped_input.device)
                pad_input[:reshaped_input.size(0)] = reshaped_input
                reshaped_input = pad_input

        reshaped_input = reshaped_input.to(self.params_dtype)
        l_aux, gates_, self.indices_, self.locations_ = self.gate(reshaped_input)

        S, M, GE, C = self.expected_sample_size, self.model_dim, self.num_global_experts, self.capacity

        if not hasattr(self, 'ones_helper'):
            self.ones_helper = torch.ones([self.expected_sample_size, 2], dtype=reshaped_input.dtype, device=reshaped_input.device)

        dispatched_input = GatingEncoder.apply(self, reshaped_input)
        dispatched_input = AllToAll.apply(self.expert_group, dispatched_input)
        
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, M)

        if len(self.experts) == 1:
            expert_output = self.experts[0](dispatched_input)
        else:
            chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
            expert_outputs = [expert(chunk) for chunk, expert in zip(chunks, self.experts)]
            expert_output = torch.cat(expert_outputs, dim=1)

        expert_output = AllToAll.apply(self.expert_group, expert_output)

        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, M)

        result_output = GatingDecoder.apply(self, expert_output.view(GE * C, M), *gates_)
        
        result_output = result_output[:reshaped_input_samples, :]
        result_output = result_output.view(original_shape).to(original_dtype)
        result_output.l_aux = l_aux

        return self.result_func(result_output) if self.result_func is not None else result_output

    def _create_jit(self, expected_sample_size):
        self.params_dtype = next(iter(self.experts.parameters())).dtype
        self.expected_sample_size = expected_sample_size
        self.capacity = self.gate.capacity(expected_sample_size)
        self.aligned_dim = self.model_dim // (2 if self.params_dtype == torch.float16 else 1)

        from .jit_kernels import sparse as jit_kernel
        self.func_fwd = jit_kernel.create_forward(expected_sample_size, self.num_global_experts, self.capacity, self.aligned_dim, self.params_dtype)
        self.func_bwd_data = jit_kernel.create_backward_data(expected_sample_size, self.num_global_experts, self.capacity, self.aligned_dim, self.params_dtype)
        self.func_bwd_gate = jit_kernel.create_backward_gate(expected_sample_size, self.num_global_experts, self.capacity, self.aligned_dim, self.params_dtype)
