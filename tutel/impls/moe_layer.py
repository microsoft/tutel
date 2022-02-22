# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

from ..impls.fast_dispatch import fast_dispatcher
from ..jit_kernels.gating import fast_cumsum_sub_one
from ..impls import communicate as C


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result

def load_balance(gates, mask1, num_global_experts, fp32_gate):
    if gates.dtype == torch.float32 or fp32_gate:
        me = torch.sum(gates.float(), dim=0)
        ce = torch.sum(mask1.to(me.dtype), dim=0)
        l_loss = torch.sum(me * ce) * (num_global_experts / (gates.size(0) * gates.size(0)))
    else:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.to(gates.dtype), dim=0)
        l_loss = torch.sum(me * ce) * num_global_experts
    return l_loss


class TopKGate(torch.nn.Module):
    """General-purpose Top-K Gate for MoE
    """
 
    def __init__(
        self,
        model_dim,
        num_global_experts,
        a2a_ffn_overlap_degree=1,
        capacity_factor=1.0,
        top_k=2,
        batch_prioritized_routing=False,
        **kwargs,
    ):
        super().__init__()
        top_k = min(top_k, num_global_experts)
        self.top_k = top_k
        assert self.top_k > 0, "Top-k value %d is not valid." % self.top_k

        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)

        self.fp32_gate = kwargs.get('fp32_gate', False)
        if self.fp32_gate:
          self.wg = self.wg.float()

        self.capacity_factor = float(os.environ.get('CAP_FACTOR', capacity_factor))
        self.is_ones_gate = (int(os.environ.get('ONES_GATE', 0)) == 1)
        self.num_global_experts = num_global_experts

        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True

        self.is_postnorm = kwargs.get('is_postnorm', True)
        input_dropout_p = kwargs.get('input_dropout_p', 0)
        self.input_dropout = torch.nn.Dropout(p=input_dropout_p) if input_dropout_p else None

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2d_a2a = os.environ.get('TUTEL_ALLTOALL_ALGO', '').upper() == '2D'

    def compute_sorted_location(self, x, importance_scores):
        sorted_x = x[importance_scores.argsort(dim=0)]
        sorted_cumsum = fast_cumsum_sub_one(sorted_x) * sorted_x
        return sorted_cumsum[importance_scores.argsort(dim=0).argsort(dim=0)]

    def apply_on_expert_fn(self, input, expert_fn, group, sharded_count):
        if self.input_dropout:
            input = self.input_dropout(input)

        logits = self.wg(input.to(next(iter(self.wg.parameters())).dtype))

        topk_indices = torch.topk(logits, self.top_k, dim=1).indices

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.top_k, dim=1)]
        masks_se = [one_hot_with_dtype(x, num_classes=self.num_global_experts, dtype=x.dtype) for x in indices_s]

        gates = F.softmax(logits, dim=1)
        gates_s = [(gates * x).sum(dim=1) for x in masks_se]

        l_loss = load_balance(gates, masks_se[0], self.num_global_experts, self.fp32_gate)

        if self.batch_prioritized_routing:
            importance_scores = -1 * gates.max(dim=1)[0]
            self.compute_location = lambda x: self.compute_sorted_location(x, importance_scores)
        else:
            self.compute_location = fast_cumsum_sub_one

        locations1 = self.compute_location(masks_se[0])

        locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

        if self.top_k > 1:
          acc_base = None

          for k in range(1, self.top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = self.compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

          # Normalize Gate
          denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
          gates_s = [x / denom_s for x in gates_s]

        indices_s = [x.to(torch.int32) for x in indices_s]

        S, M, GE = input.size(0), input.size(1), self.num_global_experts
        world_size = C.get_world_size(group)

        if not hasattr(self, '_fdr'):
            capacity = self.top_k * int(self.capacity_factor * ((S + self.num_global_experts - 1) // self.num_global_experts))
            self._fdr = fast_dispatcher(num_global_experts=GE, capacity=capacity, model_dim=M, dispatch_dtype=input.dtype)
        else:
            capacity = self._fdr.capacity

        if self.is_ones_gate:
            gates_s = [torch.ones_like(x) for x in gates_s]
        self._fdr.update(indices_s, locations_s, gates_s, capacity=capacity, is_postnorm=self.is_postnorm)

        dispatched_input = self._fdr.encode(input)
        # dispatched_input = dispatched_input.repeat(sharded_count, 1)
        if sharded_count > 1:
            dispatched_input = dispatched_input.reshape(world_size, 1, -1, M)
        else:
            dispatched_input = dispatched_input.reshape(world_size, -1, capacity, M)

        if self.a2a_ffn_overlap_degree == -1:
            expert_output = expert_fn(dispatched_input)
            expert_output = expert_output.to(input.dtype)
        elif self.a2a_ffn_overlap_degree == 1:
            if self.use_2d_a2a:
                C.AllToAllStatus.init(group, 1, -1)
                dispatched_input = \
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(
                                    C.CurrentStreamRelease.apply(dispatched_input, 0), 0)), 0), 0)
            else:
                dispatched_input = C.AllToAll.apply(group, dispatched_input)

            expert_output = expert_fn(dispatched_input)
            expert_output = expert_output.to(input.dtype)

            if self.use_2d_a2a:
                expert_output = \
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(
                                    C.CurrentStreamRelease.apply(expert_output, 0), 0)), 0), 0)
            else:
                expert_output = C.AllToAll.apply(group, expert_output)
        else:
            split_dim = 2
            C.AllToAllStatus.init(group, self.a2a_ffn_overlap_degree, split_dim)

            # Implicit x.contiguous() in CurrentStreamRelease.forward() and CurrentStreamAcquire.backward()
            if self.use_2d_a2a:
                split_size = dispatched_input.shape[split_dim] // self.a2a_ffn_overlap_degree
                dispatched_input_split = dispatched_input.split(split_size, dim=split_dim)
                dispatched_input_scattered_after_a2a = [
                    C.NcclStreamRelease.apply(
                        C.AllToAll2DAsync.apply(
                            C.NcclStreamAcquire.apply(
                                C.CurrentStreamRelease.apply(x, i), i)), i)
                    for i, x in enumerate(dispatched_input_split)]
            else:
                dispatched_input_ready = C.CurrentStreamRelease.apply(dispatched_input, 0)
                dispatched_input_scattered_after_a2a = C.AllToAllScatterAsync.apply(dispatched_input_ready)

            expert_output_scattered = [
                C.CurrentStreamRelease.apply(expert_fn(C.CurrentStreamAcquire.apply(x, i)).to(input.dtype), i)
                for i, x in enumerate(dispatched_input_scattered_after_a2a)]

            if self.use_2d_a2a:
                expert_output_gathered_after_a2a = [
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(x, i)), i), i)
                    for i, x in enumerate(expert_output_scattered)]
                expert_output = torch.cat(expert_output_gathered_after_a2a, dim=split_dim)
            else:
                expert_output_gathered_after_a2a = C.AllToAllGatherAsync.apply(*expert_output_scattered)
                expert_output = C.CurrentStreamAcquire.apply(expert_output_gathered_after_a2a, 0)

        expert_output = expert_output.reshape(-1, GE, capacity, M)
        if expert_output.size(0) > 1:
            expert_output = torch.sum(expert_output, dim=0)
        expert_output = expert_output.view(GE * self._fdr.capacity, M)

        result_output = self._fdr.decode(expert_output)
        return result_output, l_loss


class MegatronLMGate(torch.nn.Module):
    """Megatron-LM Tensor Parallel over MoE Gate Type
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.l_zero = None
        self._modules = dict()
        self._parameters = dict()
        self._buffers = dict()

    def named_parameters(self):
        return []

    def apply_on_expert_fn(self, input, expert_fn, group, sharded_count):
        if self.l_zero is None:
            self.l_zero = torch.tensor(0, dtype=input.dtype, device=input.device)
        assert sharded_count == 1
        gathered_input = C.PreAllreduceSum.apply(group, input)
        result_output = expert_fn(gathered_input)
        result_output = C.PostAllreduceSum.apply(group, result_output)
        return result_output, self.l_zero


class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """

    def __init__(self, gate_type, model_dim: int, experts = None, scan_expert_func = None, result_func = None, group: Optional[Any] = None, seeds = None, a2a_ffn_overlap_degree = 1, **kwargs):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        if not isinstance(experts, dict):
            self.num_local_experts = len(self.experts)
        else:
            self.num_local_experts = experts.get('count_per_node', 1)
            if not isinstance(self.num_local_experts, int):
                self.num_local_experts = -int(1 / (self.num_local_experts + 1e-5))

        self.ffn_zero_group = None
        num_devices = C.get_world_size(self.group)
        if self.num_local_experts < 0:
            sharded_count = -self.num_local_experts
            assert num_devices >= sharded_count, f"Expected to use {sharded_count} devices to maintain 1 expert, while the number of global devices is only {num_devices}"
            assert num_devices % sharded_count == 0, f"Cannot evenly divide {num_devices} global devices by sharded experts each of whose slice count = {sharded_count}."
            assert experts['hidden_size_per_expert'] % sharded_count == 0, f"Cannot evenly divide hidden_size_per_expert ({experts['hidden_size_per_expert']}) to {sharded_count} slices."
            self.num_global_experts = num_devices // sharded_count
            self.num_local_experts, experts['hidden_size_per_expert'] = 1, experts['hidden_size_per_expert'] // sharded_count
            self.sharded_count = sharded_count
            self.ffn_zero_group = C.create_groups_from_world(group_count=self.num_global_experts).model_group
        else:
            self.num_global_experts = num_devices * self.num_local_experts
            sharded_count = 1

        self.model_dim = model_dim
        self.sharded_count = sharded_count

        if not isinstance(experts, dict):
            self.experts = cast(ModuleList, experts) if type(experts) == ModuleList else ModuleList(experts)
        else:
            experts = copy.deepcopy(experts)
            if experts['type'] == 'attention':
                experts['type'] = 'ffn'
                experts['activation_fn'] = experts['attention_fn']
                experts['hidden_size_per_expert'] = model_dim

            if experts['type'] == 'ffn':
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
                    output[W, E, C, M]  =  hidden[E, W, C, M]
                '''

                fused_custom_fn = experts.get('fused_custom_fn')
                if fused_custom_fn is None:
                    activation_fn = experts.get('activation_fn', lambda x: F.relu(x))
                implicit_dropout_p = experts.get('implicit_dropout_p', 0)

                class FusedExpertsNetwork(torch.nn.Module):
                    def __init__(self, model_dim, hidden_size, local_experts, ffn_zero_group):
                        super().__init__()
                        self.skip_expert = (int(os.environ.get('SKIP_EXPERT', '0')) != 0)

                        fc1_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc2_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
                        fc2_bias = torch.empty(1, local_experts, 1, (model_dim + sharded_count - 1) // sharded_count)

                        for i in range(local_experts):
                            fc1 = torch.nn.Linear(model_dim, hidden_size)
                            fc2 = torch.nn.Linear(hidden_size, model_dim)
                            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = fc1.weight, fc1.bias
                            fc2_weight[0, i, :, :], fc2_bias[0, i, :, :] = fc2.weight.t(), fc2.bias[:fc2_bias.size(-1)]

                        self.model_dim, self.hidden_size, self.local_experts = model_dim, hidden_size, local_experts
                        self.ffn_zero_group = ffn_zero_group
                        if self.ffn_zero_group is not None:
                            assert self.local_experts == 1
                            fc1_weight = fc1_weight.view(self.hidden_size, self.model_dim)
                            fc2_weight = fc2_weight.view(self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.hidden_size)
                            fc2_bias = fc2_bias.view(-1)
                        elif self.local_experts == 1:
                            fc1_weight = fc1_weight.view(self.hidden_size, self.model_dim)
                            fc2_weight = fc2_weight.view(self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.hidden_size)
                            fc2_bias = fc2_bias.view(-1)
                        else:
                            fc1_weight = fc1_weight.view(self.local_experts, self.hidden_size, self.model_dim)
                            fc2_weight = fc2_weight.view(self.local_experts, self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.local_experts, 1, self.hidden_size)
                            fc2_bias = fc2_bias.view(self.local_experts, 1, -1)

                        self.register_parameter(name='fc1_weight', param=torch.nn.Parameter(fc1_weight))
                        self.register_parameter(name='fc2_weight', param=torch.nn.Parameter(fc2_weight))
                        self.register_parameter(name='fc1_bias', param=torch.nn.Parameter(fc1_bias))
                        self.register_parameter(name='fc2_bias', param=torch.nn.Parameter(fc2_bias))

                        if implicit_dropout_p:
                            self.dropout_fc1 = torch.nn.Dropout(p=implicit_dropout_p)
                            self.dropout_fc2 = torch.nn.Dropout(p=implicit_dropout_p)
                        else:
                            self.dropout_fc1 = self.dropout_fc2 = lambda x: x

                    def extra_repr(self):
                        return 'model_dim=%d, hidden_size=%d, local_experts=%d, bias=%s' % (self.model_dim, self.hidden_size, self.local_experts, self.fc1_bias is not None)

                    def forward(self, x):
                        if self.skip_expert:
                            return x
                        if fused_custom_fn is not None:
                            return fused_custom_fn(self, x)

                        fc1_weight, fc2_weight, fc1_bias, fc2_bias = self.fc1_weight, self.fc2_weight, self.fc1_bias, self.fc2_bias
                        if self.ffn_zero_group is not None:
                            fc1_weight = C.PreAllreduceSum.apply(self.ffn_zero_group, self.fc1_weight)
                            fc2_weight = C.PreAllreduceSum.apply(self.ffn_zero_group, self.fc2_weight)
                            fc1_bias = C.PreAllreduceSum.apply(self.ffn_zero_group, self.fc1_bias)
                            fc2_bias = C.PreAllreduceSum.apply(self.ffn_zero_group, self.fc2_bias)
                            if fc2_bias.size(-1) != self.model_dim:
                                fc2_bias = fc2_bias[:, :self.model_dim]

                        if self.local_experts == 1:
                            original_shape, x = x.shape, x.view(-1, self.model_dim)
                            x = torch.addmm(fc1_bias.unsqueeze(0), x, fc1_weight.t())
                            x = activation_fn(x.unsqueeze(0)).squeeze(0)
                            x = self.dropout_fc1(x)
                            x = torch.addmm(fc2_bias.unsqueeze(0), x, fc2_weight)
                            x = self.dropout_fc2(x)
                            x = x.view(original_shape)
                        else:
                            x = x.permute(1, 0, 2, 3)
                            original_shape, x = x.shape, x.reshape(self.local_experts, -1, self.model_dim)
                            x = torch.matmul(x, fc1_weight.swapaxes(1, 2)) + fc1_bias
                            x = activation_fn(x)
                            x = self.dropout_fc1(x)
                            x = torch.matmul(x, fc2_weight) + fc2_bias
                            x = self.dropout_fc2(x)
                            x = x.reshape(self.local_experts, original_shape[1], original_shape[2], self.model_dim)
                            x = x.permute(1, 0, 2, 3)
                        return x

                    def to(self, *args, **kwargs):
                        self = super().to(*args, **kwargs)
                        self.fc1_weight = self.fc1_weight.to(*args, **kwargs)
                        self.fc2_weight = self.fc2_weight.to(*args, **kwargs)
                        self.fc1_bias = self.fc1_bias.to(*args, **kwargs)
                        self.fc2_bias = self.fc2_bias.to(*args, **kwargs)
                        return self

                if seeds is not None and seeds[1] is not None:
                    torch.manual_seed(seeds[1])
                self.experts = ModuleList([FusedExpertsNetwork(model_dim, experts['hidden_size_per_expert'], self.num_local_experts, self.ffn_zero_group)])
            else:
                raise Exception('Builtin expert type is not recognized: %s' % experts['type'])

        if scan_expert_func is not None:
            for expert in self.experts:
                for n, p in expert.named_parameters():
                    scan_expert_func(n, p)

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in tutel.moe_layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            if single_gate_type['type'] == 'top':
                if seeds is not None and seeds[0] is not None:
                    torch.manual_seed(seeds[0] + gi)
                if "fp32_gate" in kwargs:
                    logging.warning(f'`fp32_gate` option in tutel.moe_layer has been deprecated, please move this option to gate_type = {{.., "fp32_gate": {kwargs["fp32_gate"]}}} instead.')
                    single_gate_type["fp32_gate"] = kwargs["fp32_gate"]

                self.gates += [TopKGate(model_dim=model_dim, top_k=single_gate_type['k'], num_global_experts=self.num_global_experts, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, **single_gate_type)]
            elif single_gate_type['type'] == 'megatron':
                self.gates += [MegatronLMGate(**single_gate_type)]
                assert isinstance(experts, dict), "Gate type `megatron` requires dict-type expert description."
                assert self.num_local_experts == 1, "Gate type `megatron` requires `count_per_node` == 1 in expert attributions."
                assert experts['type'] == 'ffn', "Gate type `megatron` requires `type` == `ffn` in expert attributions."
            else:
                raise Exception("Unrecognized gate_type: %s" % single_gate_type)

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

        def expert_fn(dispatched_input):
            if len(self.experts) == 1:
                expert_output = self.experts[0](dispatched_input)
            else:
                chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
                expert_output = torch.cat([expert(chunk) for chunk, expert in zip(chunks, self.experts)], dim=1)
            return expert_output

        self.expert_fn = expert_fn
        self.expected_sample_size = 0

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def forward(self, input: Tensor, gate_index=0, **kwargs: Any):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(input.shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        reshaped_input = input.reshape(-1, input.shape[-1])
        reshaped_input_samples = reshaped_input.shape[0]

        self.expected_sample_size = self.expected_sample_size or reshaped_input.size(0)
        if reshaped_input.size(0) != self.expected_sample_size:
            if reshaped_input.size(0) > self.expected_sample_size:
                raise Exception('MoE JIT is designed to work on sample size = %s, while receiving sample size = %s (> %s)' % (self.expected_sample_size, reshaped_input.size(0), self.expected_sample_size))
            else:
                if C.get_world_rank(self.group) == 0:
                    logging.warning('MoE is initialized to keep working on sample size = %s, while receiving sample size = %s (will slow down this forward step)' % (self.expected_sample_size, reshaped_input.size(0)))
                pad_input = torch.zeros([self.expected_sample_size, self.model_dim], dtype=reshaped_input.dtype, layout=reshaped_input.layout, device=reshaped_input.device)
                pad_input[:reshaped_input.size(0)] = reshaped_input
                reshaped_input = pad_input

        reshaped_input = reshaped_input.to(next(iter(self.experts.parameters())).dtype)
        result_output, l_aux = self.gates[gate_index].apply_on_expert_fn(reshaped_input, self.expert_fn, self.group, sharded_count=self.sharded_count)

        result_output = result_output[:reshaped_input_samples, :]
        result_output = result_output.view(original_shape).to(original_dtype)
        self.l_aux = result_output.l_aux = l_aux
        return self.result_func(result_output) if self.result_func is not None else result_output

moe_layer = MOELayer
