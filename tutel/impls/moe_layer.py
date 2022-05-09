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

from ..impls import communicate as C
from ..impls.fast_dispatch import fast_encode, fast_decode, extract_critical


class FusedExpertsNetwork(torch.nn.Module):
    def __init__(self, model_dim, hidden_size, local_experts, sharded_count, activation_fn):
        super().__init__()
        self.skip_expert = (int(os.environ.get('SKIP_EXPERT', '0')) != 0)
        self.model_dim, self.hidden_size, self.local_experts = model_dim, hidden_size, local_experts
        self.activation_fn = activation_fn

        fc1_weight = torch.empty(1, local_experts, hidden_size, model_dim)
        fc2_weight = torch.empty(1, local_experts, hidden_size, model_dim)
        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
        fc2_bias = torch.empty(1, local_experts, 1, (model_dim + sharded_count - 1) // sharded_count)

        for i in range(local_experts):
            fc1 = torch.nn.Linear(model_dim, hidden_size)
            fc2 = torch.nn.Linear(hidden_size, model_dim)
            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = fc1.weight, fc1.bias
            fc2_weight[0, i, :, :], fc2_bias[0, i, :, :] = fc2.weight.t(), fc2.bias[:fc2_bias.size(-1)]

        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(fc1_weight.squeeze(0)))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(fc2_weight.squeeze(0)))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(fc1_bias.squeeze(0)))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(fc2_bias.squeeze(0)))

    def extra_repr(self):
        return 'model_dim=%d, hidden_size=%d, local_experts=%d' % (self.model_dim, self.hidden_size, self.local_experts)

    def forward(self, x, ctx):
        if self.skip_expert:
            return x

        batched_fc1_w = self.batched_fc1_w
        batched_fc2_w = self.batched_fc2_w
        batched_fc1_bias = self.batched_fc1_bias
        batched_fc2_bias = self.batched_fc2_bias

        if ctx.ffn_zero_group is not None:
            if not ctx.use_model_parallel:
                batched_fc1_w = C.zero_gather(batched_fc1_w, group=ctx.ffn_zero_group).view(1, -1, self.model_dim)
                batched_fc2_w = C.zero_gather(batched_fc2_w, group=ctx.ffn_zero_group).view(1, -1, self.model_dim)
                batched_fc1_bias = C.zero_gather(batched_fc1_bias, group=ctx.ffn_zero_group).view(1, 1, -1)

            batched_fc2_bias = C.zero_gather(batched_fc2_bias, group=ctx.ffn_zero_group)
            batched_fc2_bias = batched_fc2_bias.view(self.batched_fc2_bias.size(0), self.batched_fc2_bias.size(1), -1)
            if batched_fc2_bias.size(-1) != self.model_dim:
                batched_fc2_bias = batched_fc2_bias[:, :, :self.model_dim]

            if ctx.use_model_parallel:
                batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.sharded_count)

        y = torch.add(torch.matmul(x, batched_fc1_w.permute(0, 2, 1)), batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.add(torch.matmul(y, batched_fc2_w), batched_fc2_bias)
        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.fc1_weight = self.fc1_weight.to(*args, **kwargs)
        self.fc2_weight = self.fc2_weight.to(*args, **kwargs)
        self.fc1_bias = self.fc1_bias.to(*args, **kwargs)
        self.fc2_bias = self.fc2_bias.to(*args, **kwargs)
        return self


class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """
    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-5))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, "Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts

    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        is_postscore=True,
        batch_prioritized_routing=False,
        parallel_type='auto',
        use_2dh=False,
        **kwargs
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        if 'pad_samples' in kwargs:
            logging.warning(f"`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.")
            kwargs.pop('pad_samples')
        for k in kwargs:
            raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)

        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        assert isinstance(experts, dict), "Non-builtin experts module is not supported by standalone Tutel Moe-layer, please follows helloworld_from_scratch.py for custom construction instead."

        self.num_local_experts = experts.get('count_per_node', 1)
        self.num_global_experts = MOELayer.global_expert_count(self.num_local_experts, self.group)

        self.world_size = C.get_world_size(self.group)
        if self.num_global_experts < self.world_size:
            sharded_count = self.world_size // self.num_global_experts
            assert experts['hidden_size_per_expert'] % sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({experts['hidden_size_per_expert']}) to {sharded_count} slices"
            self.num_local_experts, experts['hidden_size_per_expert'] = 1, experts['hidden_size_per_expert'] // sharded_count
            self.ffn_zero_group = C.create_groups_from_world(group_count=self.num_global_experts).model_group
        else:
            sharded_count = 1
            self.ffn_zero_group = None

        if sharded_count == 1:
            self.auto_parallel, self.use_model_parallel = False, False
        elif parallel_type == 'auto':
            self.auto_parallel, self.use_model_parallel = True, False
        else:
            self.auto_parallel, self.use_model_parallel = False, (parallel_type == 'model')

        self.hidden_size = experts['hidden_size_per_expert']
        self.model_dim = model_dim
        self.sharded_count = sharded_count

        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if not isinstance(experts, dict):
            self.experts = cast(ModuleList, experts) if type(experts) == ModuleList else ModuleList(experts)
        else:
            if experts['type'] == 'ffn':
                activation_fn = experts.get('activation_fn', lambda x: F.relu(x))

                assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."
                assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

                if seeds is not None and seeds[1] is not None:
                    torch.manual_seed(seeds[1])
                self.experts = FusedExpertsNetwork(model_dim, self.hidden_size, self.num_local_experts, self.sharded_count, activation_fn)
            else:
                raise Exception('Builtin expert type is not recognized: %s' % experts['type'])

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            if single_gate_type['type'] == 'top':
                if seeds is not None and seeds[0] is not None:
                    torch.manual_seed(seeds[0] + gi)

                single_gate_type.pop('type')
                assert 'input_dropout_p' not in single_gate_type, "`input_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=input_dropout_p) before Tutel Moe-layer instead."

                # Create Gating Module
                fp32_gate = single_gate_type.get('fp32_gate', False)
                single_gate = torch.nn.Linear(model_dim, self.num_global_experts, bias=False, dtype=torch.float32 if fp32_gate else None)
                single_gate.fp32_gate = fp32_gate
                single_gate.capacity_factor = float(os.environ.get('CAP_FACTOR', single_gate_type.get('capacity_factor', 1.0)))
                single_gate.top_k = min(self.num_global_experts, int(single_gate_type.get('k', 2)))
                for k in single_gate_type:
                    if k not in ('fp32_gate', 'capacity_factor', 'k'):
                        raise Exception('Unrecognized argument provided to gate_type of Tutel Moe-layer: %s' % k)
                self.gates += [single_gate]
            else:
                raise Exception("Unrecognized gate_type: %s" % single_gate_type)

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])


    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def forward(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_overlap_degree=None):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype

        assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        reshaped_input = input.reshape(-1, input.size(-1))

        x = reshaped_input.to(next(iter(self.experts.parameters())).dtype)

        gctx = self.gates[gate_index]
        logits = gctx(x.to(next(iter(gctx.parameters())).dtype))
        scores = F.softmax(logits, dim=1)

        crit, l_aux = extract_critical(scores,
            top_k = gctx.top_k if top_k is None else top_k,
            capacity_factor = gctx.capacity_factor if capacity_factor is None else capacity_factor,
            fp32_gate = gctx.fp32_gate,
            batch_prioritized_routing = self.batch_prioritized_routing,
            alignment = self.sharded_count
        )

        y = fast_encode(x, crit, self.is_postscore)

        if self.auto_parallel:
            self.use_model_parallel = (y.numel() * (self.sharded_count - 1) < self.model_dim * self.hidden_size * self.sharded_count)

        if self.num_global_experts < self.world_size:
            if self.use_model_parallel:
                y = y.repeat(1, self.sharded_count, 1).view(self.world_size, -1, y.size(2))
            else:
                y = y.view(self.world_size, -1, y.size(2))

        a2a_overlap_degree = a2a_overlap_degree if a2a_overlap_degree is not None else self.a2a_ffn_overlap_degree
        # TODO: Overlap (only need to implement degree == 1 and degree > 1)
        if a2a_overlap_degree > 1:
            logging.warning(f"`a2a_overlap_degree` is currently not handled in this branch, please use `v0.1.x` instead.")

        y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh)
        y = self.experts(y, self)
        y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh)

        if self.num_global_experts < self.world_size:
            if self.use_model_parallel:
                y = torch.sum(y.view(self.num_global_experts, self.sharded_count, -1, y.size(2)), dim=1)
            else:
                y = y.view(self.num_global_experts, -1, y.size(2))

        y = fast_decode(y, crit, self.is_postscore)
        result_output = y

        '''
        group = ctx.group
        S, M, g_experts = input.size(0), input.size(1), self.num_global_experts
        world_size = C.get_world_size(group)

        if not hasattr(self, '_fdr'):
            self._fdr = fast_dispatcher(num_global_experts=g_experts, capacity=capacity, model_dim=M, dispatch_dtype=input.dtype)

        self._fdr.update(*crit[1:], is_postscore=self.is_postscore)

        dispatched_input = self._fdr.encode(input)

        if ctx.auto_parallel:
            ctx.use_model_parallel = (dispatched_input.numel() < ctx.model_dim * ctx.hidden_size)

        if ctx.use_model_parallel:
            dispatched_input = dispatched_input.reshape(g_experts, -1).repeat(1, ctx.sharded_count)

        if ctx.sharded_count > 1:
            dispatched_input = dispatched_input.reshape(world_size, 1, -1, M)
        else:
            dispatched_input = dispatched_input.reshape(world_size, -1, capacity, M)

        if self.a2a_ffn_overlap_degree == -1:
            expert_output = ctx.experts(dispatched_input, ctx)
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
                dispatched_input = C.all_to_all_single(dispatched_input, group=group)

            expert_output = ctx.experts(dispatched_input, ctx)
            expert_output = expert_output.to(input.dtype)

            if self.use_2d_a2a:
                expert_output = \
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(
                                    C.CurrentStreamRelease.apply(expert_output, 0), 0)), 0), 0)
            else:
                expert_output = C.all_to_all_single(expert_output, group=group)
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
                C.CurrentStreamRelease.apply(ctx.experts(C.CurrentStreamAcquire.apply(x, i), ctx).to(input.dtype), i)
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

        expert_output = expert_output.reshape(-1, g_experts, capacity, M)
        if expert_output.size(0) > 1:
            expert_output = torch.sum(expert_output.view(g_experts, -1, capacity, M), dim=1)
        expert_output = expert_output.view(g_experts * capacity, M)

        result_output = self._fdr.decode(expert_output)
        '''

        result_output = result_output.view(original_shape).to(original_dtype)
        self.l_aux = result_output.l_aux = l_aux
        return self.result_func(result_output) if self.result_func is not None else result_output

moe_layer = MOELayer
