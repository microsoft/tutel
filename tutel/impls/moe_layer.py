# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections
import importlib

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

from ..impls import communicate as C
from ..impls.fast_dispatch import fast_encode, fast_decode, extract_critical, get_dispatch_count
from ..impls.overlap import a2a_ffn_overlap_forward
from . import losses


def cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        # casts inputs to autocast dtype which enables all2all to be done in low precision
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        elif tensor.device.type == 'xpu':
            dtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
        elif tensor.device.type == 'hpu':
            dtype = torch.hpu.get_autocast_hpu_dtype()  # type: ignore[attr-defined]
        else:
            raise RuntimeError('User specified autocast device_type must be \'cuda\' or \'cpu\'')
        return tensor.to(dtype=dtype)


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
        assert world_size % -num_local_experts == 0, f"Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buff_name = prefix + '_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts
            assert state_experts == expect_experts, "Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        for name, param in self.experts.named_parameters():
            buff_name = prefix + 'experts.' + name
            if buff_name not in state_dict:
                logging.warning("Could not find parameter `%s` in state_dict, zero values will be filled into this parameter." % buff_name)
                state_dict[buff_name] = torch.zeros_like(param)
            if state_dict[buff_name].numel() == param.numel():
                state_dict[buff_name] = state_dict[buff_name].view(param.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    @property
    def num_global_experts(self):
        return int(self._num_global_experts)

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
        normalize_gate=True,
        is_gshard_loss=True,
        parallel_type='adaptive:1',
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

        self.num_local_experts = experts.pop('count_per_node', 1)
        if self.num_local_experts == -1:
            self.num_local_experts = 1
        self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))

        self.world_size = C.get_world_size(self.group)
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
            self.num_local_experts = 1
        else:
            self.sharded_count = 1

        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]

        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

        self.model_dim = model_dim

        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])

        experts_type = experts.pop('type')
        if experts_type == 'custom':
            expert_module = experts.pop('module')
            experts['model_dim'] = self.model_dim
            experts['local_experts'] = self.num_local_experts
            experts['sharded_count'] = self.sharded_count
            self.experts = cast(ModuleList, expert_module(**experts))
        else:
            assert re.match(r'[a-zA-Z0-9\_]+', experts_type), "Expert type must only include digits, letters and underline characters."
            try:
                fused_experts = importlib.import_module(f'...experts.{experts_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Builtin expert type is not recognized: %s' % experts_type)

            if experts_type == 'ffn':
                assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."
                assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

            experts['model_dim'] = self.model_dim
            experts['local_experts'] = self.num_local_experts
            experts['sharded_count'] = self.sharded_count
            self.experts = fused_experts.ExpertModule(**experts)

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.experts.named_parameters():
            setattr(p, '_tutel_expert', True)

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            gate_type = single_gate_type['type']
            single_gate_type.pop('type')
            assert re.match(r'[a-zA-Z0-9\_]+', gate_type), "Gate type must only include digits, letters and underline characters."

            if seeds is not None and seeds[0] is not None:
                torch.manual_seed(seeds[0] + gi)
            try:
                single_gate = importlib.import_module(f'...gates.{gate_type}', __name__)
            except ModuleNotFoundError:
                raise Exception("Unrecognized gate_type: %s" % gate_type)

            gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, **single_gate_type)
            if not hasattr(gate_module, 'gate_noise'):
                gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)
            if not hasattr(gate_module, 'capacity_factor'):
                gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

            self.gates += [gate_module]

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def extra_repr(self):
        return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % (
            [f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates],
            self.num_global_experts,
            self.world_size,
        )

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def expert_local(self, x, reserve_shape):
        y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)
        self.protected_shape = y.shape
        return y.reshape(y.size(0), y.size(1), -1)

    def forward(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=False, adaptive_r=None, megablocks_size=0):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"

        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        if torch.is_autocast_enabled():
            x = cast_if_autocast_enabled(x)
        else:
            for p in self.experts.parameters():
                x = x.to(p.dtype)
                break
        gctx = self.gates[gate_index]
        if a2a_ffn_overlap_degree is not None:
            self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        a2a_ffn_overlap_degree = self.a2a_ffn_overlap_degree

        top_k = top_k or gctx.top_k

        if megablocks_size > 0:
            if self.num_local_experts <= 1 or torch.is_grad_enabled() or self.world_size > 1:
                megablocks_size = 0

        def routing():
            logits = gctx(x)

            if self.training and gctx.gate_noise > 0:
                logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
            else:
                logits_w_noise = logits

            scores = F.softmax(logits_w_noise, dim=1)
            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
            else:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                    F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),
                    self.num_global_experts, gctx.gate_noise)

            mega_up = max(megablocks_size, 1)
            alignment = (self.sharded_count * a2a_ffn_overlap_degree + mega_up - 1) // mega_up * mega_up
            if alignment > 256:
                alignment = (alignment + 127) // 128 * 128

            return logits.dtype, extract_critical(scores,
                top_k = top_k,
                loss_fn = _loss_fn,
                capacity_factor = capacity_factor or gctx.capacity_factor,
                batch_prioritized_routing = self.batch_prioritized_routing,
                normalize_gate = self.normalize_gate,
                group = self.group,
                alignment = alignment,
                inequivalent_tokens = inequivalent_tokens,
            )


        if x.is_cuda:
            try:
                with torch.amp.autocast('cuda', enabled=False):
                    logits_dtype, (crit, l_aux) = routing()
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    logits_dtype, (crit, l_aux) = routing()
        else:
            logits_dtype, (crit, l_aux) = routing()

        self.megablocks_size = megablocks_size
        self.dispatch_count = get_dispatch_count(crit)
        y = fast_encode(x.to(logits_dtype), crit, self.is_postscore).to(x.dtype)

        if adaptive_r is not None:
            self.adaptive_degree = adaptive_r

        if self.adaptive_degree == 0:
            y = self.expert_local(y, original_shape[-reserve_dims:])
        else:
            if self.auto_parallel:
                self.use_model_parallel = (y.numel() * (self.sharded_count - 1) * 2 < sum([x.numel() for x in self.experts.parameters()]))

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))
                else:
                    y = y.view(self.world_size, -1, y.size(2))

            if a2a_ffn_overlap_degree > 1 and y.is_cuda:
                def expert_fn(expert_input):
                    return self.expert_local(expert_input, original_shape[-reserve_dims:])
                y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)
            else:
                y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)
                y = self.expert_local(y, original_shape[-reserve_dims:])
                y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)
                else:
                    y = y.view(self.num_global_experts, -1, y.size(2))

        y = fast_decode(y.to(logits_dtype), crit, self.is_postscore)

        y = y.view(list(original_shape[:-reserve_dims]) + list(self.protected_shape[-reserve_dims:])).to(original_dtype)
        self.l_aux = y.l_aux = l_aux
        return self.result_func(y) if self.result_func is not None else y

moe_layer = MOELayer
