# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch.distributed as dist
from functools import partial
from .impls.communicate import get_world_size, get_world_rank, create_groups_from_world, barrier
# Communication without Backward Compute
from .impls.communicate import simple_all_reduce, simple_all_to_all, simple_split, simple_reduce_scatter, \
    simple_all_gather
# Communication with Backward Compute
from .impls.communicate import all_to_all, all_to_all_single, all_gather, zero_gather, zero_scatter, spatial_split, \
    reduce_scatter, allreduce_forward, allreduce_backward


class TutelDistributedOptimizer:
    def __init__(self, params, group=None, average_shared=False):
        """
        params:
            iterable of parameters, like: [params] or dicts defining shared_param_groups and expert_param_groups, like:
            {'shared': shared_param_groups, 'expert': expert_param_groups}
        """
        if isinstance(params, list):
            shared_params = [x for x in params if not hasattr(x, '_tutel_expert')]
            expert_params = [x for x in params if hasattr(x, '_tutel_expert')]
        elif isinstance(params, dict):
            shared_params = params['shared']
            expert_params = params['expert']
        else:
            raise NotImplementedError("TutelDistributedOptimizer got unsupported inputs ")

        shared_param_groups = list(shared_params)
        expert_param_groups = list(expert_params)
        if len(shared_param_groups) == 0:
            raise ValueError("Tutel optimizer got an empty shared parameter list")
        else:
            if not isinstance(shared_param_groups[0], dict):
                shared_param_groups = [{'params': shared_param_groups}]

        if len(expert_param_groups) == 0:
            print("Tutel optimizer got an empty expert parameter list")
        else:
            if not isinstance(expert_param_groups[0], dict):
                expert_param_groups = [{'params': expert_param_groups}]

        self.shared_param_groups = shared_param_groups
        self.expert_param_groups = expert_param_groups
        self.shared_param_groups_shapes = []
        for param_group in self.shared_param_groups:
            self.shared_param_groups_shapes.append([x.shape for x in param_group['params']])
        self.group = group
        self.average_shared = average_shared

        self.original_param_groups = self.shared_param_groups + self.expert_param_groups

    def chunk_param(self):
        mock_groups = []
        for param_group in self.shared_param_groups:
            mocks = []
            for p in param_group['params']:
                mocks += [zero_scatter(p.data, simple_split, group=self.group)[0]]
            _group = {k: v for k, v in param_group.items() if k != 'params'}
            _group['params'] = mocks
            mock_groups.append(_group)
        self.virt_param_groups = mock_groups

    def chunk_grad(self):
        for param_group, virt_param_group in zip(self.shared_param_groups, self.virt_param_groups):
            for i, p in enumerate(param_group['params']):
                if hasattr(p, 'grad') and p.grad is not None:
                    virt_param_group['params'][i].grad, _ = zero_scatter(p.grad.view(-1), simple_split,
                                                                         group=self.group)

    def restore(self):
        for virt_param_group, param_group, param_group_shapes in \
                zip(self.virt_param_groups, self.shared_param_groups, self.shared_param_groups_shapes):
            for i, p in enumerate(virt_param_group['params']):
                data = simple_all_gather(p.data, group=self.group).view(-1)
                param_group['params'][i].data = data[:param_group_shapes[i].numel()].view(param_group_shapes[i])

    def warp_local(self, local_optim_func, *args, **kwargs):
        self.chunk_param()
        self.local_optim_func = partial(local_optim_func, *args, **kwargs)
        self.local_optim = self.local_optim_func(self.virt_param_groups + self.expert_param_groups)
        self.local_param_groups = self.local_optim.param_groups
        return self

    def reset_zero_optim(self):
        self.chunk_param()
        self.local_optim = self.local_optim_func(self.virt_param_groups + self.expert_param_groups)
        self.local_param_groups = self.local_optim.param_groups

    def zero_grad(self):
        for group in self.shared_param_groups + self.expert_param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def step(self):
        self.chunk_grad()
        self.local_optim.step()
        self.restore()

    def state_dict(self):
        return self.local_optim.state_dict()

    def load_state_dict(self, state_dict):
        self.reset_zero_optim()
        self.local_optim.load_state_dict(state_dict)

    def all_reduce_grad(self):
        for param_group in self.shared_param_groups:
            for i, p in enumerate(param_group['params']):
                if hasattr(p, 'grad') and p.grad is not None:
                    if self.average_shared:
                        grad = p.grad / get_world_size(self.group)
                    else:
                        grad = p.grad
                    p.grad = simple_all_reduce(grad, group=self.group)


def sync_params_and_buffers(model, process_group=None, authoritative_rank=0):
    # Modified from https://github.com/pytorch/pytorch/blob/76cff182428fbd165b5725f3de29dbd91a1512fa/torch/distributed/utils.py#L120
    if process_group is None:
        process_group = dist.distributed_c10d._get_default_group()

    if hasattr(model, "_ddp_params_and_buffers_to_ignore"):
        parameters_to_ignore = model._ddp_params_and_buffers_to_ignore
    else:
        parameters_to_ignore = []
    broadcast_bucket_size = int(250 * 1024 * 1024)

    module_states = []
    for name, param in model.state_dict().items():
        if name not in parameters_to_ignore:
            module_states.append(param)

    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, authoritative_rank
        )
