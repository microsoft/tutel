# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from .impls.communicate import get_world_size, get_world_rank, create_groups_from_world, create_standalone_group, barrier
# Communication without Backward Compute
from .impls.communicate import simple_all_reduce, simple_all_to_all,simple_split, simple_reduce_scatter, simple_all_gather
# Communication with Backward Compute
from .impls.communicate import all_to_all, all_to_all_single, all_gather, zero_gather, zero_scatter, spatial_split, reduce_scatter, allreduce_forward, allreduce_backward
# Communication with Batch-based Compute
from .impls.communicate import batch_all_to_all_v, batch_all_gather_v


class TutelDistributedOptimizer:
    def __init__(self, params, group=None, average_shared=False):
        params = [x for x in params]
        self.params = [x for x in params if not hasattr(x, '_tutel_expert')]
        self.expert_params = [x for x in params if hasattr(x, '_tutel_expert')]
        self.shapes = [x.shape for x in self.params]
        self.group = group
        self.average_shared = average_shared

    def chunk_param(self):
        mocks = []
        for p in self.params:
            mocks += [zero_scatter(p.data, simple_split, group=self.group)[0]]
        self.virt_params = mocks

    def chunk_grad(self):
        for i, p in enumerate(self.params):
            if hasattr(p, 'grad') and p.grad is not None:
                if self.average_shared:
                    grad = p.grad.view(-1) / get_world_size(self.group)
                else:
                    grad = p.grad.view(-1)
                self.virt_params[i].grad, _ = zero_scatter(grad, simple_reduce_scatter, group=self.group)

    def restore(self):
        for i, p in enumerate(self.virt_params):
            data = simple_all_gather(p.data, group=self.group).view(-1)
            self.params[i].data = data[:self.shapes[i].numel()].view(self.shapes[i])

    def warp_local(self, local_optim, *args, **kwargs):
        self.chunk_param()
        self.local_optim = local_optim(self.virt_params + self.expert_params, *args, **kwargs)
        return self

    def zero_grad(self):
        for p in self.params + self.expert_params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        self.chunk_grad()
        self.local_optim.step()
        self.restore()
