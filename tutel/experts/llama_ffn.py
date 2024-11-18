# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .. import net

class LlamaFFNNetwork(torch.nn.Module):

    def _create_sharded_param(self, *full_shape, **kwargs):
        full_shape = torch.Size(full_shape)
        sharded_shape = (full_shape.numel() + self.sharded_count - 1) // self.sharded_count
        return torch.nn.Parameter(torch.empty(sharded_shape, **kwargs)), full_shape

    def _get_gathered_param(self, param, full_shape, parent_group):
        sharded_group = net.create_groups_from_world(group_count=-self.sharded_count, parent_group=parent_group).model_group
        return net.zero_gather(param, group=sharded_group).view(-1).narrow(0, 0, full_shape.numel()).view(full_shape)

    def __init__(self, model_dim, hidden_size_per_expert, local_experts, sharded_count, activation_fn=torch.nn.functional.silu):
        super().__init__()
        self.sharded_count = sharded_count
        self.W_fc1, self.W_fc1_full_shape = self._create_sharded_param(local_experts, model_dim, hidden_size_per_expert)
        self.W_fc2, self.W_fc2_full_shape = self._create_sharded_param(local_experts, model_dim, hidden_size_per_expert)
        self.W_fc3, self.W_fc3_full_shape = self._create_sharded_param(local_experts, hidden_size_per_expert, model_dim)
        self.activation_fn = activation_fn
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
          self.W_fc1.normal_(0, 0.01)
          self.W_fc2.normal_(0, 0.01)
          self.W_fc3.normal_(0, 0.01)

    def forward(self, x, ctx):
        W_fc1_full = self._get_gathered_param(self.W_fc1, self.W_fc1_full_shape, ctx.group)
        W_fc2_full = self._get_gathered_param(self.W_fc2, self.W_fc2_full_shape, ctx.group)
        W_fc3_full = self._get_gathered_param(self.W_fc3, self.W_fc3_full_shape, ctx.group)

        y1 = torch.matmul(x, W_fc1_full)
        y2 = torch.matmul(x, W_fc2_full)
        y = self.activation_fn(y1) * y2
        y = torch.matmul(y, W_fc3_full)
        return y


ExpertModule = LlamaFFNNetwork 
