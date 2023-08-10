# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .. import net

class FusedExpertsNetwork(torch.nn.Module):
    def __init__(self, hidden_size_per_expert, activation_fn=None, activation_fn_with_self=None, output_dim=None):
        super().__init__()
        self.skip_expert = (int(torch.os.environ.get('SKIP_EXPERT', '0')) != 0)
        self.hidden_size_per_expert = hidden_size_per_expert
        self.output_dim = output_dim

        if activation_fn_with_self is not None:
            assert activation_fn is None, "Option `activation_fn_with_self` has been specified, please keep exactly one of them."
            activation_fn = lambda x: activation_fn_with_self(x, self)
        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        self.activation_fn = activation_fn

    def update(self, ctx):
        if ctx.sharded_count > 1:
            assert self.hidden_size_per_expert % ctx.sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({self.hidden_size_per_expert}) to {ctx.sharded_count} slices."

        hidden_size = self.hidden_size_per_expert // ctx.sharded_count
        model_dim = ctx.model_dim
        local_experts = ctx.num_local_experts
        self.output_dim = self.output_dim or model_dim

        fc1_weight = torch.empty(1, local_experts, hidden_size, model_dim)
        fc2_weight = torch.empty(1, local_experts, hidden_size, self.output_dim)
        fc1_bias = torch.empty(1, local_experts, hidden_size)
        fc2_bias = torch.empty(1, local_experts, (self.output_dim + ctx.sharded_count - 1) // ctx.sharded_count)

        for i in range(local_experts):
            fc1 = torch.nn.Linear(model_dim, hidden_size)
            fc2 = torch.nn.Linear(hidden_size, self.output_dim)
            fc1_weight[0, i, :, :], fc1_bias[0, i, :] = fc1.weight, fc1.bias
            fc2_weight[0, i, :, :], fc2_bias[0, i, :] = fc2.weight.t(), fc2.bias[:fc2_bias.size(-1)]

        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(fc1_weight.squeeze(0)))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(fc2_weight.squeeze(0)))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(fc1_bias.squeeze(0)))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(fc2_bias.squeeze(0)))

    def extra_repr(self):
        return 'model_dim=%d, hidden_size=%d, output_dim=%d, local_experts=%d' % (
            self.batched_fc1_w.size(2), self.batched_fc1_w.size(1), self.batched_fc2_w.size(2), self.batched_fc1_w.size(0)
        )

    def forward(self, x, ctx):
        if self.skip_expert:
            return x

        batched_fc1_w = self.batched_fc1_w
        batched_fc2_w = self.batched_fc2_w
        batched_fc1_bias = self.batched_fc1_bias.unsqueeze(1)
        batched_fc2_bias = self.batched_fc2_bias.unsqueeze(1)

        # Implementation of https://arxiv.org/pdf/2211.15841.pdf in Tutel v0.3.x
        #   which benifits decoder inference on single-GPU if num_local_experts >= 2
        if ctx.megablocks_size > 0:
            sparse_size = ctx.megablocks_size
            sparse_groups = torch.div(ctx.dispatch_count + (sparse_size - 1), sparse_size, rounding_mode='floor')
            sparse_groups = torch.minimum(sparse_groups, torch.tensor(x.size(1) // sparse_size, dtype=torch.int32, device=x.device))
            y = torch.ops.tutel_ops.sparse_bmm_infer(x, batched_fc1_w, sparse_groups, True, sparse_size)
            y = torch.add(y, batched_fc1_bias)
            y = self.activation_fn(y)
            y = torch.ops.tutel_ops.sparse_bmm_infer(y, batched_fc2_w, sparse_groups, False, sparse_size)
            y = torch.add(y, batched_fc2_bias)
            return y

        if ctx.adaptive_degree == 0:
            batched_fc1_w = net.zero_gather(batched_fc1_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc1_w.size(2))
            batched_fc2_w = net.zero_gather(batched_fc2_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc2_w.size(2))
            batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
            batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
        else:
            if ctx.sharded_count > 1:
                group_size = ctx.sharded_count // ctx.adaptive_degree
                if group_size > 1:
                    ffn_zero_group = net.create_groups_from_world(group_count=-group_size).model_group
                    batched_fc1_w = net.zero_gather(batched_fc1_w, group=ffn_zero_group).view(1, -1, ctx.model_dim)
                    batched_fc2_w = net.zero_gather(batched_fc2_w, group=ffn_zero_group).view(1, -1, self.output_dim)
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=net.create_groups_from_world(group_count=ctx.num_global_experts).model_group)
                batched_fc2_bias = batched_fc2_bias.view(1, 1, -1)

                if ctx.adaptive_degree > 1:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.adaptive_degree)

        if batched_fc2_bias.size(-1) != self.output_dim:
            batched_fc2_bias = batched_fc2_bias[:, :, :self.output_dim]

        y = torch.add(torch.matmul(x, batched_fc1_w.permute(0, 2, 1)), batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.add(torch.matmul(y, batched_fc2_w), batched_fc2_bias)
        return y


ExpertModule = FusedExpertsNetwork 
