# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .. import net

class FusedExpertsNetwork(torch.nn.Module):
    def __init__(self, model_dim, hidden_size_per_expert, local_experts, sharded_count, activation_fn=None, activation_fn_with_self=None, output_dim=None, has_fc1_bias=True, has_fc2_bias=True):
        super().__init__()
        self.skip_expert = (int(torch.os.environ.get('SKIP_EXPERT', '0')) != 0)
        assert hidden_size_per_expert % sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({hidden_size_per_expert}) to {sharded_count} slices."
        self.model_dim = model_dim
        self.hidden_size_per_expert = hidden_size_per_expert
        self.local_experts = local_experts
        self.sharded_count = sharded_count
        self.hidden_size = hidden_size_per_expert // sharded_count
        self.output_dim = output_dim or model_dim

        if activation_fn_with_self is not None:
            assert activation_fn is None, "Option `activation_fn_with_self` has been specified, please keep exactly one of them."
            activation_fn = lambda x: activation_fn_with_self(x, self)
        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        self.activation_fn = activation_fn

        self.batched_fc1_w = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size, model_dim))
        self.batched_fc2_w = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size, self.output_dim))
        if has_fc1_bias:
             self.batched_fc1_bias = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size))
        else:
             self.register_parameter('batched_fc1_bias', None)
        if has_fc2_bias:
            self.batched_fc2_bias = torch.nn.Parameter(torch.empty(local_experts, (self.output_dim + sharded_count - 1) // sharded_count))
        else:
            self.register_parameter('batched_fc2_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.batched_fc1_w.size(0)):
                fc1 = torch.nn.Linear(self.model_dim, self.hidden_size, bias=self.batched_fc1_bias is not None)
                fc2 = torch.nn.Linear(self.hidden_size, self.output_dim, bias=self.batched_fc2_bias is not None)
                self.batched_fc1_w[i] = fc1.weight
                if self.batched_fc1_bias is not None:
                    self.batched_fc1_bias[i] = fc1.bias
                self.batched_fc2_w[i] = fc2.weight.t()
                if self.batched_fc2_bias is not None:
                    self.batched_fc2_bias[i] = fc2.bias[:self.batched_fc2_bias.size(-1)]

    def extra_repr(self):
        return 'model_dim=%d, hidden_size=%d, output_dim=%d, local_experts=%d. has_fc1_bias=%s, has_fc2_bias=%s.' % (
            self.batched_fc1_w.size(2), self.batched_fc1_w.size(1), self.batched_fc2_w.size(2), self.batched_fc1_w.size(0),
            self.batched_fc1_bias is not None, self.batched_fc2_bias is not None
        )

    def forward(self, x, ctx):
        if self.skip_expert:
            return x

        batched_fc1_w = self.batched_fc1_w
        batched_fc2_w = self.batched_fc2_w
        if self.batched_fc1_bias is not None:
            batched_fc1_bias = self.batched_fc1_bias.unsqueeze(1)
        if self.batched_fc2_bias is not None:
            batched_fc2_bias = self.batched_fc2_bias.unsqueeze(1)

        # Implementation of https://arxiv.org/pdf/2211.15841.pdf in Tutel v0.3.x
        #   which benifits decoder inference on single-GPU if num_local_experts >= 2
        if ctx.megablocks_size > 0:
            sparse_size = ctx.megablocks_size
            sparse_groups = torch.div(ctx.dispatch_count + (sparse_size - 1), sparse_size, rounding_mode='floor')
            sparse_groups = torch.minimum(sparse_groups, torch.tensor(x.size(1) // sparse_size, dtype=torch.int32, device=x.device))
            y = torch.ops.tutel_ops.sparse_bmm_infer(x, batched_fc1_w, sparse_groups, True, sparse_size)
            if self.batched_fc1_bias is not None:
                y = torch.add(y, batched_fc1_bias)
            y = self.activation_fn(y)
            y = torch.ops.tutel_ops.sparse_bmm_infer(y, batched_fc2_w, sparse_groups, False, sparse_size)
            if self.batched_fc2_bias is not None:
                y = torch.add(y, batched_fc2_bias)
            return y

        if ctx.adaptive_degree == 0:
            batched_fc1_w = net.zero_gather(batched_fc1_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc1_w.size(2))
            batched_fc2_w = net.zero_gather(batched_fc2_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc2_w.size(2))
            if self.batched_fc1_bias is not None:
                batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
            if self.batched_fc2_bias is not None:
                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
        else:
            if ctx.sharded_count > 1:
                mesh_size = net.get_world_size(ctx.group)
                if mesh_size > 1 and mesh_size < net.get_world_size():
                    ctx.adaptive_degree = ctx.sharded_count
                group_size = ctx.sharded_count // ctx.adaptive_degree

                if group_size > 1:
                    ffn_zero_group = net.create_groups_from_world(group_count=-group_size, parent_group=ctx.group).model_group
                    batched_fc1_w = net.zero_gather(batched_fc1_w, group=ffn_zero_group).view(1, -1, ctx.model_dim)
                    batched_fc2_w = net.zero_gather(batched_fc2_w, group=ffn_zero_group).view(1, -1, self.output_dim)
                    if self.batched_fc1_bias is not None:
                        batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                if self.batched_fc2_bias is not None:
                    batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=net.create_groups_from_world(group_count=ctx.num_global_experts, parent_group=ctx.group).model_group)
                    batched_fc2_bias = batched_fc2_bias.view(1, 1, -1)

                    if ctx.adaptive_degree > 1:
                        batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.adaptive_degree)

        if self.batched_fc2_bias is not None and batched_fc2_bias.size(-1) != self.output_dim:
            batched_fc2_bias = batched_fc2_bias[:, :, :self.output_dim]

        y = torch.matmul(x, batched_fc1_w.permute(0, 2, 1))
        if self.batched_fc1_bias is not None:
            y = torch.add(y, batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.matmul(y, batched_fc2_w)
        if self.batched_fc2_bias is not None:
            y = torch.add(y, batched_fc2_bias)
        return y


ExpertModule = FusedExpertsNetwork 
