# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import warnings
import torch
from torch.nn import functional as F
from torch.nn import init
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

    def update(self, ctx, dtype=None, device=None):
        self.sharded_count = ctx.sharded_count
        if ctx.sharded_count > 1:
            assert self.hidden_size_per_expert % ctx.sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({self.hidden_size_per_expert}) to {ctx.sharded_count} slices."

        hidden_size = self.hidden_size_per_expert // ctx.sharded_count
        model_dim = ctx.model_dim
        local_experts = ctx.num_local_experts
        self.output_dim = self.output_dim or model_dim

        fc1_weight = torch.empty(local_experts, hidden_size, model_dim, dtype=dtype, device=device)
        fc2_weight = torch.empty(local_experts, hidden_size, self.output_dim, dtype=dtype, device=device)
        fc1_bias = torch.empty(local_experts, hidden_size, dtype=dtype, device=device)
        fc2_bias = torch.empty(local_experts, (self.output_dim + ctx.sharded_count - 1) // ctx.sharded_count, dtype=dtype, device=device)

        self.batched_fc1_w = torch.nn.Parameter(fc1_weight)
        self.batched_fc2_w = torch.nn.Parameter(fc2_weight)
        self.batched_fc1_bias = torch.nn.Parameter(fc1_bias)
        self.batched_fc2_bias = torch.nn.Parameter(fc2_bias)

        self.reset_parameters()

    def _kaiming_uniform(self, tensor, fan, a=0, nonlinearity='leaky_relu'):
        gain = init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def reset_parameters(self) -> None:
        # imitate reset_parameters from torch.nn.Linear
        mode = 'fan_in'
        a = math.sqrt(5)

        # init fc1
        # fan is calculated per expert; index weight for expert [0]; note the transpose
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.batched_fc1_w[0].t())
        fan_out *= self.sharded_count # fan_out should be multiplied by sharded_count
        fan = fan_in if mode == 'fan_in' else fan_out
        self._kaiming_uniform(self.batched_fc1_w, fan, a=a)
        if self.batched_fc1_bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.batched_fc1_bias, -bound, bound)

        # init fc2
        # fan is calculated per expert; index weight for expert [0]
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.batched_fc2_w[0])
        fan_in *= self.sharded_count # fan_out should be multiplied by sharded_count
        fan = fan_in if mode == 'fan_in' else fan_out
        self._kaiming_uniform(self.batched_fc2_w, fan, a=a)
        if self.batched_fc2_bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.batched_fc2_bias, -bound, bound)

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

        # Note: since parameters are in full precision,
        # the zero_gathers are done in full precision
        # which also means the grad scatters are in full precision
        if ctx.force_data_parallel:
            batched_fc1_w = net.zero_gather(batched_fc1_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc1_w.size(2))
            batched_fc2_w = net.zero_gather(batched_fc2_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc2_w.size(2))
            batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
            batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
        elif ctx.force_adaptive:
            if ctx.sharded_count > 1:
                group_size = ctx.sharded_count // ctx.adaptive_degree
                if group_size > 1:
                    ffn_zero_group = net.create_groups_from_world(group_count=-group_size).model_group
                    batched_fc1_w = net.zero_gather(batched_fc1_w, group=ffn_zero_group).view(1, -1, ctx.model_dim)
                    batched_fc2_w = net.zero_gather(batched_fc2_w, group=ffn_zero_group).view(1, -1, self.output_dim)
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                ffn_zero_group2 = net.create_groups_from_world(group_count=ctx.num_global_experts).model_group
                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ffn_zero_group2)
                batched_fc2_bias = batched_fc2_bias.view(1, 1, -1)

                if ctx.adaptive_degree > 1:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.adaptive_degree)
        else:
            if ctx.sharded_count > 1:
                ffn_zero_group = net.create_groups_from_world(group_count=ctx.num_global_experts).model_group
                if not ctx.use_model_parallel:
                    batched_fc1_w = net.zero_gather(batched_fc1_w, group=ffn_zero_group).view(1, -1, ctx.model_dim)
                    batched_fc2_w = net.zero_gather(batched_fc2_w, group=ffn_zero_group).view(1, -1, self.output_dim)
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ffn_zero_group)
                batched_fc2_bias = batched_fc2_bias.view(self.batched_fc2_bias.size(0), 1, -1)

                if ctx.use_model_parallel:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.sharded_count)

        if batched_fc2_bias.size(-1) != self.output_dim:
            batched_fc2_bias = batched_fc2_bias[:, :, :self.output_dim]

        y = torch.add(torch.matmul(x, batched_fc1_w.permute(0, 2, 1)), batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.add(torch.matmul(y, batched_fc2_w), batched_fc2_bias)
        return y


ExpertModule = FusedExpertsNetwork 
