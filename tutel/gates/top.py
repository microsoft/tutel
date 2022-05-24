# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os

class LinearTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, **options):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False, dtype=torch.float32 if fp32_gate else None)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
        return self.wg(x)

Gate = LinearTopKGate
