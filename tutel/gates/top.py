# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

class LinearTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, device=None, dtype=None, **options):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False, device=device, dtype=dtype)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
        with torch.autocast(device_type=x.device.type, enabled=not self.fp32_gate):
            out = self.wg(x)
            return out


Gate = LinearTopKGate
