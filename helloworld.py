#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


batch_size = 4
num_tokens = 512
model_dim = 2048
hidden_size = 1024
num_local_experts = 2
top_value = 2
device = torch.device('cuda', 0)
activation_fn = lambda x: F.relu(x)

assert top_value in (1, 2), "Only support top_value = 1, 2 in this version."


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        from tutel.moe_layer import MOELayer
        self._moe_layer = MOELayer(
            'Top2Gate',
            model_dim,
            builtin_experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_size, 'activation_fn': activation_fn},
            fp32_gate=True
        ).to(device)


    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel()
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)


average_time, num_steps = 0, 20

for i in range(num_steps):
    torch.cuda.synchronize()
    t_start = time.time()
    optimizer.zero_grad()

    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t_stop = time.time()
    print('STEP-%s: DONE, loss = %s, step_time = %s sec.' % (i, float(loss.data), t_stop - t_start))

    if i + 10 >= num_steps:
        average_time += t_stop - t_start

average_time /= 10
print('\n[Summary] Average synchronized step_time = %s sec.' % average_time)
