#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tutel import system
from tutel import moe
from tutel import net

if torch.cuda.is_available():
  pa_env = system.init_data_model_parallel(backend='nccl')
else:
  pa_env = system.init_data_model_parallel(backend='gloo')

num_samples = 4096
model_dim, hidden_size = 2048, 4096
num_local_experts = 16
num_global_experts = num_local_experts * pa_env.global_size


class CustomGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.register_parameter(name='wg', param=torch.nn.Parameter(torch.randn([model_dim, num_global_experts]) * 1e-3))

    def forward(self, x):
        return torch.matmul(x, self.wg)

class CustomExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(pa_env.global_rank)
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([num_local_experts, model_dim, hidden_size]) * 1e-3))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([num_local_experts, hidden_size, model_dim]) * 1e-3))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, model_dim])))

        for x in self.parameters(): setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = torch.nn.functional.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y

class CustomMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = CustomGate()
        self.expert = CustomExpert()

    def forward(self, x, k=2):
        logits = self.gate(x)
        scores = torch.nn.functional.softmax(logits, dim=1)
        crit, l_aux = moe.extract_critical(scores, top_k=k)
        y = moe.fast_encode(x, crit)
        y = net.all_to_all(y, 1, 0)
        y = self.expert(y)
        y = net.all_to_all(y, 0, 1)
        output = moe.fast_decode(y, crit)
        return output, l_aux

model = CustomMoE().to(pa_env.local_device)

torch.manual_seed(0)
data = torch.randn([num_samples, model_dim], device=pa_env.local_device)
label = torch.LongTensor(num_samples).random_(1).to(pa_env.local_device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for i in range(10):
    optimizer.zero_grad()
    result, l_aux = model(data)
    loss = torch.nn.functional.nll_loss(result, label) + 0.0001 * l_aux
    loss.backward()
    optimizer.step()
    pa_env.dist_print(f'Step-{i}: custom MoE loss = {loss}')

    for p in model.parameters():
        if not hasattr(p, 'skip_allreduce'):
            p.grad = net.simple_all_reduce(p.grad)
