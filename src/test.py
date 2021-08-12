#!/usr/bin/env python3

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from moe.moe_layer import MOELayer

batch_size = 4
num_tokens = 512
model_dim = 2048

device = 'cuda'
torch.manual_seed(1)

torch.set_default_dtype(torch.float32)

class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        experts = nn.ModuleList([nn.Linear(model_dim, model_dim), nn.Linear(model_dim, model_dim)])
        self._moe_layer = MOELayer('Top1Gate', experts, model_dim).to(device)

    def forward(self, input):
        result = self._moe_layer(input)

        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

model = ExampleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

x = torch.randn([batch_size, num_tokens, model_dim], device=device, requires_grad=True)
y = torch.LongTensor(batch_size).random_(1).to(device)

for i in range(10):
  torch.cuda.synchronize()
  t_start = time.time()
  optimizer.zero_grad()

  output = model(x)
  loss = F.nll_loss(output, y)
  loss.backward()
  optimizer.step()

  torch.cuda.synchronize()
  t_stop = time.time()
  print(f'STEP-{i}: DONE, loss = {loss.data}, step_time = {t_stop - t_start} sec.')
