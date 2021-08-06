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

experts = nn.ModuleList([nn.Linear(model_dim, model_dim), nn.Linear(model_dim, model_dim)])
moe_layer = MOELayer('Top1Gate', experts, model_dim).to(device)

x = torch.ones([batch_size, num_tokens, model_dim], dtype=torch.float32, device=device, requires_grad=True)
y = torch.ones([batch_size], dtype=torch.int64, device=device)

def model(input):
  x = input
  x = moe_layer(x)
  x = torch.einsum('ijk->ij', x)
  x = F.log_softmax(x, dim=1)
  return x

for i in range(10):
  torch.cuda.synchronize()
  t_start = time.time()

  output = model(x)
  loss = F.nll_loss(output, y)
  loss.backward()

  torch.cuda.synchronize()
  t_stop = time.time()
  print(f'STEP-{i}: DONE, step_time = {t_stop - t_start} sec.')

