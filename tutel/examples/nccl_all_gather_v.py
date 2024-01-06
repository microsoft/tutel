#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tutel import system, net

parallel_env = system.init_data_model_parallel(backend='nccl', group_count=1)
local_device = parallel_env.local_device

assert parallel_env.global_size == 2, "This test case is set for World Size == 2 only"

if parallel_env.global_rank == 0:
  input = torch.tensor([10, 10, 10, 10, 10], device=local_device)
else:
  input = torch.tensor([20, 20, 20], device=local_device)

print(f'Device-{parallel_env.global_rank} sends: {[input,]}')

net.barrier()

print(f'Device-{parallel_env.global_rank} recvs: {net.batch_all_gather_v([input,])[0]}')
