#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
import argparse

from tutel import system, net

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--size_mb', type=int, default=256)
parser.add_argument('--loop', type=int, default=50)

args = parser.parse_args()

parallel_env = system.init_data_model_parallel(backend='nccl' if args.device == 'cuda' else 'gloo')
local_device = parallel_env.local_device

x = torch.randn([((args.size_mb + 3) // 4 * 1024 * 1024 + parallel_env.global_size - 1) // parallel_env.global_size * parallel_env.global_size], device=local_device, dtype=torch.float32)

if args.device == 'cuda':
  wait = lambda: torch.cuda.synchronize() or time.perf_counter()
else:
  wait = lambda: time.perf_counter()

with torch.no_grad():
  for _ in range(args.loop):
    t0 = wait()
    net.simple_all_to_all(x.view(parallel_env.global_size, -1))
    t1 = wait()
    parallel_env.dist_print(f'AllToAll bandwidth across {parallel_env.global_size} node(s) = %.4f GB/s' % ((x.numel() * 4) * 1e-9 / (t1 - t0)))
    t0 = wait()
    net.simple_all_reduce(x.view(-1), inplace=True)
    t1 = wait()
    parallel_env.dist_print(f'AllReduce bandwidth across {parallel_env.global_size} node(s) = %.4f GB/s' % ((x.numel() * 4) * 1e-9 / (t1 - t0)))
    t0 = wait()
    net.simple_all_gather(x.view(parallel_env.global_size, -1)[parallel_env.global_rank])
    t1 = wait()
    parallel_env.dist_print(f'AllGather bandwidth across {parallel_env.global_size} node(s) = %.4f GB/s' % ((x.numel() * 4) * 1e-9 / (t1 - t0)))
    t0 = wait()
    net.simple_reduce_scatter(x.view(parallel_env.global_size, -1))
    t1 = wait()
    parallel_env.dist_print(f'ReduceScatter bandwidth across {parallel_env.global_size} node(s) = %.4f GB/s' % ((x.numel() * 4) * 1e-9 / (t1 - t0)))
    parallel_env.dist_print('')
    time.sleep(0.5)

