#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import torch.distributed as dist
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fp16', default=False, action='store_true')
parser.add_argument('--w', type=int, default=1)
parser.add_argument('--e', type=int, default=2)
parser.add_argument('--m', type=int, default=2048)
parser.add_argument('--h', type=int, default=2048)

args = parser.parse_args()

assert args.h % args.e == 0
args.h //= args.e

device = torch.device('cuda', 0)
default_dtype = torch.float16 if args.fp16 else torch.float32

default_dtype = torch.float16 if args.fp16 else torch.float32
X = torch.randn([args.w, args.e, args.m, args.m], dtype=default_dtype, device=device)
Y = torch.randn([1, args.e, args.m, args.h], dtype=default_dtype, device=device)
Z = torch.randn([args.w, args.e, args.m, args.h], dtype=default_dtype, device=device)

print('X = %s, Y = %s => Z = %s' % (X.size(), Y.size(), Z.size()))

def evaluate(func_name):
  func = eval(func_name)
  average_time, num_steps = 0, 30
  for i in range(num_steps):
    torch.cuda.synchronize()
    t_start = time.time()
    func()
    torch.cuda.synchronize()
    t_stop = time.time()
    if i + 10 >= num_steps:
        average_time += t_stop - t_start
  average_time /= 10
  tflops = (2.0 * args.w * args.e * args.h * args.m * args.m) / average_time * 1e-12
  print('\n[Summary] Average synchronized step_time of `%s:%s` = %s sec. (Tflops = %s)' % (
    func_name, default_dtype, average_time, tflops))
  return average_time

X_l = torch.randn([args.w * args.e * args.m, args.m], dtype=default_dtype, device=device)
Y_l = torch.randn([args.m, args.h], dtype=default_dtype, device=device)
def layout_sgemm():
  torch.matmul(X_l, Y_l)

def auto_broadcast_bgemm():
  torch.matmul(X, Y)

def manual_broadcast_bgemm():
  torch.matmul(X, Y.repeat(X.size(0), 1, 1, 1))

Y_one = Y.repeat(X.size(0), 1, 1, 1).contiguous()
def skip_broadcast_bgemm():
  torch.matmul(X, Y)

X_one = X[0, :].contiguous()
def world_bgemm():
  for i in range(X.size(0)):
    torch.matmul(X_one, Y)

X_two, Y_two = X[:, 0, :].contiguous(), Y[:, 0, :].contiguous()
def expert_bgemm():
  for i in range(X.size(1)):
    torch.matmul(X_two, Y_two)

def backward_reduce_no_sum():
  torch.matmul(X, Z)

def backward_reduce():
  middle = torch.matmul(X, Z)
  torch.sum(middle, dim=0)

evaluate('auto_broadcast_bgemm')
evaluate('manual_broadcast_bgemm')
evaluate('skip_broadcast_bgemm')
evaluate('world_bgemm')
evaluate('expert_bgemm')
evaluate('layout_sgemm')
evaluate('backward_reduce_no_sum')
evaluate('backward_reduce')
