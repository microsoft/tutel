# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from ..jit import JitKernel


def get_gating_kernel(samples, global_experts):
  if int(os.environ.get('GATE', '1')) == 0 or (samples & (samples - 1)) != 0:
    print('[WARN]', "`samples` (= %s) isn't in the form of 2^k, which is outside optimization scope and may result in big performance regression." % samples)

    def general_gating(indices1_s, mask1):
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations1_s = torch.sum(locations1 * mask1, dim=1).to(torch.int32)
        return locations1_s
    return general_gating

  tensor_cnt = samples * global_experts
  thread_num = min(1024, samples)
  batch_num = global_experts

  base_kernel = JitKernel.generate_kernel({'tensor_cnt': tensor_cnt, 'thread_num': thread_num, 'batch_num': batch_num}, '''
    #define tensor_cnt  (@tensor_cnt@)
    #define thread_num  (@thread_num@)
    #define batch_num   (@batch_num@)
    #define __out__

    extern "C" __global__ __launch_bounds__(thread_num) void cumsum(int* __restrict__ indices1_s, __out__ int* __restrict__ locations1_s) {
        // HINT: blockIdx.x, threadIdx.x = batch_num, thread_num

        // [thread_extent] blockIdx.x = @batch_num@
        // [thread_extent] threadIdx.x = @thread_num@

        __shared__  int temp[thread_num + 1];
        int thid = threadIdx.x, bid = blockIdx.x;
        int last_sum = -1;
        constexpr int size_per_batch = tensor_cnt / batch_num, step = size_per_batch / thread_num;
        for (int S = 0; S < step; ++S, locations1_s += thread_num, indices1_s += thread_num) {
            int offset = 1;
            temp[thid] = (thid < thread_num) ? (bid == indices1_s[thid]) : 0;
            for (int d = thread_num >> 1; d > 0; d >>= 1) {
                    __syncthreads();
                    if (thid < d)
                            temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                    offset *= 2;
            }
            if (thid == 0)
                    temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
            for (int d = 1; d < thread_num; d *= 2) {
                    offset >>= 1;
                    __syncthreads();
                    if (thid < d) {
                            int ai = offset * (2 * thid + 1) - 1;
                            int bi = offset * (2 * thid + 2) - 1;
                            int t = temp[ai];
                            temp[ai] = temp[bi];
                            temp[bi] += t;
                    }
            }
            __syncthreads();
            if (bid == indices1_s[thid] && temp[thid + 1] + last_sum >= 0) {
                    locations1_s[thid] = temp[thid + 1] + last_sum;
            }
            last_sum += temp[thread_num];
        }
    }
  ''')

  def optimized_gating(indices1_s, mask1 = None):
    indices1_s = indices1_s.to(torch.int32).contiguous()
    locations1_s = torch.empty([samples,], dtype=torch.int32, device=indices1_s.device).contiguous()
    base_kernel(indices1_s, locations1_s)
    return locations1_s

  return optimized_gating
