# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from ..impls.jit_compiler import JitCompiler


def get_kernel_dtype(param_dtype):
  if param_dtype == torch.float16:
      return '__half2'
  elif param_dtype == torch.float32:
      return 'float'
  else:
      raise Exception("Unrecognized data type: %s" % param_dtype)


def create_forward(param_dtype, is_cuda=True):
  if not is_cuda:
    return JitCompiler.generate_cpu_kernel(kernel_type=0)

  return JitCompiler.generate_kernel({'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define __dtype @dtype@

    extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity) {
      // [thread_extent] blockIdx.x = 512
      // [thread_extent] threadIdx.x = 1024

      for (int i = blockIdx.x; i < samples; i += gridDim.x)
          if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
                  dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j] = gates1_s[i] * reshaped_input[i * (hidden) + j];
          }
    }
  ''')


def create_backward_data(param_dtype, is_cuda=True):
  if not is_cuda:
    return JitCompiler.generate_cpu_kernel(kernel_type=1)

  return JitCompiler.generate_kernel({'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define __dtype @dtype@

    extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity) {
      // [thread_extent] blockIdx.x = 512
      // [thread_extent] threadIdx.x = 1024

      for (int i = blockIdx.x; i < samples; i += gridDim.x)
          if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
                  grad_reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
          } else {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
    #if @IS_FLOAT@
                  grad_reshaped_input[i * hidden + j] = __dtype(0);
    #else
                  grad_reshaped_input[i * hidden + j] = __dtype(0, 0);
    #endif
          }
    }
  ''')


def create_backward_gate(param_dtype, is_cuda=True):
  if not is_cuda:
    return JitCompiler.generate_cpu_kernel(kernel_type=2)

  return JitCompiler.generate_kernel({'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
  #define __dtype @dtype@

  extern "C" __global__ __launch_bounds__(32) void execute(void* __restrict__ grad_gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity) {
    // [thread_extent] blockIdx.x = 512
    // [thread_extent] threadIdx.x = 32
    for (int index = blockIdx.x; index < samples; index += gridDim.x) {
      if (locations1_s[index] >= capacity || indices1_s[index] < 0) {
        if (((int)threadIdx.x) == 0)
    #if @IS_FLOAT@
          ((float*)grad_gates1_s)[index] = 0;
    #else
          ((half*)grad_gates1_s)[index] = __float2half_rn(0.000000e+00f);
    #endif
        continue;
      }
      int indice = indices1_s[index] * capacity + locations1_s[index];
    #if @IS_FLOAT@
      __dtype grad_gates1_s_rf = 0.000000e+00f;
    #else
      __dtype grad_gates1_s_rf = __dtype(0, 0);
    #endif
      for (int i = threadIdx.x; i < hidden; i += 32)
        grad_gates1_s_rf += dispatched_input[indice * (hidden) + i] * reshaped_input[index * (hidden) + i];

  #if !defined(__HIPCC__)
      __dtype red_buf0[1];
      unsigned int mask[1];
      __dtype t0[1];
      red_buf0[(0)] = grad_gates1_s_rf;
      mask[(0)] = __activemask();
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  #else
      __shared__ __dtype red_buf0[32];
      __syncthreads();
      ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = grad_gates1_s_rf;
      if (((int)threadIdx.x) < 16) {
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 16))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 8))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 4))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 2))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 1))]));
      }
      __syncthreads();
  #endif
      if (((int)threadIdx.x) == 0)
  #if @IS_FLOAT@
        ((float*)grad_gates1_s)[index] = red_buf0[(0)];
  #else
        ((half*)grad_gates1_s)[index] = red_buf0[(0)].x + red_buf0[(0)].y;
  #endif
    }
  }
  ''')
