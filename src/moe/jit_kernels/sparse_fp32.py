import torch
from ..jit import JitKernel, IS_HIP_EXTENSION

from ..moe_layer import shared_data

func_fwd = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)

extern "C" __global__ __launch_bounds__(1024) void forward_dispatched_input(float* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
      if (locations1_s[i] < capacity) {
          #pragma unroll
          for (int j = threadIdx.x; j < hidden; j += blockDim.x)
              atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], gates1_s[i] * reshaped_input[i * (hidden) + j]);
      }
}
'''.replace('@capacity@', str(shared_data.capacity)).replace('@samples@', str(shared_data.gates1_s.shape[0])).replace('@hidden@', str(shared_data.model_dim)))

if not IS_HIP_EXTENSION:
    func_bwd_gate = JitKernel.create('''
#define capacity (@capacity@)

extern "C" __global__ __launch_bounds__(32) void backward_gates1_s(float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.x = 32
  float grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = 0.000000e+00f;
  for (int HIDDEN_outer = 0; HIDDEN_outer < 64; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((int)blockIdx.x))] < capacity) ? (dispatched_input[(((((indices1_s[(((int)blockIdx.x))] * (2048 * capacity)) + (locations1_s[(((int)blockIdx.x))] * 2048)) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))] * reshaped_input[((((((int)blockIdx.x) * 2048) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))]) : 0.000000e+00f));
  }
  float red_buf0[1];
  unsigned mask[1];
  float t0[1];
  red_buf0[(0)] = grad_gates1_s_rf[(0)];
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
  if (((int)threadIdx.x) == 0) {
    grad_gates1_s[(((int)blockIdx.x))] = red_buf0[(0)];
  }
}
'''.replace('@capacity@', str(shared_data.capacity)))
else:
    func_bwd_gate = JitKernel.create('''
#define capacity (@capacity@)

extern "C" __global__ __launch_bounds__(64) void template_op_kernel0(float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.x = 64
  float grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = 0.000000e+00f;
  for (int HIDDEN_outer = 0; HIDDEN_outer < 32; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((int)blockIdx.x))] < capacity) ? (dispatched_input[(((((indices1_s[(((int)blockIdx.x))] * (2048 * capacity)) + (locations1_s[(((int)blockIdx.x))] * 2048)) + (HIDDEN_outer * 64)) + ((int)threadIdx.x)))] * reshaped_input[((((((int)blockIdx.x) * 2048) + (HIDDEN_outer * 64)) + ((int)threadIdx.x)))]) : 0.000000e+00f));
  }
  __shared__ float red_buf0[64];
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = grad_gates1_s_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    grad_gates1_s[(((int)blockIdx.x))] = ((volatile float*)red_buf0)[(0)];
  }
}
'''.replace('@capacity@', str(shared_data.capacity)))

func_bwd_data = JitKernel.create('''
#define capacity (@capacity@)

extern "C" __global__ __launch_bounds__(32) void backward_reshaped_input(float* __restrict__ gates1_s, float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 256
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 32
  // [thread_extent] threadIdx.y = 32
  grad_reshaped_input[((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))] = ((locations1_s[((((int)blockIdx.x) * 8))] < capacity) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[(((((indices1_s[((((int)blockIdx.x) * 8))] * (2048 * capacity)) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))] = ((locations1_s[((((int)blockIdx.x) * 8))] < capacity) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[((((((indices1_s[((((int)blockIdx.x) * 8))] * (2048 * capacity)) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2048))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2080))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4096))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4128))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6144))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6176))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8192))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8224))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10240))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10272))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12288))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12320))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14336))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14368))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < capacity) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
}
'''.replace('@capacity@', str(shared_data.capacity)))
