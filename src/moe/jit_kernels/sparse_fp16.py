import torch
from ..jit import JitKernel, IS_HIP_EXTENSION

from ..moe_layer import shared_data as s_cfg

if IS_HIP_EXTENSION:
  func_fwd_stage = JitKernel.create('''
#define capacity (@capacity@)

extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, half* __restrict__ reshaped_input, float* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 2
  // [thread_extent] threadIdx.x = 2
  // [thread_extent] blockIdx.y = 32
  // [thread_extent] threadIdx.y = 16
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    for (int vthread_s1 = 0; vthread_s1 < 32; ++vthread_s1) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] < capacity) ? atomicAdd(&dispatched_input[(((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * (2048 * capacity)) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * reshaped_input[(((((((((int)blockIdx.x) * 2097152) + (vthread_s * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s2 = 0; vthread_s2 < 16; ++vthread_s2) {
    for (int vthread_s3 = 0; vthread_s3 < 32; ++vthread_s3) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] < capacity) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * (2048 * capacity)) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 1))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s2 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s3 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 1))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s4 = 0; vthread_s4 < 16; ++vthread_s4) {
    for (int vthread_s5 = 0; vthread_s5 < 32; ++vthread_s5) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] < capacity) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * (2048 * capacity)) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 32))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s4 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s5 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 32))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s6 = 0; vthread_s6 < 16; ++vthread_s6) {
    for (int vthread_s7 = 0; vthread_s7 < 32; ++vthread_s7) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] < capacity) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * (2048 * capacity)) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 33))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s6 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s7 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 33))])) : 0.000000e+00f);
    }
  }
}
'''.replace('@capacity@', str(s_cfg.capacity)))

  def func_fwd(*inputs):
    inputs = list(inputs)
    output = inputs[-1]
    inputs[-1] = torch.zeros(output.shape, dtype=torch.float32, device=output.device)
    func_fwd_stage(*inputs)
    output.copy_(inputs[-1].to(output.dtype))
else:
  func_fwd_stage = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)

extern "C" __global__ __launch_bounds__(1024) void forward_dispatched_input(__half2* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __half2* __restrict__ reshaped_input, __half2* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
      if (locations1_s[i] < capacity) {
          #pragma unroll
          for (int j = threadIdx.x; j < hidden; j += blockDim.x)
              atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], gates1_s[i] * reshaped_input[i * (hidden) + j]);
      }
}
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.model_dim // 2)))

  def func_fwd(gates, *inputs):
    gates_hf2 = gates.view(-1, 1).repeat((1, 2))
    func_fwd_stage(gates_hf2, *inputs)


if IS_HIP_EXTENSION:
  func_bwd_gate = JitKernel.create('''
#define capacity (@capacity@)

extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, half* __restrict__ reshaped_input, half* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 1024
  // [thread_extent] threadIdx.y = 2
  // [thread_extent] threadIdx.x = 16
  half grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = __float2half_rn(0.000000e+00f);
  for (int HIDDEN_outer = 0; HIDDEN_outer < 128; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] < capacity) ? (dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] * (2048 * capacity)) + (locations1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] * 2048)) + (HIDDEN_outer * 16)) + ((int)threadIdx.x)))] * reshaped_input[(((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 2048)) + (HIDDEN_outer * 16)) + ((int)threadIdx.x)))]) : __float2half_rn(0.000000e+00f)));
  }
  __shared__ half red_buf0[32];
  __syncthreads();
  ((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = grad_gates1_s_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    ((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = ((half)(((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))]) + (half)(((volatile half*)red_buf0)[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 8))]));
    ((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = ((half)(((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))]) + (half)(((volatile half*)red_buf0)[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 4))]));
    ((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = ((half)(((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))]) + (half)(((volatile half*)red_buf0)[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 2))]));
    ((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = ((half)(((volatile half*)red_buf0)[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))]) + (half)(((volatile half*)red_buf0)[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 1))]));
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    grad_gates1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] = (half)(((volatile half*)red_buf0)[((((int)threadIdx.y) * 16))]);
  }
}
'''.replace('@capacity@', str(s_cfg.capacity)))
else:
  func_bwd_gate = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)
#define __dtype __half2

extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(__dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, half* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = @samples@
  // [thread_extent] threadIdx.x = 32
  if (locations1_s[blockIdx.x] >= capacity) {
    if (((int)threadIdx.x) == 0)
      grad_gates1_s[(((int)blockIdx.x))] = __float2half_rn(0.000000e+00f);
    return;
  }

  int indice = indices1_s[(int)blockIdx.x] * capacity + locations1_s[(int)blockIdx.x];
  __dtype grad_gates1_s_rf = __dtype(0, 0);

  for (int i = threadIdx.x; i < hidden; i += 32) {
    grad_gates1_s_rf += dispatched_input[indice * (hidden) + i] * reshaped_input[((int)blockIdx.x) * (hidden) + i];
  }

  __dtype red_buf0[1];
  uint mask[1];
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
  if (((int)threadIdx.x) == 0) {
    grad_gates1_s[(((int)blockIdx.x))] = red_buf0[(0)].x + red_buf0[(0)].y;
  }
}
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.model_dim // 2)))


_func_bwd_data = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)
#define __dtype half

extern "C" __global__ __launch_bounds__(1024) void template_op_kernel0(__dtype* __restrict__ gates1_s, __dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
      if (locations1_s[i] < capacity) {
          #pragma unroll
          for (int j = threadIdx.x; j < hidden; j += blockDim.x)
              grad_reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
      } else {
          #pragma unroll
          for (int j = threadIdx.x; j < hidden; j += blockDim.x)
              grad_reshaped_input[i * hidden + j] = __dtype(0);
      }
}
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.model_dim)))

def func_bwd_data(gates1_s, *ts):
    _func_bwd_data(gates1_s, *ts)

