import torch
from ..jit import JitKernel

from ..moe_layer import shared_data as s_cfg

if s_cfg.message_dtype == torch.float16:
    str_dtype = '__half2'
elif s_cfg.message_dtype == torch.float32:
    str_dtype = 'float'
else:
    raise Exception("Unrecognized data type: %s" % s_cfg.message_dtype)

func_fwd = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)
#define __dtype @dtype@

extern "C" __global__ __launch_bounds__(1024) void forward_dispatched_input(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
      if (locations1_s[i] < capacity) {
          #pragma unroll
          for (int j = threadIdx.x; j < hidden; j += 1024)
              atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], gates1_s[i] * reshaped_input[i * (hidden) + j]);
      }
}
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.aligned_dim)).replace('@dtype@', str(str_dtype)))


func_bwd_gate = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)
#define __dtype @dtype@

extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(__dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, void* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = @samples@
  // [thread_extent] threadIdx.x = 32
  if (locations1_s[blockIdx.x] >= capacity) {
    if (((int)threadIdx.x) == 0)
#if @IS_FLOAT@
      ((float*)grad_gates1_s)[(((int)blockIdx.x))] = 0;
#else
      ((half*)grad_gates1_s)[(((int)blockIdx.x))] = __float2half_rn(0.000000e+00f);
#endif
    return;
  }
  int indice = indices1_s[(int)blockIdx.x] * capacity + locations1_s[(int)blockIdx.x];
#if @IS_FLOAT@
  __dtype grad_gates1_s_rf = 0.000000e+00f;
#else
  __dtype grad_gates1_s_rf = __dtype(0, 0);
#endif
  for (int i = threadIdx.x; i < hidden; i += 32)
    grad_gates1_s_rf += dispatched_input[indice * (hidden) + i] * reshaped_input[((int)blockIdx.x) * (hidden) + i];

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
  if (((int)threadIdx.x) == 0)
#if @IS_FLOAT@
    ((float*)grad_gates1_s)[(((int)blockIdx.x))] = red_buf0[(0)];
#else
    ((half*)grad_gates1_s)[(((int)blockIdx.x))] = red_buf0[(0)].x + red_buf0[(0)].y;
#endif
}
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.aligned_dim)).replace('@dtype@', str(str_dtype)).replace('@IS_FLOAT@', '1' if str_dtype == 'float' else '0'))

func_bwd_data = JitKernel.create('''
#define capacity (@capacity@)
#define samples (@samples@)
#define hidden (@hidden@)
#define __dtype @dtype@

extern "C" __global__ __launch_bounds__(1024) void template_op_kernel0(__dtype* __restrict__ gates1_s, __dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
      if (locations1_s[i] < capacity) {
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
'''.replace('@capacity@', str(s_cfg.capacity)).replace('@samples@', str(s_cfg.samples)).replace('@hidden@', str(s_cfg.aligned_dim)).replace('@dtype@', str(str_dtype)).replace('@IS_FLOAT@', '1' if str_dtype == 'float' else '0'))

