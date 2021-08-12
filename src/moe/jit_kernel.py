import torch
import custom_kernel

try:
	from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
	IS_HIP_EXTENSION = False

class JitKernel:
    @staticmethod
    def create(source):
        if not hasattr(JitKernel, '__CTX__'):
            torch.cuda.init()
            JitKernel.__CTX__ = 0
        __ctx__ = JitKernel.__CTX__
        JitKernel.__CTX__ += 1
        with open(f'/tmp/{__ctx__}.cu', 'w') as fp:
            if IS_HIP_EXTENSION:
              fp.write('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n')
            else:
              fp.write('#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n')
            fp.write(source)

        def func(*inputs):
            custom_kernel.invoke(inputs, __ctx__)
        return func

func_fwd = JitKernel.create('''
extern "C" __global__ __launch_bounds__(64) void forward_dispatched_input(float* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 64
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 8
  // [thread_extent] threadIdx.y = 64
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[(((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))])) : 0.000000e+00f);
    ((locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))], (gates1_s[(((((int)blockIdx.x) * 32) + vthread_s))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))])) : 0.000000e+00f);
  }
  for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[(((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32768))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32832))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32896))])) : 0.000000e+00f);
    ((locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192))], (gates1_s[((((((int)blockIdx.x) * 32) + vthread_s1) + 16))] * reshaped_input[((((((((int)blockIdx.x) * 65536) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 32960))])) : 0.000000e+00f);
  }
}
''')

if not IS_HIP_EXTENSION:
    func_bwd_gate = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void backward_gates1_s(float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.x = 32
  float grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = 0.000000e+00f;
  for (int HIDDEN_outer = 0; HIDDEN_outer < 64; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((int)blockIdx.x))] < 1024) ? (dispatched_input[(((((indices1_s[(((int)blockIdx.x))] * 2097152) + (locations1_s[(((int)blockIdx.x))] * 2048)) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))] * reshaped_input[((((((int)blockIdx.x) * 2048) + (HIDDEN_outer * 32)) + ((int)threadIdx.x)))]) : 0.000000e+00f));
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
''')
else:
    func_bwd_gate = JitKernel.create('''
extern "C" __global__ __launch_bounds__(64) void template_op_kernel0(float* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ reshaped_input, float* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.x = 64
  float grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = 0.000000e+00f;
  for (int HIDDEN_outer = 0; HIDDEN_outer < 32; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((int)blockIdx.x))] < 1024) ? (dispatched_input[(((((indices1_s[(((int)blockIdx.x))] * 2097152) + (locations1_s[(((int)blockIdx.x))] * 2048)) + (HIDDEN_outer * 64)) + ((int)threadIdx.x)))] * reshaped_input[((((((int)blockIdx.x) * 2048) + (HIDDEN_outer * 64)) + ((int)threadIdx.x)))]) : 0.000000e+00f));
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
''')

func_bwd_data = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void backward_reshaped_input(float* __restrict__ dispatched_input, float* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, float* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 256
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 32
  // [thread_extent] threadIdx.y = 32
  grad_reshaped_input[((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))] = ((locations1_s[((((int)blockIdx.x) * 8))] < 1024) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[(((((indices1_s[((((int)blockIdx.x) * 8))] * 2097152) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))] = ((locations1_s[((((int)blockIdx.x) * 8))] < 1024) ? (gates1_s[((((int)blockIdx.x) * 8))] * dispatched_input[((((((indices1_s[((((int)blockIdx.x) * 8))] * 2097152) + (locations1_s[((((int)blockIdx.x) * 8))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2048))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2080))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 1))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 1))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 1))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 1))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4096))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4128))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 2))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 2))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 2))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 2))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6144))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 6176))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 3))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 3))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 3))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 3))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8192))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8224))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 4))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 4))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 4))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 4))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10240))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 10272))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 5))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 5))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 5))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 5))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12288))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12320))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 6))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 6))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 6))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 6))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14336))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]) : 0.000000e+00f);
  grad_reshaped_input[(((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 14368))] = ((locations1_s[(((((int)blockIdx.x) * 8) + 7))] < 1024) ? (gates1_s[(((((int)blockIdx.x) * 8) + 7))] * dispatched_input[((((((indices1_s[(((((int)blockIdx.x) * 8) + 7))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 8) + 7))] * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]) : 0.000000e+00f);
}
''')

fwd_cumsum = JitKernel.create('''

#define tensor_cnt  (4096)
#define thread_num  (1024)
#define batch_num   (2)
#define capacity    (1024)
#define __out__

extern "C" __global__ __launch_bounds__(thread_num) void cumsum(int* __restrict__ indices1_s, __out__ int* __restrict__ locations1_s) {
    // HINT: blockIdx.x, threadIdx.x = batch_num, thread_num

    // [thread_extent] blockIdx.x = 2
    // [thread_extent] threadIdx.x = 1024

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
        if (bid == indices1_s[thid]) {
          locations1_s[thid] = temp[thid + 1] + last_sum;
        }
        last_sum += temp[thread_num];
    }
}
''')
