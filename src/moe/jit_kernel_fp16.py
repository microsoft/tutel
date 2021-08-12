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

def func_fwd(*inputs):
  func_fwd_stage = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, half* __restrict__ reshaped_input, float* __restrict__ dispatched_input) {
  // [thread_extent] blockIdx.x = 2
  // [thread_extent] threadIdx.x = 2
  // [thread_extent] blockIdx.y = 32
  // [thread_extent] threadIdx.y = 16
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    for (int vthread_s1 = 0; vthread_s1 < 32; ++vthread_s1) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] < 1024) ? atomicAdd(&dispatched_input[(((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * 2097152) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s * 64)) + (((int)threadIdx.x) * 32)) + vthread_s1))] * reshaped_input[(((((((((int)blockIdx.x) * 2097152) + (vthread_s * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s1 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s2 = 0; vthread_s2 < 16; ++vthread_s2) {
    for (int vthread_s3 = 0; vthread_s3 < 32; ++vthread_s3) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * 2097152) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 1))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s2 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s3))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s2 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s3 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 1))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s4 = 0; vthread_s4 < 16; ++vthread_s4) {
    for (int vthread_s5 = 0; vthread_s5 < 32; ++vthread_s5) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * 2097152) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 32))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s4 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s5))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s4 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s5 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 32))])) : 0.000000e+00f);
    }
  }
  for (int vthread_s6 = 0; vthread_s6 < 16; ++vthread_s6) {
    for (int vthread_s7 = 0; vthread_s7 < 32; ++vthread_s7) {
      ((locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] < 1024) ? atomicAdd(&dispatched_input[((((((indices1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * 2097152) + (locations1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 33))], __half2float(gates1_s[(((((((int)blockIdx.x) * 1024) + (vthread_s6 * 64)) + (((int)threadIdx.x) * 32)) + vthread_s7))] * reshaped_input[((((((((((int)blockIdx.x) * 2097152) + (vthread_s6 * 131072)) + (((int)threadIdx.x) * 65536)) + (vthread_s7 * 2048)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + 33))])) : 0.000000e+00f);
    }
  }
}
''')
  inputs = list(inputs)

  output = inputs[-1]
  inputs[-1] = torch.zeros(output.shape, dtype=torch.float32, device=output.device)

  func_fwd_stage(*inputs)
  output.copy_(inputs[-1].to(output.dtype))


func_bwd_gate = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, half* __restrict__ reshaped_input, half* __restrict__ grad_gates1_s) {
  // [thread_extent] blockIdx.x = 1024
  // [thread_extent] threadIdx.y = 2
  // [thread_extent] threadIdx.x = 16
  half grad_gates1_s_rf[1];
  grad_gates1_s_rf[(0)] = __float2half_rn(0.000000e+00f);
  for (int HIDDEN_outer = 0; HIDDEN_outer < 128; ++HIDDEN_outer) {
    grad_gates1_s_rf[(0)] = (grad_gates1_s_rf[(0)] + ((locations1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] < 1024) ? (dispatched_input[(((((indices1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] * 2097152) + (locations1_s[(((((int)blockIdx.x) * 2) + ((int)threadIdx.y)))] * 2048)) + (HIDDEN_outer * 16)) + ((int)threadIdx.x)))] * reshaped_input[(((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 2048)) + (HIDDEN_outer * 16)) + ((int)threadIdx.x)))]) : __float2half_rn(0.000000e+00f)));
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
''')

func_bwd_data = JitKernel.create('''
extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ dispatched_input, half* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, half* __restrict__ grad_reshaped_input) {
  // [thread_extent] blockIdx.x = 16
  // [thread_extent] threadIdx.x = 8
  // [thread_extent] blockIdx.y = 256
  // [thread_extent] threadIdx.y = 4
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    grad_reshaped_input[((((((((int)blockIdx.x) * 262144) + (vthread_s * 16384)) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.y) * 2)))] = ((locations1_s[((((((int)blockIdx.x) * 128) + (vthread_s * 8)) + ((int)threadIdx.x)))] < 1024) ? (gates1_s[((((((int)blockIdx.x) * 128) + (vthread_s * 8)) + ((int)threadIdx.x)))] * dispatched_input[(((((indices1_s[((((((int)blockIdx.x) * 128) + (vthread_s * 8)) + ((int)threadIdx.x)))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 128) + (vthread_s * 8)) + ((int)threadIdx.x)))] * 2048)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.y) * 2)))]) : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
    grad_reshaped_input[(((((((((int)blockIdx.x) * 262144) + (vthread_s1 * 16384)) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.y) * 2)) + 1))] = ((locations1_s[((((((int)blockIdx.x) * 128) + (vthread_s1 * 8)) + ((int)threadIdx.x)))] < 1024) ? (gates1_s[((((((int)blockIdx.x) * 128) + (vthread_s1 * 8)) + ((int)threadIdx.x)))] * dispatched_input[((((((indices1_s[((((((int)blockIdx.x) * 128) + (vthread_s1 * 8)) + ((int)threadIdx.x)))] * 2097152) + (locations1_s[((((((int)blockIdx.x) * 128) + (vthread_s1 * 8)) + ((int)threadIdx.x)))] * 2048)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.y) * 2)) + 1))]) : __float2half_rn(0.000000e+00f));
  }
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
