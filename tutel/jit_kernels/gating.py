import torch
import torch.nn.functional as F

from ..jit import JitKernel

def get_gating_kernel(batched_tokens, global_experts):
    global GATING_FUNC
    try:
        return GATING_FUNC
    except:
        import os
        if int(os.environ.get('GATE', '1')) == 0 or (batched_tokens & (batched_tokens - 1)) != 0:
            print('[WARN]', f"`batched_tokens` (= {batched_tokens}) isn't in the form of 2^k, which is outside optimization scope and may result in big performance regression.")
            def general_gating(indices1_s, locations1_s):
                # Un-fused Version
                mask1 = F.one_hot(indices1_s.to(torch.int64), num_classes=global_experts)
                locations1 = torch.cumsum(mask1, dim=0) - 1
                _locations1_s = torch.sum(locations1 * mask1, dim=1).to(torch.int32)
                locations1_s.copy_(_locations1_s)
            return general_gating

        tensor_cnt = batched_tokens * global_experts
        thread_num = min(1024, batched_tokens)
        batch_num = global_experts

        GATING_FUNC = JitKernel.create('''
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
'''.replace('@tensor_cnt@', str(tensor_cnt)).replace('@thread_num@', str(thread_num)).replace('@batch_num@', str(batch_num)))

        return GATING_FUNC

