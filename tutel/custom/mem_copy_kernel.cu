#include <stdint.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#undef CHECK_EQ
#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")


static int g_mem_stride_copy_gridsize = 0;
static int g_mem_stride_copy_blocksize = 0;

template <typename T>
__global__ void memStrideCopyKernel(
    T *__restrict__ out, const T *__restrict__ in,
    const uint64_t size, const uint64_t height, const uint64_t width) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t i = tid; i < size * height * width; i += gridDim.x * blockDim.x) {
        const uint64_t index = i / size, offset = i % size;
        const uint64_t j = (width * (index % height) + (index / height)) * size + offset;
        out[j] = in[i];
    }
}

cudaError_t memStrideCopy(
    void* dst, const void* src, const size_t slice_size,
    const size_t height, const size_t width,
    cudaStream_t stream) {
    if (g_mem_stride_copy_gridsize == 0 || g_mem_stride_copy_blocksize == 0) {
        CHECK_EQ(cudaSuccess, cudaOccupancyMaxPotentialBlockSize(
            &g_mem_stride_copy_gridsize, &g_mem_stride_copy_blocksize, memStrideCopyKernel<uint4>));
    }

    if (slice_size < sizeof(uint4)) {
        memStrideCopyKernel<char><<<g_mem_stride_copy_gridsize, g_mem_stride_copy_blocksize, 0, stream>>>(
            (char*)dst, (char*)src, slice_size, height, width);
    } else {
        memStrideCopyKernel<uint4><<<g_mem_stride_copy_gridsize, g_mem_stride_copy_blocksize, 0, stream>>>(
            (uint4*)dst, (uint4*)src, slice_size / sizeof(uint4), height, width);
    }
    return cudaGetLastError();
}
