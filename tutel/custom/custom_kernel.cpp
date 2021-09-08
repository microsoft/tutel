// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


static void experts_gemm_forward(const std::vector<torch::Tensor> &ts, int algo_id) {
  // C[W, ..] += B[W, ..] * A[1, ..]
  auto &B = ts[0], &A = ts[1], &C = ts[2];
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  int ax = A.size(3), ay = A.size(2), bx = B.size(3), by = B.size(2);
  const int world = B.size(0), experts = B.size(1);
  CHECK_EQ(ay, bx);

  if (torch::kFloat32 == C.dtype()) {
    static const float alpha = 1.0f, beta = 0.0f;
    if (algo_id == 1) {
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, ax, by, ay,
          &alpha, ((float*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), 0, ((float*)B.data_ptr()) + i * B.size(2) * B.size(3), B.size(3), B.size(1) * B.size(2) * B.size(3),
          &beta, ((float*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3), C.size(1) * C.size(2) * C.size(3), world));
      }
    } else {
      for (int j = 0; j < world; ++j)
        for (int i = 0; i < experts; ++i) {
          CHECK_EQ(0, cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ax, by, ay,
            &alpha, ((float*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), ((float*)B.data_ptr()) + (j * experts + i) * B.size(2) * B.size(3), B.size(3),
            &beta, ((float*)C.data_ptr()) + (j * experts + i) * C.size(2) * C.size(3), C.size(3)));
        }
    }
  } else {
    CHECK_EQ(torch::kFloat16, C.dtype());

    static const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    if (algo_id == 1) {
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, ax, by, ay,
          &alpha, ((__half*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), 0, ((__half*)B.data_ptr()) + i * B.size(2) * B.size(3), B.size(3), B.size(1) * B.size(2) * B.size(3),
          &beta, ((__half*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3), C.size(1) * C.size(2) * C.size(3), world));
      }
    } else {
      for (int j = 0; j < world; ++j)
        for (int i = 0; i < experts; ++i) {
          CHECK_EQ(0, cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ax, by, ay,
            &alpha, ((half*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), ((half*)B.data_ptr()) + (j * experts + i) * B.size(2) * B.size(3), B.size(3),
            &beta, ((half*)C.data_ptr()) + (j * experts + i) * C.size(2) * C.size(3), C.size(3)));
        }
    }
  }
}

static void experts_gemm_backward_weight(const std::vector<torch::Tensor> &ts, int algo_id) {
  // C[1, ..] += B[W, ..].t() * A[W, ..]
  auto &B = ts[0], &A = ts[1], &C = ts[2];
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  int ax = A.size(3), ay = A.size(2), bx = B.size(2), by = B.size(3);
  const int world = B.size(0), experts = B.size(1);
  CHECK_EQ(ay, bx);

  if (torch::kFloat32 == C.dtype()) {
    static const float alpha = 1.0f, beta = 0.0f;
    for (int j = 0; j < world; ++j)
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ax, by, ay,
          &alpha, ((float*)A.data_ptr()) + (j * experts + i) * A.size(2) * A.size(3), A.size(3), ((float*)B.data_ptr()) + j * experts + i * B.size(2) * B.size(3), B.size(3),
          (j == 0 ? &beta : &alpha), ((float*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3)));
      }
  } else {
    CHECK_EQ(torch::kFloat16, C.dtype());

    static const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    for (int j = 0; j < world; ++j)
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ax, by, ay,
          &alpha, ((__half*)A.data_ptr()) + (j * experts + i) * A.size(2) * A.size(3), A.size(3), ((__half*)B.data_ptr()) + j * experts + i * B.size(2) * B.size(3), B.size(3),
          (j == 0 ? &beta : &alpha), ((__half*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3)));
      }
  }
}

static void experts_gemm_backward_data(const std::vector<torch::Tensor> &ts, int algo_id) {
  // C[W, ..] += B[W, ..] * A[1, ..].t()
  auto &B = ts[0], &A = ts[1], &C = ts[2];
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  int ax = A.size(2), ay = A.size(3), bx = B.size(3), by = B.size(2);
  const int world = B.size(0), experts = B.size(1);
  CHECK_EQ(ay, bx);

  if (torch::kFloat32 == C.dtype()) {
    static const float alpha = 1.0f, beta = 0.0f;
    if (algo_id == 1) {
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, ax, by, ay,
          &alpha, ((float*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), 0, ((float*)B.data_ptr()) + i * B.size(2) * B.size(3), B.size(3), B.size(1) * B.size(2) * B.size(3),
          &beta, ((float*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3), C.size(1) * C.size(2) * C.size(3), world));
      }
    } else {
      for (int j = 0; j < world; ++j)
        for (int i = 0; i < experts; ++i) {
          CHECK_EQ(0, cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ax, by, ay,
            &alpha, ((float*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), ((float*)B.data_ptr()) + (j * experts + i) * B.size(2) * B.size(3), B.size(3),
            &beta, ((float*)C.data_ptr()) + (j * experts + i) * C.size(2) * C.size(3), C.size(3)));
        }
    }
  } else {
    CHECK_EQ(torch::kFloat16, C.dtype());

    static const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    if (algo_id == 1) {
      for (int i = 0; i < experts; ++i) {
        CHECK_EQ(0, cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, ax, by, ay,
          &alpha, ((__half*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), 0, ((__half*)B.data_ptr()) + i * B.size(2) * B.size(3), B.size(3), B.size(1) * B.size(2) * B.size(3),
          &beta, ((__half*)C.data_ptr()) + i * C.size(2) * C.size(3), C.size(3), C.size(1) * C.size(2) * C.size(3), world));
      }
    } else {
      for (int j = 0; j < world; ++j)
        for (int i = 0; i < experts; ++i) {
          CHECK_EQ(0, cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ax, by, ay,
            &alpha, ((__half*)A.data_ptr()) + i * A.size(2) * A.size(3), A.size(3), ((__half*)B.data_ptr()) + (j * experts + i) * B.size(2) * B.size(3), B.size(3),
            &beta, ((__half*)C.data_ptr()) + (j * experts + i) * C.size(2) * C.size(3), C.size(3)));
        }
    }
  }
}

static void invoke(const std::vector<torch::Tensor> &ts, int _key) {
  struct ModuleConfig {
    CUmodule hMod = nullptr;
    CUfunction hFunc = nullptr;

    dim3 blocks, threads;
  };

  static std::vector<ModuleConfig> gpuMods;

  int key_int = (_key & 255), ctx = _key >> 8;
  if (ctx >= gpuMods.size())
    gpuMods.resize(ctx + 1);

  auto &gm = gpuMods[ctx];
  if (gm.hFunc == nullptr) {
    std::string key = std::to_string(key_int);
    std::string file_name = "/tmp/" + std::to_string(ctx) + "-" + key + ".cu";
    FILE *fp = fopen(file_name.c_str(), "rb");
    CHECK_EQ(true, fp != nullptr);
    fseek(fp, 0, SEEK_END);
    size_t code_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> code(code_size + 1);
    CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
    fclose(fp);

    int dev = key_int;
    CHECK_EQ(0, cudaSetDevice(dev));

#if !defined(__HIP_PLATFORM_HCC__)
    std::string cc = "30";
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);
    CHECK_EQ(0, system(("/usr/local/cuda/bin/nvcc " + file_name + " -o " + file_name + ".fatbin --fatbin -O2 -gencode arch=compute_" + arch  + ",code=sm_" + arch).c_str()));
#else
    hipDeviceProp_t prop;
    CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
    std::string arch = std::to_string(prop.gcnArch);
    CHECK_EQ(0, system(("/opt/rocm/bin/hipcc " + file_name + " -o " + file_name + ".fatbin --genco -O2 --amdgpu-target=gfx" + arch).c_str()));
#endif
    CHECK_EQ(0, cuModuleLoad(&gm.hMod, (file_name + ".fatbin").c_str()));

    const char *source = code.data(), *pos, *tail;
    CHECK_EQ(true, nullptr != (pos = strstr(source, " void ")));
    pos += 6; CHECK_EQ(true, nullptr != (tail = strchr(pos, '(')));

    CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc, gm.hMod, std::string(pos, tail - pos).c_str()));
    CHECK_EQ(true, nullptr != gm.hFunc);

    { char tag[] = "// [thread_extent] blockIdx.x = ";  pos = strstr(source, tag); gm.blocks.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
    { char tag[] = "// [thread_extent] blockIdx.y = ";  pos = strstr(source, tag); gm.blocks.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
    { char tag[] = "// [thread_extent] blockIdx.z = ";  pos = strstr(source, tag); gm.blocks.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
    { char tag[] = "// [thread_extent] threadIdx.x = "; pos = strstr(source, tag); gm.threads.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
    { char tag[] = "// [thread_extent] threadIdx.y = "; pos = strstr(source, tag); gm.threads.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
    { char tag[] = "// [thread_extent] threadIdx.z = "; pos = strstr(source, tag); gm.threads.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  }

  std::vector<void*> pargs(ts.size()), ppargs(ts.size());
  for (int i = 0; i < ts.size(); ++i) {
    pargs[i] = (void*)ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }

  CHECK_EQ(0, cuLaunchKernel(gm.hFunc, gm.blocks.x, gm.blocks.y, gm.blocks.z, gm.threads.x, gm.threads.y, gm.threads.z, 0, nullptr, ppargs.data(), nullptr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke",
        &invoke, 
        "Generic Invoke (CUDA)"
    );

    m.def("experts_gemm_forward",
        &experts_gemm_forward,
        "Experts Gemm Forward (CUDA)"
    );

    m.def("experts_gemm_backward_data",
        &experts_gemm_backward_data,
        "Experts Gemm Backward Data (CUDA)"
    );

    m.def("experts_gemm_backward_weight",
        &experts_gemm_backward_weight,
        "Experts Gemm Backward Weight (CUDA)"
    );
}
