// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

static std::string file_read(const char *path) {
  FILE *fp = fopen(path, "rb");
  CHECK_EQ(true, fp != nullptr);
  fseek(fp, 0, SEEK_END);
  size_t code_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  std::string code;
  code.resize(code_size);
  CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
  fclose(fp);
  return code;
}

static void file_write(const char *path, const std::string &code) {
  FILE *fp = fopen(path, "wb");
  CHECK_EQ(true, fp != nullptr);
  CHECK_EQ(code.size(), fwrite((void*)code.data(), 1, code.size(), fp));
  fclose(fp);
}

static std::string nvcc_compile(const char* code, const std::string &arch, int code_id, int dev_id) {
  std::string code_path = "/tmp/" + std::to_string(code_id) + "-" + std::to_string(dev_id) + ".cu";
  file_write(code_path.data(), code);
#if !defined(__HIP_PLATFORM_HCC__)
  CHECK_EQ(0, system(("/usr/local/cuda/bin/nvcc " + code_path + " -o " + code_path + ".fatbin --fatbin -O4 -gencode arch=compute_" + arch.substr(3) + ",code=" + arch).c_str()));
#else
  CHECK_EQ(0, system(("/opt/rocm/bin/hipcc " + code_path + " -o " + code_path + ".fatbin --genco -O4 -w --amdgpu-target=" + arch).c_str()));
#endif
  auto image = file_read((code_path + ".fatbin").data());
  remove((code_path + ".fatbin").data());
  return image;
}

static std::string nvrtc_compile(const char* code, const std::string &arch) {
  std::string arch_option = "--gpu-architecture=" + arch;
#if !defined(__HIP_PLATFORM_HCC__)
  std::vector<const char*> param_cstrings = {"--restrict", "--include-path=/usr/local/cuda/include", arch_option.c_str(), "--use_fast_math", "--extra-device-vectorization"};
#else
  std::vector<const char*> param_cstrings = {arch_option.c_str(), "-O4"};
#endif
  nvrtcProgram prog;

  CHECK_EQ(0, nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  CHECK_EQ(0, nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  CHECK_EQ(0, nvrtcGetProgramLog(prog, &log[0]));
  if (0 != res) {
    LOG(ERROR) << log << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another approach by external CUDA compiler..";
    return "";
  }

  size_t ptx_size;
  CHECK_EQ(0, nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  CHECK_EQ(0, nvrtcGetPTX(prog, &ptx[0]));
  CHECK_EQ(0, nvrtcDestroyProgram(&prog));
  return ptx;
}

struct ModuleConfig {
  CUmodule hMod = nullptr;
  CUfunction hFunc = nullptr;

  dim3 blocks, threads;
};

static std::vector<ModuleConfig> gpuModules;

static void invoke(const std::vector<torch::Tensor> &ts, int code_id) {
  auto &gm = gpuModules[code_id];
  std::vector<void*> pargs(ts.size()), ppargs(ts.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_CUDA(ts[i]);
    pargs[i] = (void*)ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }
  CHECK_EQ(0, cuLaunchKernel(gm.hFunc, gm.blocks.x, gm.blocks.y, gm.blocks.z, gm.threads.x, gm.threads.y, gm.threads.z, 0, nullptr, ppargs.data(), nullptr));
}

static void invoke_with_source(const std::vector<torch::Tensor> &ts, int code_id, int flags, const std::string &code) {

#if !defined(__HIP_PLATFORM_HCC__)
#if 0
  static void *libcuda = nullptr;
  static int (*cuModuleLoad)(...) = nullptr;
  static int (*cuModuleGetFunction)(...) = nullptr;
  static int (*cuLaunchKernel)(...) = nullptr;

  if (libcuda == nullptr) {
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/cuda/compat/lib/libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/cuda/compat/lib/libcuda.so", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/lib/x86_64-linux-gnu/libcuda.so", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/cuda/lib64/libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/cuda/lib64/libcuda.so", RTLD_LAZY | RTLD_GLOBAL)) : 0);
    (libcuda == nullptr ? (libcuda = dlopen("/usr/local/cuda/lib64/stubs/libcuda.so", RTLD_LAZY | RTLD_GLOBAL)) : 0);

    CHECK_NE(nullptr, libcuda);
    CHECK_NE(nullptr, (cuModuleLoad = (decltype(cuModuleLoad))dlsym(libcuda, "cuModuleLoad")));
    CHECK_NE(nullptr, (cuModuleGetFunction = (decltype(cuModuleGetFunction))dlsym(libcuda, "cuModuleGetFunction")));
    CHECK_NE(nullptr, (cuLaunchKernel = (decltype(cuLaunchKernel))dlsym(libcuda, "cuLaunchKernel")));
  }
#endif
#endif

  if (code_id >= (int)gpuModules.size())
    gpuModules.resize(code_id + 1);

  auto &gm = gpuModules[code_id];
  if (gm.hFunc == nullptr) {
    CHECK_CUDA(ts[0]);
    int dev = int(ts[0].device().index());
    CHECK_EQ(0, cudaSetDevice(dev));

#if !defined(__HIP_PLATFORM_HCC__)
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
#else
    hipDeviceProp_t prop;
    CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
#endif
    const char *source = code.data(), *pos, *tail;

    int no_nvrtc = flags & 1;
    std::string image;
    if (no_nvrtc || (image = nvrtc_compile(source, arch)) == "")
        image = nvcc_compile(source, arch, code_id, dev);

    long launch_bound;
    { char tag[] = " __launch_bounds__(";  pos = strstr(source, tag); launch_bound = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L; }

    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
    static void* values[] = {(void*)4L, (void*)launch_bound};
    CHECK_EQ(0, cuModuleLoadDataEx(&gm.hMod, image.c_str(), sizeof(options) / sizeof(*options), options, values));

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

  return invoke(ts, code_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke",
        &invoke,
        "Generic Invoke (CUDA)"
    );
    m.def("invoke_with_source",
        &invoke_with_source,
        "Generic Invoke with Source (CUDA)"
    );
}
