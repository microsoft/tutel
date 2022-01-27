// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>

#if defined(USE_NCCL)
#include <nccl.h>
#endif

#include <vector>
#include <pwd.h>
#include <sys/wait.h>

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_CPU
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

namespace jit {

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

static std::string get_cache_path() {
  char *home_path;
  struct stat st = {0};
  if ((home_path = getenv("HOME")) == NULL) {
    home_path = getpwuid(getuid())->pw_dir;
  }
  std::string cache_path(home_path);
  cache_path += "/.cache/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }
  cache_path += "tutel/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }
  cache_path += "kernels/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }

  return cache_path;
}

static std::string nvcc_compile(const char* code, const std::string &arch) {
  char code_path[] = "/tmp/torch-tutel-XXXXXX.cu";
  CHECK_NE(-1, mkstemps(code_path, 3));

  file_write(code_path, code);
  std::string fatbin_path = code_path + std::string(".fatbin");

  pid_t  pid = fork();
  if (pid == 0) {
#if !defined(__HIP_PLATFORM_HCC__)
    CHECK_EQ(-1, execl("/usr/local/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc", code_path, "-o", fatbin_path.c_str(), "--fatbin", "-O4", "-gencode", ("arch=compute_" + arch + ",code=sm_" + arch).c_str(), (char *)NULL));
#else
    CHECK_EQ(-1, execl("/opt/rocm/bin/hipcc", "/opt/rocm/bin/hipcc", code_path, "-o", fatbin_path.c_str(), "--genco", "-O4", "-w" , ("--amdgpu-target=" + arch).c_str(), (char *)NULL));
#endif
    exit(1);
  } else {
    wait(NULL);
  }
  auto image = file_read(fatbin_path.data());
  unlink(fatbin_path.data());
  unlink(code_path);
  return image;
}

static std::string nvrtc_compile(const char* code, const std::string &arch) {
#if !defined(__HIP_PLATFORM_HCC__)
  std::string arch_option = "--gpu-architecture=compute_" + arch;
  std::vector<const char*> param_cstrings = {"--restrict", "--include-path=/usr/local/cuda/include", arch_option.c_str(), "--use_fast_math", "--extra-device-vectorization"};
#else
  std::string arch_option = "--gpu-architecture=" + arch;
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
    LOG(ERROR) << log << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another approach using CUDA compiler.. (To always disable NVRTC, please: export USE_NVRTC=0)";
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
  // Handling JIT compilation in Multi-gpu cases
  std::vector<CUfunction> hFunc;
  std::string code;
  dim3 blocks, threads;
};

static std::vector<ModuleConfig> _gms;

static void jit_execute(const std::vector<const void*> &ppargs, int fd, int dev, const dim3 &blocks, const dim3 &threads) {
  auto &gm = _gms[fd];
  if (gm.hFunc.size() <= dev)
    gm.hFunc.resize(dev + 1);

  if (gm.hFunc[dev] == nullptr) {
#if !defined(__HIP_PLATFORM_HCC__)
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);
#else
    hipDeviceProp_t prop;
    CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
#endif
    const char *source = gm.code.data(), *pos, *tail;

    int use_nvrtc = getenv("USE_NVRTC") ? std::atoi(getenv("USE_NVRTC")) : 1;
    std::string image;
    if (!use_nvrtc || (image = nvrtc_compile(source, arch)) == "") {
        image = nvcc_compile(source, arch);
    }

    long launch_bound;
    { char tag[] = " __launch_bounds__(";  const char *pos = strstr(source, tag); launch_bound = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L; }

    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
    static void* values[] = {(void*)4L, (void*)launch_bound};

    CUmodule hMod = nullptr;
    CHECK_EQ(0, cuModuleLoadDataEx(&hMod, image.c_str(), sizeof(options) / sizeof(*options), options, values));
    CHECK_NE(nullptr, hMod);

    CHECK_NE(nullptr, (pos = strstr(source, " void ")));
    pos += 6; CHECK_NE(nullptr, (tail = strchr(pos, '(')));

    std::string fname = std::string(pos, tail - pos);
    CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc[dev], hMod, fname.c_str()));
    CHECK_NE(nullptr, gm.hFunc[dev]);
  }

  CHECK_EQ(0, cuLaunchKernel(gm.hFunc[dev], blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, nullptr, (void**)ppargs.data(), nullptr));
}

static int inject_source(const std::string &headless_code) {
  int fd = _gms.size();
  _gms.resize(fd + 1);

  auto &gm = _gms[fd];
#if !defined(__HIP_PLATFORM_HCC__)
  gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;
#else
  gm.code = "#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n" + headless_code;
#endif

  const char *source = headless_code.c_str();
  { char tag[] = "// [thread_extent] blockIdx.x = ";  const char *pos = strstr(source, tag); gm.blocks.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] blockIdx.y = ";  const char *pos = strstr(source, tag); gm.blocks.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] blockIdx.z = ";  const char *pos = strstr(source, tag); gm.blocks.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.x = "; const char *pos = strstr(source, tag); gm.threads.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.y = "; const char *pos = strstr(source, tag); gm.threads.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.z = "; const char *pos = strstr(source, tag); gm.threads.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }

  return fd;
}

static void invoke(const std::vector<torch::Tensor> &ts, int fd) {
#if 0
  int fd0 = inject_source(R"(
extern "C" __global__ void helloworld(float *data, int N) {
  for (int i = blockIdx.x; i < N; i += gridDim.x)
    data[i] = i + 1.2f;
}
)");
  float *d_data, h_data[1]; int N = 1024;
  cudaMalloc(&d_data, N * sizeof(*d_data));

  jit::jit_execute({&d_data, &N}, fd0, ts[0].device().index(), dim3(4, 1, 1), dim3(1, 1, 1));
  cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(nullptr);
  printf("@@ %f\n", *h_data);
#endif

  std::vector<const void*> pargs(ts.size()), ppargs(ts.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads);
}

} // namespace jit

template<typename dtype> static void invoke_cpu(const std::vector<torch::Tensor> &ts, const int &kernel_type, const int &capacity) {
  int samples = ts[1].sizes()[0];
  int hidden = ts[3].sizes()[1];
  if (kernel_type == 0) { //forward
    for (int i = 0; i < samples; ++i) {
      if ((ts[2][i].item<int>() < capacity) && (ts[1][i].item<int>() >= 0)) {
        for (int j = 0; j < hidden; ++j) {
          if (ts[0].sizes().size() == 1) {
            ts[4][ts[1][i].item<int>() * capacity + ts[2][i].item<int>()][j] += ts[0][i].item<dtype>() * ts[3][i][j].item<dtype>();
          } else {
            ts[4][ts[1][i].item<int>() * capacity + ts[2][i].item<int>()][j] += ts[0][i][0].item<dtype>() * ts[3][i][j].item<dtype>();
          }
        }
      }
    }
  } else if (kernel_type == 1) { //backward_data
    for (int i = 0; i < samples; ++i) {
      if ((ts[2][i].item<int>() < capacity) && (ts[1][i].item<int>() >= 0)) {
        for (int j = 0; j < hidden; ++j) {
          if (ts[0].sizes().size() == 1) {
            ts[3][i][j] = ts[0][i].item<dtype>() * ts[4][ts[1][i].item<int>() * capacity + ts[2][i].item<int>()][j];
          } else {
            ts[3][i][j] = ts[0][i][0].item<dtype>() * ts[4][ts[1][i].item<int>() * capacity + ts[2][i].item<int>()][j];
          }
        }
      } else {
        for (int j = 0; j < hidden; ++j) {
          ts[4][i][j] = 0;
        }
      }
    }
  } else { //backward_gate
    for (int block = 0; block < samples; ++block) {
      ts[0][block] = 0;
      dtype grad_gates1_s_rf = 0.0;
      for (int thread = 0; thread < 32; ++thread) {
        if (ts[2][block].item<int>() >= capacity || ts[1][block].item<int>() < 0) {
          if (thread == 0)
            if (ts[0].sizes().size() == 1)
              ts[0][block] = 0;
            else
              ts[0][block][0] = 0;
          return;
        }
        int indice = ts[1][block].item<int>() * capacity + ts[2][block].item<int>();
        for (int i = thread; i < hidden; i += 32)
          grad_gates1_s_rf += ts[4][indice][i].item<dtype>() * ts[3][block][i].item<dtype>();
      }
      ts[0][block] = grad_gates1_s_rf;
    }
  }
}

#if defined(USE_NCCL)

static ncclComm_t g_nccl_comm;
static std::vector<at::cuda::CUDAEvent> g_cuda_events;
static int g_num_split = 0;
static int g_num_slices_per_split = 0;
static int g_world_size = 0;

static size_t get_nccl_unique_id_size() {
  return sizeof(ncclUniqueId);
}

static void get_nccl_unique_id(torch::Tensor &nccl_unique_id_tensor) {
  ncclUniqueId nccl_unique_id;

  CHECK_EQ(0, ncclGetUniqueId(&nccl_unique_id));
  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy((void *)nccl_unique_id_tensor.data_ptr(), &nccl_unique_id, sizeof(ncclUniqueId));
}

static void init_nccl(
    const torch::Tensor &nccl_unique_id_tensor,
    int world_size,
    int world_rank,
    int num_split,
    int num_slices_per_split) {
  ncclUniqueId nccl_unique_id;

  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy(&nccl_unique_id, (void *)nccl_unique_id_tensor.data_ptr(), sizeof(ncclUniqueId));
  CHECK_EQ(0, ncclGroupStart());
  CHECK_EQ(0, ncclCommInitRank(&g_nccl_comm, world_size, nccl_unique_id, world_rank));
  CHECK_EQ(0, ncclGroupEnd());

  g_num_split = num_split;
  g_cuda_events.resize(num_split);
  g_num_slices_per_split = num_slices_per_split;
  g_world_size = world_size;
}

static at::cuda::CUDAStream& get_nccl_stream() {
  static at::cuda::CUDAStream nccl_stream = at::cuda::getStreamFromPool();
  return nccl_stream;
}

static torch::Tensor& current_stream_release(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].record(at::cuda::getCurrentCUDAStream());
  return tensor;
}

static torch::Tensor& current_stream_acquire(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].block(at::cuda::getCurrentCUDAStream());
  return tensor;
}

static void nccl_all_to_all_scatter_async(
    const torch::Tensor &input,
    std::vector<torch::Tensor> &output_list,
    bool is_backward) {
  CHECK_CUDA(input);
  CHECK_EQ(g_num_split, output_list.size());
  for (auto& output : output_list) {
    CHECK_CUDA(output);
  }

  CHECK_EQ(0, g_num_slices_per_split % g_world_size);
  size_t length = input.nbytes();
  size_t num_slices = g_num_slices_per_split * g_num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;

  // Allocator will add blocking event to nccl stream after nccl kernels
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());
  for (auto& output : output_list) {
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), get_nccl_stream());
  }

  // Acquire 0-th event for single input
  g_cuda_events[0].block(get_nccl_stream());

  for (int i = 0; i < g_num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? g_num_split - 1 - i : i;

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < g_num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input.data_ptr()) + (j * g_num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / g_num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / g_num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // Release calc_idx-th event
    g_cuda_events[calc_idx].record(get_nccl_stream());
  }
}

static void nccl_all_to_all_gather_async(
    const std::vector<torch::Tensor> &input_list,
    torch::Tensor &output,
    bool is_backward) {
  CHECK_EQ(g_num_split, input_list.size());
  for (auto& input : input_list) {
    CHECK_CUDA(input);
  }
  CHECK_CUDA(output);

  CHECK_EQ(0, g_num_slices_per_split % g_world_size);
  size_t length = output.nbytes();
  size_t num_slices = g_num_slices_per_split * g_num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;

  // Allocator will add blocking event to nccl stream after nccl kernels
  for (auto& input : input_list) {
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());
  }
  c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), get_nccl_stream());

  for (int i = 0; i < g_num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? g_num_split - 1 - i : i;

    // Acquire calc_idx-th event
    g_cuda_events[calc_idx].block(get_nccl_stream());

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < g_num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / g_num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output.data_ptr()) + (j * g_num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / g_num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());
  }

  // Release 0-th event for single output
  g_cuda_events[0].record(get_nccl_stream());
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke",
        &jit::invoke,
        "Generic Invoke for GPU (CUDA)"
    );
    m.def("inject_source",
        &jit::inject_source,
        "Inject Source for GPU (CUDA)"
    );
    m.def("invoke_cpu_fp32",
        &invoke_cpu<float>,
        "Generic Invoke (CPU)"
    );
    m.def("invoke_cpu_fp64",
        &invoke_cpu<double>,
        "Generic Invoke (CPU)"
    );
#if defined(USE_NCCL)
    m.def("get_nccl_unique_id_size",
        &get_nccl_unique_id_size,
        "Get size of ncclUniqueId in bytes"
    );
    m.def("get_nccl_unique_id",
        &get_nccl_unique_id,
        "Get ncclUniqueId for NCCL initialization"
    );
    m.def("init_nccl",
        &init_nccl,
        "NCCL initialization"
    );
    m.def("current_stream_release",
        &current_stream_release,
        "Record CUDA event on current stream to i-th event slot"
    );
    m.def("current_stream_acquire",
        &current_stream_acquire,
        "Let current stream wait CUDA event in i-th event slot"
    );
    m.def("nccl_all_to_all_scatter_async",
        &nccl_all_to_all_scatter_async,
        "NCCL AllToAll (Scatter Async)"
    );
    m.def("nccl_all_to_all_gather_async",
        &nccl_all_to_all_gather_async,
        "NCCL AllToAll (Gather Async)"
    );
#endif
}
