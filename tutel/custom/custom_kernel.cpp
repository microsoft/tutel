// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>

#if defined(USE_GPU)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>
#else
#undef USE_NCCL
#endif

#if defined(USE_NCCL)
#include <nccl.h>
#endif

#include <regex>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#endif

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_CPU
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_LE(x, y) AT_ASSERTM((x) <= (y), "CHECK_LE fails.")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#if defined(USE_GPU)
namespace jit {

inline static std::string file_read(const char *path) {
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

inline static void file_write(const char *path, const std::string &code) {
  FILE *fp = fopen(path, "wb");
  CHECK_EQ(true, fp != nullptr);
  CHECK_EQ(code.size(), fwrite((void*)code.data(), 1, code.size(), fp));
  fclose(fp);
}

static std::string __sdk_home__;

static void update_sdk_home(const torch::Tensor &sdk_path) {
  CHECK_CPU(sdk_path);
  __sdk_home__ = static_cast<char*>(sdk_path.data_ptr());
}

inline std::string sdk_path(const std::string &rel = "") {
  static std::string cuda_home, cc;
  if (cuda_home.size() == 0) {
#if !defined(__HIP_PLATFORM_HCC__)
    cc = "bin/nvcc";
#else
    cc = "bin/hipcc";
#endif

#if defined(__linux__)
    cuda_home = __sdk_home__ + std::string("/");
#else
    cuda_home = __sdk_home__ + std::string("\\");
#endif
  }
  if (rel.size() > 0)
    return cuda_home + rel;
  return cuda_home + cc;
}

static std::string nvcc_compile(const char* code, const std::string &arch) {
#if defined(__linux__)
  char code_path[] = "/tmp/torch-tutel-XXXXXX.cu";
  CHECK_NE(-1, mkstemps(code_path, 3));

  file_write(code_path, code);
  std::string fatbin_path = code_path + std::string(".fatbin");

  std::string entry = sdk_path();
  if (access(entry.c_str(), F_OK) != 0) {
    LOG(FATAL) << "Failed to detect CUDA compiler file: " << entry << ", please set CUDA_HOME environment to configure CUDA SDK location correctly.";
    exit(1);
  }
  pid_t  pid = fork();
  if (pid == 0) {
#if !defined(__HIP_PLATFORM_HCC__)
    CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--fatbin", "-O4", "-gencode", ("arch=compute_" + arch + ",code=sm_" + arch).c_str(), (char *)NULL));
#else
    CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--genco", "-O4", "-w" , ("--amdgpu-target=" + arch).c_str(), (char *)NULL));
#endif
    exit(1);
  } else {
    wait(NULL);
  }
  auto image = file_read(fatbin_path.data());
  unlink(fatbin_path.data());
  unlink(code_path);
  return image;
#else
  return "";
#endif
}

static std::string nvrtc_compile(const char* code, const std::string &arch) {
#if !defined(__HIP_PLATFORM_HCC__)
  std::string arch_option = "--gpu-architecture=compute_" + arch, include_path = "--include-path=" + sdk_path("include");
  std::vector<const char*> param_cstrings = {"--restrict", include_path.c_str(), arch_option.c_str(), "--use_fast_math", "--extra-device-vectorization"};
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
    static bool once_flag = false;
    if (!once_flag) {
      once_flag = true;
      LOG(WARNING) << log << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another approach using CUDA compiler.. (To always disable NVRTC, please: export USE_NVRTC=0)";
    }
    CHECK_EQ(0, nvrtcDestroyProgram(&prog));
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
  std::string code, fname;
  dim3 blocks, threads;
};

static std::vector<ModuleConfig> _gms;

inline static CUfunction jit_activate(int fd, int dev) {
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

    int use_nvrtc = getenv("USE_NVRTC") ? std::atoi(getenv("USE_NVRTC")) : 0;
    std::string image;
    if (use_nvrtc || (image = nvcc_compile(source, arch)) == "") {
        image = nvrtc_compile(source, arch);
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
    gm.fname = fname;
    CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc[dev], hMod, fname.c_str()));
    CHECK_NE(nullptr, gm.hFunc[dev]);
  }

  return gm.hFunc[dev];
}

static void jit_execute(const std::vector<const void*> &ppargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
  CUfunction hfunc = jit_activate(fd, dev);
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, stream, (void**)ppargs.data(), nullptr));
}

static void jit_execute_with_values(const std::vector<const void*> &pargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
  std::vector<const void*> ppargs(pargs.size());
  for (int i = 0; i < ppargs.size(); ++i)
    ppargs[i] = &pargs[i];
  jit_execute(ppargs, fd, dev, blocks, threads, stream);
}

static int inject_source(const std::string &headless_code) {
  int fd = _gms.size();
  _gms.resize(fd + 1);

  auto &gm = _gms[fd];
#if !defined(__HIP_PLATFORM_HCC__)
  gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;
#else
  gm.code = "#include <hip/hip_runtime.h>\n" + headless_code;
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

static void invoke(const std::vector<torch::Tensor> &ts, const std::vector<long> &args, const std::vector<int> &blocks, int fd) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()], ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  if (blocks.size() == 0)
    jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else if (blocks.size() == 1)
    jit_execute(ppargs, fd, dev, dim3(blocks[0]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else if (blocks.size() == 2)
    jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else
    jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1], blocks[2]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
}

} // namespace jit
#endif

template<typename dtype> static void invoke_cpu(const std::vector<torch::Tensor> &ts, const std::vector<int> &extra, int kernel_type) {
  int samples = extra[0];
  int hidden = extra[1];
  int capacity = extra[2];
  dtype *gates1_s = static_cast<dtype*>(ts[0].data_ptr());
  int *indices1_s = static_cast<int*>(ts[1].data_ptr());
  int *locations1_s = static_cast<int*>(ts[2].data_ptr());
  dtype *reshaped_input = static_cast<dtype*>(ts[3].data_ptr());
  dtype *dispatched_input = static_cast<dtype*>(ts[4].data_ptr());

  for (int i = 0; i < (int)ts.size(); ++i)
    CHECK_CONTIGUOUS(ts[i]);

  if (kernel_type == 0) { //forward
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j] += gates1_s[i] * reshaped_input[i * (hidden) + j];
        }
      }
    }
  } else if (kernel_type == 1) { //backward_data
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
        }
      } else {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] = 0;
        }
      }
    }
  } else { //backward_gate
    for (int i = 0; i < samples; ++i) {
      gates1_s[i] = 0;
      if (locations1_s[i] >= capacity || indices1_s[i] < 0)
        continue;
      for (int j = 0; j < hidden; ++j) {
        gates1_s[i] += dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j] * reshaped_input[i * hidden + j];
      }
    }
  }
}

#if defined(USE_NCCL)

static ncclComm_t g_nccl_comm;
static std::vector<at::cuda::CUDAEvent> g_cuda_events;
static int g_world_size = 0;
static int g_world_rank = 0;
static int g_local_size = 0;
static int g_local_rank = 0;

// jit
static int mem_stride_copy_char_fd = -1;
static int mem_stride_copy_uint4_fd = -1;
static int mem_stride_copy_gridsize = 1;
static int mem_stride_copy_blocksize = 1;

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
    int max_num_split) {
  ncclUniqueId nccl_unique_id;

  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy(&nccl_unique_id, (void *)nccl_unique_id_tensor.data_ptr(), sizeof(ncclUniqueId));
  CHECK_EQ(0, ncclGroupStart());
  CHECK_EQ(0, ncclCommInitRank(&g_nccl_comm, world_size, nccl_unique_id, world_rank));
  CHECK_EQ(0, ncclGroupEnd());

  g_cuda_events.resize(max_num_split);
  g_world_size = world_size;
  g_world_rank = world_rank;

  if (const char* local_size = std::getenv("LOCAL_SIZE")) {
    g_local_size = std::atoi(local_size);
  } else {
    CHECK_EQ(0, cudaGetDeviceCount(&g_local_size));
  }
  CHECK_EQ(0, ncclCommCuDevice(g_nccl_comm, &g_local_rank));

  // jit for nccl
  if (mem_stride_copy_uint4_fd == -1) {
    std::string mem_stride_copy_cu = R"(
extern "C" __global__ void memStrideCopyKernel(
    $T *__restrict__ out, const $T *__restrict__ in,
    const size_t size, const int height, const int width) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < size * height * width; i += gridDim.x * blockDim.x) {
        const size_t index = i / size, offset = i % size;
        const size_t j = (width * (index % height) + (index / height)) * size + offset;
        out[j] = in[i];
    }
}
    )";
    mem_stride_copy_char_fd = jit::inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "char"));
    mem_stride_copy_uint4_fd = jit::inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "uint4"));
    CHECK_NE(-1, mem_stride_copy_char_fd);
    CHECK_NE(-1, mem_stride_copy_uint4_fd);
    CUfunction hfunc = jit::jit_activate(mem_stride_copy_uint4_fd, g_local_rank);
#if !defined(__HIP_PLATFORM_HCC__)
    CHECK_EQ(0, cuOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0, 0));
#else
    CHECK_EQ(0, hipModuleOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0));
#endif
  }
}

inline at::cuda::CUDAStream& get_default_stream() {
  static at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();
  return default_stream;
}

inline at::cuda::CUDAStream& get_nccl_stream() {
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

static torch::Tensor& nccl_stream_release(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].record(get_nccl_stream());
  return tensor;
}

static torch::Tensor& nccl_stream_acquire(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].block(get_nccl_stream());
  return tensor;
}

static std::vector<torch::Tensor> nccl_all_to_all_scatter_async(
    const torch::Tensor &input,
    torch::IntArrayRef output_size,
    int num_split,
    int num_slices_per_split,
    bool is_backward) {
  CHECK_CUDA(input);
  CHECK_LE(num_split, g_cuda_events.size());

  CHECK_EQ(0, num_slices_per_split % g_world_size);
  size_t length = input.nbytes();
  size_t num_slices = num_slices_per_split * num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;

  // Save original stream and switch to NCCL stream
  // Output tensors must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());

  std::vector<torch::Tensor> output_list(num_split);
  for (int i = 0; i < num_split; i++) {
    output_list[i] = torch::empty(output_size, input.device(), torch::MemoryFormat::Contiguous);
  }
  // NCCL stream allocator will add blocking event to computation stream after computation kernels
  for (auto& output : output_list) {
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), original_stream);
  }

  // Acquire 0-th event for single input
  g_cuda_events[0].block(get_nccl_stream());

  for (int i = 0; i < num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? num_split - 1 - i : i;

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input.data_ptr()) + (j * num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // Release calc_idx-th event
    g_cuda_events[calc_idx].record(get_nccl_stream());
  }

  // Switch to original stream
  at::cuda::setCurrentCUDAStream(original_stream);

  return output_list;
}

static torch::Tensor nccl_all_to_all_gather_async(
    const std::vector<torch::Tensor> &input_list,
    torch::IntArrayRef output_size,
    int num_split,
    int num_slices_per_split,
    bool is_backward) {
  CHECK_LE(num_split, g_cuda_events.size());
  CHECK_EQ(num_split, input_list.size());
  for (auto& input : input_list) {
    CHECK_CUDA(input);
  }

  CHECK_EQ(0, num_slices_per_split % g_world_size);

  // Save original stream and switch to NCCL stream
  // Output tensor must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  for (auto& input : input_list) {
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());
  }

  torch::Tensor output = torch::empty(output_size, input_list[0].device(), torch::MemoryFormat::Contiguous);
  size_t length = output.nbytes();
  size_t num_slices = num_slices_per_split * num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;
  // NCCL stream allocator will add blocking event to computation stream after computation kernels
  c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), original_stream);

  for (int i = 0; i < num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? num_split - 1 - i : i;

    // Acquire calc_idx-th event
    g_cuda_events[calc_idx].block(get_nccl_stream());

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output.data_ptr()) + (j * num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());
  }

  // Release 0-th event for single output
  g_cuda_events[0].record(get_nccl_stream());

  // Switch to original stream
  at::cuda::setCurrentCUDAStream(original_stream);

  return output;
}

static torch::Tensor nccl_all_to_all_2d_async(torch::Tensor &input) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);

  size_t length = input.nbytes();
  CHECK_EQ(0, length % g_world_size);
  size_t slice_size = length / g_world_size;
  size_t slice_size_uint4 = slice_size / sizeof(uint4);

  // Save original stream and switch to NCCL stream
  // Output tensors must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());

  int nranks = g_world_size, ngpus = g_local_size;
  CHECK_EQ(0, nranks % ngpus);
  int nnodes = nranks / ngpus;

  torch::Tensor tmp_output = torch::empty_like(input, torch::MemoryFormat::Contiguous);
  void* input_buff = (void*)input.data_ptr();
  void* tmp_output_buff = (void*)tmp_output.data_ptr();

  if (!(ngpus == 1 || nnodes == 1)) {
    int node_rank = g_world_rank / ngpus, local_rank = g_local_rank;

    // phase 0. per-gpu (ngpus) stride copy
    slice_size < sizeof(uint4)
      ? jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size, &ngpus, &nnodes}, mem_stride_copy_char_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream())
      : jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size_uint4, &ngpus, &nnodes}, mem_stride_copy_uint4_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream());

    // phase 1. intra-node alltoall
    CHECK_EQ(0, ncclGroupStart());
    for (int g = 0; g < ngpus; g++) {
      CHECK_EQ(0, ncclSend(((char*)tmp_output_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)input_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // phase 2. per-gpu (nnodes) stride copy
    slice_size < sizeof(uint4)
      ? jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size, &nnodes, &ngpus}, mem_stride_copy_char_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream())
      : jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size_uint4, &nnodes, &ngpus}, mem_stride_copy_uint4_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream());

    // phase 3. inter-node alltoall
    CHECK_EQ(0, ncclGroupStart());
    for (int n = 0; n < nnodes; n++) {
      CHECK_EQ(0, ncclSend(((char*)tmp_output_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)input_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // Switch to original stream
    at::cuda::setCurrentCUDAStream(original_stream);

    return input;
  } else {
    CHECK_EQ(0, ncclGroupStart());
    for (int r = 0; r < nranks; r++) {
      CHECK_EQ(0, ncclSend(((char*)input_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)tmp_output_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // NCCL stream allocator will add blocking event to computation stream after computation kernels
    c10::cuda::CUDACachingAllocator::recordStream(tmp_output.storage().data_ptr(), original_stream);

    // Switch to original stream
    at::cuda::setCurrentCUDAStream(original_stream);

    return tmp_output;
  }
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(USE_GPU)

    m.def("update_sdk_home",
        &jit::update_sdk_home,
        "Configure SDK HOME Path for GPU (CUDA)"
    );
    m.def("invoke",
        &jit::invoke,
        "Generic Invoke for GPU (CUDA)"
    );
    m.def("inject_source",
        &jit::inject_source,
        "Inject Source for GPU (CUDA)"
    );
#endif
    m.def("invoke_cpu_fp32",
        &invoke_cpu<float>,
        "Invoke for Sparse Ops (CPU)"
    );
    m.def("invoke_cpu_fp64",
        &invoke_cpu<double>,
        "Invoke for Sparse Ops (CPU)"
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
    m.def("nccl_stream_release",
        &nccl_stream_release,
        "Record CUDA event on NCCL stream to i-th event slot"
    );
    m.def("nccl_stream_acquire",
        &nccl_stream_acquire,
        "Let NCCL stream wait CUDA event in i-th event slot"
    );
    m.def("nccl_all_to_all_scatter_async",
        &nccl_all_to_all_scatter_async,
        "NCCL AllToAll (Scatter Async)"
    );
    m.def("nccl_all_to_all_gather_async",
        &nccl_all_to_all_gather_async,
        "NCCL AllToAll (Gather Async)"
    );
    m.def("nccl_all_to_all_2d_async",
        &nccl_all_to_all_2d_async,
        "NCCL AllToAll (2D Async, In-place if 2DH A2A is enabled)"
    );
#endif
}


#if defined(USE_GPU)
#include <torch/script.h>
#define DEFINE_KERNEL(x, y)  static int x = -1; if (x == -1) { x = y; }

torch::Tensor warp_cumsum(torch::Tensor x) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 2);
  x = x.to(torch::kInt32).contiguous();

  auto y = torch::empty_like(x);

  DEFINE_KERNEL(cumsum_fn, jit::inject_source(R"(
extern "C" __global__ void cumsum_fn(int* input0 /* (num_samples, batch_num) */, int* output0 /* (num_samples, batch_num) */, int num_samples) {
    #define thread_num  1024
    #define batch_num ((int)gridDim.x)

    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;

    for (int S = 0; S < num_samples; S += thread_num, output0 += thread_num * batch_num, input0 += thread_num * batch_num) {
        int offset = 1;
        if (S + thid < num_samples)
                temp[thid] = input0[thid * batch_num + bid];
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
        if (S + thid < num_samples)
                output0[thid * batch_num + bid] = temp[thid + 1] + last_sum;
        __syncthreads();
        last_sum += temp[thread_num];
    }
}
)"));

  jit::jit_execute_with_values({x.data_ptr(), y.data_ptr(), (void*)x.size(0)}, cumsum_fn, x.device().index(), x.size(1), 1024, nullptr);
  return y;
}

TORCH_LIBRARY(tutel_ops, m) {
  m.def("cumsum", warp_cumsum);
}
#endif
