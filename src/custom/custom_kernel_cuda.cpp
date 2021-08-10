#include <torch/extension.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


static CUmodule NVRTCCompile(const char *code) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }

  compile_params.push_back("-arch=compute_" + cc);

  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }
  CHECK_EQ(0, nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  CHECK_EQ(0, nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  CHECK_EQ(0, nvrtcGetProgramLog(prog, &log[0]));
  if (0 != compile_res) {
    fprintf(stderr, "[ERROR] %s\n", log.c_str());
    CHECK_EQ(0, compile_res);
  }
  size_t ptx_size;
  CHECK_EQ(0, nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  CHECK_EQ(0, nvrtcGetPTX(prog, &ptx[0]));
  CHECK_EQ(0, nvrtcDestroyProgram(&prog));

  CUmodule hMod;
  CHECK_EQ(0, cuModuleLoadData(&hMod, ptx.c_str()));
  return hMod;
}

std::vector<torch::Tensor> invoke(const std::vector<torch::Tensor> &ts, int ctx) {
  struct ModuleConfig {
    CUmodule hMod = nullptr;
    CUfunction hFunc = nullptr;

    dim3 blocks, threads;
  };
  static std::vector<ModuleConfig> gpuMods;
  if (ctx >= gpuMods.size())
    gpuMods.resize(ctx + 1);

  auto &gm = gpuMods[ctx];
  if (gm.hFunc == nullptr) {
    FILE *fp = fopen(("/tmp/" + std::to_string(ctx) + ".cu").c_str(), "rb");
    CHECK_EQ(true, fp != nullptr);
    fseek(fp, 0, SEEK_END);
    size_t code_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> code(code_size + 1);
    CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
    fclose(fp);

    const char *source = code.data(), *pos, *tail;
    gm.hMod = NVRTCCompile(source);
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
  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke",
        &invoke, 
        "Generic Invoke (CUDA)"
    );
}

