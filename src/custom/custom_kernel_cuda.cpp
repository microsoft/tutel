#include <torch/extension.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


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
    std::string file_name = "/tmp/" + std::to_string(ctx) + ".cu";
    FILE *fp = fopen(file_name.c_str(), "rb");
    CHECK_EQ(true, fp != nullptr);
    fseek(fp, 0, SEEK_END);
    size_t code_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> code(code_size + 1);
    CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
    fclose(fp);

    std::string cc = "30";
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    std::string arch = std::to_string(major) + std::to_string(minor);

    CHECK_EQ(0, system(("/usr/local/cuda/bin/nvcc " + file_name + " -o " + file_name + ".fatbin --fatbin -O2 -gencode arch=compute_" + arch  + ",code=sm_" + arch).c_str()));
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
  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke",
        &invoke, 
        "Generic Invoke (CUDA)"
    );
}

