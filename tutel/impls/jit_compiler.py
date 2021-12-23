# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, tempfile

assert torch.cuda.is_available() == True, "This version of Tutel MoE only supports CUDA. More backends will be supported soon."

try:
    import tutel_custom_kernel
except:
    raise Exception("Cannot import JIT optimized kernels. Did you forget to install Custom Kernel Extension?")

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

class JitCompiler:
    @staticmethod
    def create_raw(source):
        if not hasattr(JitCompiler, '__CTX__'):
            torch.cuda.init()
            JitCompiler.__CTX__ = 0
            JitCompiler.__JITTED_SET__ = set()

        __ctx__ = JitCompiler.__CTX__
        JitCompiler.__CTX__ += 1

        use_nvrtc = 1 if int(os.environ.get('USE_NVRTC', '0')) else 0
        if not IS_HIP_EXTENSION:
            source = '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n' + source
        else:
            source = '#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n' + source

        def func(*inputs):
            if __ctx__ not in JitCompiler.__JITTED_SET__:
                JitCompiler.__JITTED_SET__.add(__ctx__)
                tutel_custom_kernel.invoke_with_source(inputs, __ctx__, use_nvrtc, source)
            else:
                tutel_custom_kernel.invoke(inputs, __ctx__)
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template):
      for key in keyword_dict:
        template = template.replace('@%s@' % key, str(keyword_dict[key]))
      return JitCompiler.create_raw(template)
