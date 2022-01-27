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
        torch.cuda.init()
        __ctx__ = tutel_custom_kernel.inject_source(source)

        def func(*inputs):
            tutel_custom_kernel.invoke(inputs, __ctx__)
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template):
      for key in keyword_dict:
        template = template.replace('@%s@' % key, str(keyword_dict[key]))
      return JitCompiler.create_raw(template)

    @staticmethod
    def generate_cpu_kernel(capacity, kernel_type):
      def func(*inputs):
        if inputs[0].dtype is torch.float32:
          tutel_custom_kernel.invoke_cpu_fp32(inputs, kernel_type, capacity)
        elif inputs[0].dtype is torch.float64:
          tutel_custom_kernel.invoke_cpu_fp64(inputs, kernel_type, capacity)
        else:
          raise Exception("CPU kernel only supports float32 and float64!")
        
      return func
