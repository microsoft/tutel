# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, tempfile
import logging


try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION, CUDA_HOME, ROCM_HOME
except:
    IS_HIP_EXTENSION = False

try:
    import tutel_custom_kernel

    if hasattr(tutel_custom_kernel, 'update_sdk_home'):
        SDK_HOME = CUDA_HOME if not IS_HIP_EXTENSION else ROCM_HOME
        tutel_custom_kernel.update_sdk_home(torch.tensor([ord(x) for x in SDK_HOME] + [0], dtype=torch.int8, device='cpu'))
except:
    logging.warning("Cannot import JIT optimized kernels. CUDA extension will be disabled.")


class JitCompiler:
    @staticmethod
    def create_raw(source):
        torch.cuda.init()
        if not hasattr(tutel_custom_kernel, 'inject_source'):
            raise Exception('CUDA support is disabled during Tutel installation. Please configure CUDA correctly and reinstall Tutel to enable CUDA support, or report Tutel installation logs for help.')
        __ctx__ = tutel_custom_kernel.inject_source(source)

        def func(*inputs, extra=[], blocks=[]):
            tutel_custom_kernel.invoke(inputs, extra, blocks, __ctx__)
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template):
      for key in keyword_dict:
        template = template.replace('@%s@' % key, str(keyword_dict[key]))
      return JitCompiler.create_raw(template)

    @staticmethod
    def generate_cpu_kernel(kernel_type):
      def func(*inputs, extra=[]):
        if inputs[0].dtype is torch.float32:
          tutel_custom_kernel.invoke_cpu_fp32(inputs, extra, kernel_type)
        elif inputs[0].dtype is torch.float64:
          tutel_custom_kernel.invoke_cpu_fp64(inputs, extra, kernel_type)
        else:
          raise Exception("CPU kernel only supports float32 and float64!")
        
      return func

def create_cuda_kernel(source, keyword_dict={}):
  return JitCompiler.generate_kernel(keyword_dict, source)

