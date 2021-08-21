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

try:
	local_rank = int(os.environ.get('LOCAL_RANK', '0'))
except:
	local_rank = 0

class JitKernel:
    @staticmethod
    def create_raw(source):
        if not hasattr(JitKernel, '__CTX__'):
            torch.cuda.init()
            JitKernel.__CTX__ = 0
        __ctx__ = JitKernel.__CTX__
        JitKernel.__CTX__ += 1

        key = local_rank
        temp_loc = '%s-%s.MoE' % (tempfile.mktemp(), __ctx__)
        with open(temp_loc, 'w') as fp:
            if IS_HIP_EXTENSION:
              fp.write('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n')
            else:
              fp.write('#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n')
            fp.write(source)
        os.rename(temp_loc, '/tmp/%s-%s.cu' % (__ctx__, key))

        def func(*inputs):
            tutel_custom_kernel.invoke(inputs, __ctx__ * 256 + key)
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template):
      for key in keyword_dict:
        template = template.replace('@%s@' % key, str(keyword_dict[key]))
      return JitKernel.create_raw(template)
