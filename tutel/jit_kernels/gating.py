# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import logging
from ..impls.jit_compiler import tutel_custom_kernel

torch.ops.load_library(tutel_custom_kernel.__file__)

use_fast_cumsum = (int(os.environ.get('FAST_CUMSUM', '1')) == 1)

def torch_cumsum_sub_one(mask1):
  locations1 = torch.cumsum(mask1, dim=0) - 1
  return locations1

def fast_cumsum_sub_one(data, dim=0):
  if data.dim() != 2 or dim != 0:
    raise Exception("Unimplemented fast_cumsum_sub_one() of data = %s and dim = %s" % (data.size(), dim))
  if not data.is_cuda or not use_fast_cumsum:
    return torch_cumsum_sub_one(data)
  return torch.ops.tutel_ops.cumsum(data)
