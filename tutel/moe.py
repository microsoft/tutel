# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
High-level interface available for users:

"""

from .jit_kernels.gating import fast_cumsum_sub_one
from .impls.fast_dispatch import fast_dispatcher
from .impls.moe_layer import moe_layer

