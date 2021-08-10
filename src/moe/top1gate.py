# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from typing import Callable, Dict, Tuple, Any

import math
import torch
from torch import Tensor
import torch.nn.functional as F

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

class _DebugFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputTensor: Tensor) -> Tensor:
        print('forward')
        return inputTensor
    @staticmethod
    def backward(ctx: Any, outputTensor: Tensor) -> Tensor:
        print(outputTensor)
        print('shape: {}'.format(outputTensor.size()))
        return outputTensor

def top1gating(
    logits: torch.Tensor, 
    capacity_factor=1.0,
    eval_mode=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    logits = logits.to(torch.float32)
    num_tokens, num_experts = logits.shape[0], logits.shape[1]
    capacity = int(EVAL_CAPACITY_TOKEN_FRACTION * num_tokens) if eval_mode else int(capacity_factor * math.ceil(num_tokens / num_experts))

    indices1_s = torch.argmax(logits, dim=1)

    gates = F.softmax(logits, dim=1)
    # Create a mask for 1st's expert per token
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    gates1_s = (gates * mask1).sum(dim=1)
    # gates1_s = _DebugFunc.apply(gates1_s)
    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    return None, (indices1_s.to(torch.int32), capacity, locations1_s.to(torch.int32), gates1_s, num_experts)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    return l_aux, (indices1_s, capacity, locations1_s, gates1_s, num_experts)

    '''
    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    
    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1.to(gates1_s.dtype))
    # locations1_sc = num_tokens * capacity
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity)
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    dispatch_mask = combine1_sec.bool()
        
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, dict(), [indices1_s, capacity, locations1_s_, gates1_s]
    else:
        return l_aux, combine1_sec, dispatch_mask, dict(), [indices1_s, capacity, locations1_s_, gates1_s]

    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, dict()
    else:
        return l_aux, combine1_sec, dispatch_mask, dict()
    '''


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self, 
        model_dim: int, 
        num_experts: int, 
        input_noise_type=None,
        capacity_factor=1.0,
    ) -> None:
        # TODO: merge this to top2gate.py
        # 
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        return top1gating(
            logits=self.wg(input),
            capacity_factor=self.capacity_factor,
            eval_mode=not self.training,
        )
