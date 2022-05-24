# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result

def gshard_loss(scores_w_noise, top_ids):
    num_samples, num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    mask = _one_hot_with_dtype(top_ids[:, 0], num_global_experts, dtype=scores_w_noise.dtype,
        hot_value=num_global_experts / num_samples)
    me = torch.sum(scores_w_noise, dim=0)
    ce = torch.sum(mask, dim=0)
    l_aux = torch.sum(me * ce) / num_samples
    return l_aux
