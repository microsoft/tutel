# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from ..impls import communicate as C

def a2a_ffn_overlap_forward(input, expert_fn, a2a_ffn_overlap_degree, use_2dh, group):
    split_dim = 1
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, "Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d)." % (a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    assert input.shape[split_dim] % a2a_ffn_overlap_degree == 0, "Excepting input.shape[%d] (%d) be multiple of a2a_ffn_overlap_degree (%d)." % (split_dim, input.shape[split_dim], a2a_ffn_overlap_degree)
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)

    # Implicit x.contiguous() in CurrentStreamRelease.forward() and CurrentStreamAcquire.backward()
    if use_2dh:
        split_size = input.shape[split_dim] // a2a_ffn_overlap_degree
        input_split = input.split(split_size, dim=split_dim)
        input_scattered_after_a2a = [
            C.NcclStreamRelease.apply(
                C.AllToAll2DAsync.apply(
                    C.NcclStreamAcquire.apply(
                        C.CurrentStreamRelease.apply(
                            x,
                        i),
                    i)
                ),
            i)
            for i, x in enumerate(input_split)
        ]
    else:
        input_ready = C.CurrentStreamRelease.apply(input, 0)
        input_scattered_after_a2a = C.AllToAllScatterAsync.apply(input_ready)

    expert_output_scattered = [
        C.CurrentStreamRelease.apply(
            C.post_expert_permute(
                expert_fn(
                    C.pre_expert_permute(
                        C.CurrentStreamAcquire.apply(
                            x,
                        i),
                    group=group)
                ),
            group=group),
        i)
        for i, x in enumerate(input_scattered_after_a2a)
    ]

    if use_2dh:
        expert_output_gathered_after_a2a = [
            C.CurrentStreamAcquire.apply(
                C.NcclStreamRelease.apply(
                    C.AllToAll2DAsync.apply(
                        C.NcclStreamAcquire.apply(
                            x,
                        i)
                    ),
                i),
            i)
            for i, x in enumerate(expert_output_scattered)
        ]
        input = torch.cat(expert_output_gathered_after_a2a, dim=split_dim)
    else:
        expert_output_gathered_after_a2a = C.AllToAllGatherAsync.apply(*expert_output_scattered)
        input = C.CurrentStreamAcquire.apply(expert_output_gathered_after_a2a, 0)

    return input
