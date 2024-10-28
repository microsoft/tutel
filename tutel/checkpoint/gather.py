# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import torch
import re
import warnings

from tutel.system import apply_rank_size_from_pattern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--namespace', type=str, default='')
    parser.add_argument('--default_num_global_experts', type=int, default=0)
    args = parser.parse_args()
    args.size = args.input_size

    mutate_size, expert_dict = {}, {}

    input_file = apply_rank_size_from_pattern(args.inputs, rank=0, size=args.size)
    state_dict = torch.load(input_file, map_location=torch.device('cpu'))
    for package in args.namespace.split('/'):
        if not package:
            continue
        state_dict = state_dict[package]
    for k in state_dict:
        if k.endswith('._num_global_experts'):
             entry = k[:k.rindex('.')]
             mutate_size[entry + '.experts.'] = int(state_dict[k])

    missing_keys = []
    if not mutate_size:
        if args.default_num_global_experts <= 0:
            raise Exception('Failed to detect Tutel MoE layer in the checkpoint,\n\tas the provided checkpoint may be in legacy format with field `_num_global_experts` missing.\nPlease try again by manually providing the designed number of total experts using: --default_num_global_experts=?')
        else:
            for k in state_dict:
                if '.experts.' in k:
                    entry = k[:k.rindex('.experts.')]
                    mutate_size[entry + '.experts.'] = args.default_num_global_experts
                    missing_keys += [entry]

    for rank in range(args.size):
        input_file = apply_rank_size_from_pattern(args.inputs, rank=rank, size=args.size)
        state_dict_ = state_dict = torch.load(input_file, map_location=torch.device('cpu'))
        for package in args.namespace.split('/'):
            if not package:
                continue
            state_dict = state_dict[package]
        for k in missing_keys:
            state_dict[k + '._num_global_experts'] = args.default_num_global_experts
        for k in state_dict:
            for e in mutate_size:
                if k.startswith(e):
                    expert_dict[k] = expert_dict.get(k, [mutate_size[e],]) + [state_dict[k],]

    expert_dict = [(i, k, expert_dict[k]) for i, k in enumerate(expert_dict)]
    for i, k, v in expert_dict:
        num_global_experts, pieces = v[0], v[1:]
        if num_global_experts % args.size == 0:
            expert_dict[i] = torch.concat(pieces, dim=0).contiguous().clone()
            assert expert_dict[i].size(0) == num_global_experts, "Unexpected group size of expert with num_global_experts: %d v.s. %d. Maybe you set a wrong --size value." % (expert_dict[i].size(0), num_global_experts)
        elif args.size % num_global_experts == 0:
            expert_dict[i] = torch.concat(pieces, dim=0).contiguous()
            expert_dict[i] = expert_dict[i].view([num_global_experts, -1] + list(expert_dict[i].shape)[2:]).clone()
        else:
            raise Exception(f'Neither of "global_experts({num_global_experts}) / args.size({args.size})" nor "args.size({args.size}) / global_experts({num_global_experts})" is evenly divisible.')
        expert_dict[i] = (k, expert_dict[i])

    expert_dict = dict(expert_dict)
    for k in state_dict:
        if k in expert_dict:
            state_dict[k] = expert_dict[k]
    torch.save(state_dict_, args.output)
    print(f'Model params have been collected to: {args.output}')

if __name__ == "__main__":
    main()

