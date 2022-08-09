# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import torch
import re

from tutel.system import apply_rank_size_from_pattern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    args.size = args.input_size

    mutate_size, expert_dict = {}, {}

    input_file = apply_rank_size_from_pattern(args.inputs, rank=0, size=args.size)
    state_dict = torch.load(input_file, map_location=torch.device('cpu'))
    for k in state_dict:
        if k.endswith('._num_global_experts'):
             entry = k[:k.rindex('.')] + '.experts.'
             mutate_size[entry] = int(state_dict[k])

    if not mutate_size:
        raise Exception('No any Tutel MoE layer is found, as the provided checkpoint may be in legacy format. You need to reload this legacy checkpoint by corresponding application, re-checkpoint model\'s state_dict and get the latest format.')

    for rank in range(args.size):
        input_file = apply_rank_size_from_pattern(args.inputs, rank=rank, size=args.size)
        state_dict = torch.load(input_file, map_location=torch.device('cpu'))
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
    torch.save(state_dict, args.output)

if __name__ == "__main__":
    main()

