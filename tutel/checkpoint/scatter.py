# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import torch
import re

from tutel.system import apply_rank_size_from_pattern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_size', type=int, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument('--namespace', type=str, default='')
    args = parser.parse_args()
    args.size = args.output_size

    state_dict_ = state_dict = torch.load(args.input, map_location=torch.device('cpu'))
    mutate_size, expert_dict = {}, {}
    for package in args.namespace.split('/'):
        if not package:
            continue
        state_dict = state_dict[package]

    for k in state_dict:
        if k.endswith('._num_global_experts'):
            entry = k[:k.rindex('.')] + '.experts.'
            mutate_size[entry] = int(state_dict[k])

    if not mutate_size:
        raise Exception('No any Tutel MoE layer is found, as the provided checkpoint may be in legacy format. You need to reload this legacy checkpoint by corresponding application, re-checkpoint model\'s state_dict and get the latest format.')

    for k in state_dict:
        for e in mutate_size:
            if k.startswith(e):
                state = state_dict[k]
                shape = state.shape
                if shape[0] % args.size == 0:
                    state = state.view([args.size, shape[0] // args.size] + list(shape)[1:])
                elif args.size % shape[0] == 0:
                    divisor = args.size // shape[0]
                    for i in range(1, len(shape)):
                        if shape[i] <= 1:
                            continue
                        assert shape[i] % divisor == 0, f"The second non-squeezable dimension is to be sliced to {divisor} pieces from an parameter of shape {shape}, which isn't divisible evenly."
                    state = state.view([args.size,] + list(shape)[1:i] + [shape[i] // divisor,] + list(shape)[i+1:])
                else:
                    raise Exception(f'Neither of "global_experts({int(shape[0])}) / args.size({args.size})" nor "args.size({args.size}) / global_experts({int(shape[0])})" is evenly divisible.')
                expert_dict[k] = state

    state_bundle = {'bundle': state_dict_}
    last_dict, curr_dict, last_package = state_bundle, state_dict_, 'bundle'
    for package in args.namespace.split('/'):
        if not package:
            continue
        last_dict, last_package, curr_dict = curr_dict, package, curr_dict[package]
    last_dict[last_package] = None

    for rank in range(args.size):
        generate_dict = dict()
        for k in state_dict:
            if k not in expert_dict:
                generate_dict[k] = state_dict[k]
            else:
                generate_dict[k] = expert_dict[k][rank, :].contiguous().clone()

        output_file = apply_rank_size_from_pattern(args.outputs, rank=rank, size=args.size)
        last_dict[last_package] = generate_dict
        torch.save(state_bundle['bundle'], output_file)
        print(f'Model params have been scattered to: {output_file}')

if __name__ == "__main__":
    main()

