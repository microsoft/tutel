# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy, hashlib
import os, sys
import re
import json

spmd_primitives_dict = dict()

def register_primitive(name=None):
  if not name:
    name = 'custom_%d' % len(spmd_primitives_dict)
  def register_primitive_instance(func):
    assert name not in spmd_primitives_dict, f"Parallel Pattern with name `{name}` already exists."
    spmd_primitives_dict[name] = func
  return register_primitive_instance

def solve_partition(sess, compute_groups, input_nodes, split_pref, kwargs):
  marked_nodes, marked_comm = dict(), dict()
  last_node = compute_groups[-1][0][-1]

  run_mode = kwargs['run_mode']
  group_size = kwargs['spmd_nodes']
  glob_size = kwargs['total_nodes']
  print(f'\nDistributed for total_nodes = {glob_size}, spmd_nodes = {group_size}, run_mode = `{run_mode}`\n')

  FL = dict()
  for input in input_nodes:
    FL[input.name] = dict()
    if input.name in split_pref:
      dim = split_pref[input.name]
      FL[input.name][dim] = (0.0, {input.name: (dim, '')})
      continue
    FL[input.name][-1] = (0.0, {input.name: (-1, '')})
    FL[input.name][-2] = (0.0, {input.name: (-2, '')})
    for dim in range(len(input.shape)):
      if input.shape[dim] % group_size == 0:
        FL[input.name][dim] = (0.0, {input.name: (dim, '')})

  def do_merge(base_config, new_config):
    if new_config is None:
      return None
    new_config = new_config[1]
    for k in new_config:
      if k not in base_config:
        base_config[k] = new_config[k]
      elif base_config[k] != new_config[k]:
        return None
    return base_config

  for compute_nodes, multi_used in compute_groups:
    enum_nums = 1
    for node in multi_used:
      enum_nums *= len(node.shape) + 1

    final_FL = dict()

    for enum_inst in range(enum_nums):
      looping_restricted_config = dict()
      remainder = enum_inst
      for node in multi_used:
        jump_val = len(node.shape) + 1
        looping_restricted_config[node.name] = remainder % jump_val - 1
        remainder //= jump_val

      ##### Looping once
      for node in compute_nodes:
        output_name = node.name
        output_shape = node.shape
        FL[output_name] = dict()

        if group_size == 1:
          left, right = -1, 0
        elif output_name in split_pref:
          assert isinstance(split_pref[output_name], int)
          left, right = split_pref[output_name], split_pref[output_name] + 1
        else:
          left, right = -1, len(output_shape)

        for dim in range(left, right):
          if looping_restricted_config.get(output_name, dim) != dim:
            continue
          if dim >= 0 and output_shape[dim] % group_size != 0:
            continue
          programs = []
          for key in spmd_primitives_dict:
            rule_func = spmd_primitives_dict[key]
            try:
              merged_config = None
              for rank, source_dims, connectors in rule_func(sess, node, dim, group_size, None):
                merged_config = {node.name: (dim, f'{key}:{rank}')}
                for input_id in source_dims:
                  state = source_dims[input_id]
                  from_record = FL[node.inputs[input_id].name].get(state, None)
                  if from_record is not None and looping_restricted_config.get(node.inputs[input_id].name, state) == state:
                    merged_config = do_merge(merged_config, from_record)
                  else:
                    merged_config = None
                  if not merged_config:
                    break
                if merged_config:
                  break
              if not merged_config:
                continue
              prog = node.compile(merged_config, **kwargs)
              if prog:
                programs += [(prog, merged_config)]
            except NotImplementedError:
              pass

          best_result = (float('inf'), None)
          for index, (prog, cfg) in enumerate(programs):
            print(f'>> Try `{output_name}:{dim} [ENUM:{enum_inst+1}/{enum_nums}]` ({index+1}/{len(programs)}), config = {json.dumps(cfg)}')

            # Evaluate Program
            if enum_nums == 1 and (len(programs) == 1) and (output_name != last_node.name):
              model_cost = -1
            else:
              print('>> Program Snapshot:')
              print(prog.code)
              model_cost = prog.execute()
              model_cost = model_cost.get('step_time', float('inf'))

            if model_cost < best_result[0]:
              best_result = (model_cost, cfg)

          if best_result[1] is not None:
            FL[output_name][dim] = best_result
            print(f'>> FL_{output_name}_{dim} [ENUM:{enum_inst+1}/{enum_nums}] = {best_result}\n')

      # Update enum history best
      for dim in FL[compute_nodes[-1].name]:
        if dim not in final_FL or final_FL[dim][0] > FL[compute_nodes[-1].name][dim][0]:
          final_FL[dim] = FL[compute_nodes[-1].name][dim]

    # Persistent enum best
    for node in compute_nodes:
      FL[node.name] = None
    print(f'>> Updating Stage `{output_name}:{dim}`; Stage Enum = {enum_inst}/{enum_nums}; Valid FL_State[*] Count: {len(FL)}')
    sys.stdout.flush()
    FL[compute_nodes[-1].name] = final_FL

  return [(dim, FL[last_node.name].get(dim, None)) for dim in range(-1, len(last_node.shape))]

