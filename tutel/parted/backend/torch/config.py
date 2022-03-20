# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
import re
import json
import torch

def get_input_definition(name, shape, stat_dim, dtype, is_param, device=None):
  return f'E.sharded_randn({shape}, {stat_dim}, dtype=torch.{dtype}, requires_grad=True, is_param={is_param}, device={device})'

def get_execute_cmd(group_size, glob_size, device_type, program_path):
  if glob_size == 1:
    os_command = f'PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 {sys.executable} {program_path}'
  else:
    host_names = os.environ.get('HOSTS', 'localhost').split(',')
    assert glob_size % len(host_names) == 0, f"Cannot evenly launch {glob_size} instances on {len(host_names)} hosts."
    local_size = glob_size // len(host_names)
    os_command = f'mpiexec --allow-run-as-root -host {",".join(host_names)} -x MASTER_ADDR={host_names[0]} -x LOCAL_SIZE={local_size} {sys.executable} -m tutel.launcher.run {sys.executable} {program_path}'
  return os_command

def link(name, input_dim, output_dim, is_param=False, output_shape=None):
  if input_dim is None:
    return f'C.allreduce_forward({name}, group=E.parallel_env.model_group)' if output_dim == -1 else f'C.reduce_scatter({name}, {output_dim}, E.parallel_env.model_group)'
  if output_dim is None:
    return f'E.warp_bwd_allreduce({name}, {is_param})'
  if input_dim == -2:
    return f'C.zero_gather({name}, {output_shape}, E.parallel_env.model_group)'
  if input_dim == -1:
    return f'C.spatial_split({name}, {output_dim}, E.parallel_env.model_group)'
  if output_dim == -1:
    return f'C.all_gather({name}, {input_dim}, E.parallel_env.model_group)'
  return f'C.all_to_all({name}, {input_dim}, {output_dim}, E.parallel_env.model_group)'

def generate_framework_code(device_type, group_size, group_count, run_mode, compute_name, headers, input_list, param_list, graph_prog):
  headers = '\n'.join(headers).strip() + '\n' if headers else ''
  graph_prog = '\n    '.join(graph_prog)

  input_args = ', '.join([name for name, code in input_list])
  input_list = '\n    '.join([f'inputs["{name}"] = {code}' for name, code in input_list])

  for name, _ in param_list:
    graph_prog = re.sub(fr'\b{name}\b', f'self.{name}', graph_prog)

  param_list = '\n    '.join([f'self.register_parameter(name="{name}", param={code})' for name, code in param_list])

  source = f'''import torch

from tutel import net as C
from tutel.parted.backend.torch import executor as E

{headers}
class DistModel(torch.nn.Module):
  compute_name = '{compute_name}'

  def __init__(self):
    super().__init__()
    {param_list}

  def forward(self, {input_args}):
    {graph_prog}
    return {compute_name}

  @staticmethod
  def synthetic_inputs():
    inputs = dict()
    {input_list}
    return inputs


if __name__ == '__main__':
  E.init_session(group_size={group_size}, group_count={group_count}, device_type='{device_type}')
  E.model_executor(DistModel, is_training={run_mode == 'train'})
'''
  return source
