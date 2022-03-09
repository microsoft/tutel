# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
import json
import torch

def get_input_definition(name, shape, stat_dim, dtype, is_param, device=None):
  return f'sharded_randn({shape}, {stat_dim}, dtype=torch.{dtype}, requires_grad=True, is_param={is_param}, device={device})'

def get_execute_cmd(group_size, glob_size, device_type, file_path):
  if glob_size == 1:
    os_command = f'PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 {sys.executable} {file_path}'
  else:
    host_names = os.environ.get('HOSTS', 'localhost').split(',')
    assert glob_size % len(host_names) == 0, f"Cannot evenly launch {glob_size} instances on {len(host_names)} hosts."
    local_size = glob_size // len(host_names)
    os_command = f'mpiexec -host {",".join(host_names)} -x MASTER_ADDR={host_names[0]} -x LOCAL_SIZE={local_size} {sys.executable} -m tutel.launcher.run {sys.executable} {file_path}'
  return os_command

def generate_framework_code(device_type, group_size, group_count, run_mode, compute_name, headers, input_list, param_list, graph_prog):
  headers = '\n'.join(headers)
  graph_prog = '\n    '.join(graph_prog)

  input_args = ', '.join([name for name, code in input_list])
  input_list = '\n    '.join([f'inputs["{name}"] = {code}' for name, code in input_list])
  param_link = '\n    '.join([f'{name} = self.{name}' for name, code in param_list])
  param_list = '\n    '.join([f'self.register_parameter(name="{name}", param={code})' for name, code in param_list])

  source = f'''import os, time, sys
import json
import torch
import torch.distributed as dist
from tutel.impls import communicate as C

device_type = os.environ.get('DEVICE', '{device_type}')
verbose = int(os.environ.get('VERBOSE', '0'))
is_gpu = (device_type != 'cpu')

from tutel import system_init

parallel_env = system_init.init_data_model_parallel(group_count={group_count}, backend='nccl' if device_type == 'cuda' else 'gloo')
default_group = parallel_env.model_group

fusable_params = set()

def warp_bwd_allreduce(data, is_param):
    if is_param:
        fusable_params.add(id(data))
        return C.PrimBwdAllreduce.apply(parallel_env.global_group, data)
    return C.PrimBwdAllreduce.apply(parallel_env.model_group, data)

assert parallel_env.model_size == {group_size}, f"This codegen is designed for distributed parallelism = {group_size}, while this session activates {{parallel_env.model_size}}"

def sharded_randn(shape, dim, dtype, requires_grad=False, is_param=False, device=None):
  if device is None:
    device = parallel_env.local_device
  torch.manual_seed(1)
  complete_tensor = torch.tensor(torch.randn(shape, dtype=dtype, device='cpu').numpy(), device=device, requires_grad=requires_grad)
  if dim >= 0:
    result = torch.chunk(complete_tensor, chunks=parallel_env.model_size, dim=dim)[parallel_env.model_rank].contiguous()
  elif dim == -2:
    numel = complete_tensor.numel()
    assert numel % {group_size} == 0
    result = complete_tensor.view({group_size}, -1)[parallel_env.model_rank].contiguous()
  else:
    result = complete_tensor.contiguous()
  if is_param:
    result = torch.nn.Parameter(result * 1e-3)
    result.is_param = True
  if dim == -2:
    result._full_shape = shape
    result.is_param = True
  result.dim_state = dim
  return result

def model_executor(module):
  name = module.compute_name
  model = module().to(parallel_env.local_device)
  inputs = module.synthetic_inputs()
  output = model(**inputs)
  params = model.parameters()

  if verbose:
    sys.stderr.write('[%d] %g %g .. %g (%s)\\n' % (parallel_env.model_rank, output.flatten()[0], output.flatten()[1], output.flatten()[-1], output.shape))

  is_training = {run_mode == 'train'}

  if is_training:
    torch.manual_seed(1)
    label = torch.LongTensor(output.size(0)).random_(1).to(output.device)
    if params:
      optimizer = torch.optim.SGD(params, lr=1e-5)
    else:
      optimizer = model_executor
      optimizer.zero_grad = optimizer.step = lambda *x: None

  def next_step():
    if parallel_env.group_count > 1:
      dist.barrier()
    if device_type == 'cuda':
      torch.cuda.synchronize(parallel_env.local_device)
    t_start = time.time()

    if is_training:
      optimizer.zero_grad()
      output = model(**inputs).contiguous()
      output = torch.nn.functional.log_softmax(output.view(output.size(0), -1), dim=1)
      result = torch.nn.functional.nll_loss(output, label)
      if parallel_env.model_rank == 0 and verbose:
        sys.stderr.write(f'  Loss = {{result}} ({{output.shape}}, {{label.shape}})\\n')
      result.backward(retain_graph=True)
      if parallel_env.group_count > 1:
        for p in params:
          if id(p) not in fusable_params:
            simple_all_reduce(p.grad, parallel_env.data_group)
      optimizer.step()
    else:
      output = model(**inputs).contiguous()
      result = output.view(-1)[0]

    if parallel_env.group_count > 1:
      dist.barrier()
    if device_type == 'cuda':
      torch.cuda.synchronize(parallel_env.local_device)
    t_stop = time.time()

    step_time = t_stop - t_start
    if parallel_env.model_rank == 0 and verbose:
      sys.stderr.write('Result({run_mode}) = %g, cost = %s\\n' % (result, step_time))
    return step_time

  for i in range(5):
    next_step()
  average_step_time = sum([next_step() for _ in range(5)]) / 5
  if parallel_env.model_rank == 0:
    sys.stderr.write('  [%s] digest = %g .., time = %g\\n' % (name, output.flatten()[0], average_step_time))
    result = json.dumps({{'name': name, 'step_time': average_step_time}})
    if 'CONFIG_STORE_PATH' in os.environ:
      with open(os.environ['CONFIG_STORE_PATH'], 'w') as fp:
        fp.write(result)
    print(result)

{headers}
class DistModel(torch.nn.Module):
  compute_name = '{compute_name}'

  def __init__(self):
    super().__init__()
    {param_list}

  def forward(self, {input_args}):
    {param_link}
    {graph_prog}
    return {compute_name}

  @staticmethod
  def synthetic_inputs():
    inputs = dict()
    {input_list}
    return inputs


if __name__ == '__main__':
  model_executor(DistModel)
'''
  return source
