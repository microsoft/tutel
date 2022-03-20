# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
import time
import json
import torch
import torch.distributed as dist

from tutel import system
from tutel import net as C

def warp_bwd_allreduce(data, is_param):
    if is_param:
        fusable_params.add(id(data))
        return C.allreduce_backward(data, group=parallel_env.global_group)
    return C.allreduce_backward(data, group=parallel_env.model_group)

def sharded_randn(shape, dim, dtype, requires_grad=False, is_param=False, device=None):
  if device is None:
    device = parallel_env.local_device
  torch.manual_seed(1)
  complete_tensor = torch.tensor(torch.randn(shape, dtype=dtype, device='cpu').numpy(), device=device, requires_grad=requires_grad)
  if dim >= 0:
    result = torch.chunk(complete_tensor, chunks=parallel_env.model_size, dim=dim)[parallel_env.model_rank].contiguous()
  elif dim == -2:
    numel = complete_tensor.numel()
    assert numel % parallel_env.model_size == 0
    result = complete_tensor.view(parallel_env.model_size, -1)[parallel_env.model_rank].contiguous()
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

def init_session(group_size, group_count=1, device_type='cuda'):
  global parallel_env, fusable_params
  parallel_env = system.init_data_model_parallel(group_count=group_count, backend='nccl' if device_type == 'cuda' else 'gloo')
  fusable_params = set()
  assert parallel_env.model_size == group_size, f"This codegen is designed for distributed parallelism = {group_size}, while current session only activates {parallel_env.model_size} device.\n\nPlease retry with command: mpiexec --allow-run-as-root -host localhost -x MASTER_ADDR=localhost -x LOCAL_SIZE={group_size} {sys.executable} -m tutel.launcher.run {sys.executable} {' '.join(sys.argv)}"

def model_executor(module, is_training=True):
  name = module.compute_name
  model = module().to(parallel_env.local_device)
  inputs = module.synthetic_inputs()
  output = model(**inputs)
  params = model.parameters()

  verbose = int(os.environ.get('VERBOSE', '0'))
  is_cuda = (parallel_env.local_device.type == 'cuda')
  is_training = is_training and isinstance(output, torch.Tensor)
  start_result = output.contiguous().view(-1)[0] if isinstance(output, torch.Tensor) else -1

  if verbose:
    sys.stderr.write('[%d] %g %g .. %g (%s)\n' % (parallel_env.model_rank, output.flatten()[0], output.flatten()[1], output.flatten()[-1], output.shape))

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
    if is_cuda:
      torch.cuda.synchronize(parallel_env.local_device)
    t_start = time.time()

    if is_training:
      optimizer.zero_grad()
      result = model(**inputs).contiguous()
      result = torch.nn.functional.log_softmax(result.view(result.size(0), -1), dim=1)
      result = torch.nn.functional.nll_loss(result, label)
      if parallel_env.model_rank == 0 and verbose:
        sys.stderr.write(f'  Loss = {result} ({output.shape}, {label.shape})\n')
      result.backward(retain_graph=True)
      if parallel_env.group_count > 1:
        for p in params:
          if id(p) not in fusable_params:
            p.grad = simple_all_reduce(p.grad, group=parallel_env.data_group)
      optimizer.step()
    else:
      result = model(**inputs)
      result = result.contiguous().view(-1)[0] if isinstance(result, torch.Tensor) else -1

    if parallel_env.group_count > 1:
      dist.barrier()
    if is_cuda:
      torch.cuda.synchronize(parallel_env.local_device)
    t_stop = time.time()

    step_time = t_stop - t_start
    if parallel_env.model_rank == 0 and verbose:
      sys.stderr.write('Result(is_training=%s) = %g, cost = %s\n' % (is_training, result, step_time))
    return step_time

  for i in range(5):
    next_step()
  average_step_time = sum([next_step() for _ in range(5)]) / 5
  if parallel_env.model_rank == 0:
    sys.stderr.write('  [%s] digest = %g .., time = %g\n' % (name, start_result, average_step_time))
    result = json.dumps({'name': name, 'step_time': average_step_time})
    if 'CONFIG_STORE_PATH' in os.environ:
      with open(os.environ['CONFIG_STORE_PATH'], 'w') as fp:
        fp.write(result)
    print(result)
