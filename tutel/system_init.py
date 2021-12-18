# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import logging

def init_affinity_at_program_beginning():
    try:
        numa_type = int(os.environ.get('NUMA_TYPE', '1'))
        if numa_type <= 0:
            return
        group_rank = int(os.environ.get('LOCAL_RANK', '0'))
        nodes = sorted([int(x[4:]) for x in os.listdir('/sys/devices/system/node') if re.match('node[0-9]+', x)])
        cpus = [sorted([int(x[3:]) for x in os.listdir('/sys/devices/system/node/node%d' % node_id) if re.match('cpu[0-9]+', x)]) for node_id in nodes]
        sel_node = (group_rank // numa_type) % len(nodes)
        os.sched_setaffinity(0, cpus[sel_node])
        logging.info('LOCAL_RANK %d is to set NUMA node: %d (total NUMA nodes = %d)' % (group_rank, sel_node, len(nodes)))
    except Exception as ex:
        if group_rank == 0:
            logging.warning('Failed to set NUMA status: %s' % ex)

def init_data_model_parallel(group_count=1, backend='nccl'):
  import torch.distributed as dist
  try:
    if dist.is_available():
      dist.init_process_group(backend=backend)
    glob_world_size, glob_world_rank = dist.get_world_size(), dist.get_rank()
    is_distributed = True

    def dist_print(*args):
        if glob_world_rank == 0:
            print(*args)
  except:
    glob_world_size, glob_world_rank = 1, 0
    is_distributed = False
    dist_print = print

  assert glob_world_size % group_count == 0, f"Expected to evenly divide devices into {group_count} groups, while the world size of current sesion is {glob_world_size}."

  dist_group_size = group_count
  dist_world_size = glob_world_size // dist_group_size
  dist_world_rank = glob_world_rank % dist_world_size
  dist_group_rank = glob_world_rank // dist_world_size
  dist_local_rank = int(os.environ.get('LOCAL_RANK', 0))

  if is_distributed:
    groups, inner_ranks = [], []
    for gr in range(dist_group_size):
      group_ranks = [x for x in range(gr * dist_world_size, (gr + 1) * dist_world_size)]
      groups += [dist.new_group(ranks=group_ranks)]
      inner_ranks += [group_ranks]
    model_group = groups[dist_group_rank]

    groups, outer_ranks = [], []
    for gr in range(dist_world_size):
      group_ranks = [x for x in range(gr, dist_world_size * dist_group_size, dist_world_size)]
      groups += [dist.new_group(ranks=group_ranks)]
      outer_ranks += [group_ranks]

    data_group = groups[dist_world_rank]
    global_group = dist.new_group(ranks=list(range(dist_world_size * dist_group_size)))
  else:
    model_group, data_group, global_group = None, None, None

  result = init_data_model_parallel
  result.global_size = glob_world_size
  result.global_rank = glob_world_rank
  result.group_count = dist_group_size
  result.data_rank = dist_group_rank
  result.model_rank = dist_world_rank
  result.local_rank = dist_local_rank

  result.data_group = data_group
  result.model_group = model_group
  result.global_group = global_group

  result.is_distributed = is_distributed
  result.dist_print = dist_print

  logging.info(f'Registering device global rank {result.global_rank}: data_rank = {result.data_rank}, model_rank = {result.model_rank}')
  return result
