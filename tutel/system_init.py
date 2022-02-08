# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import logging

TUTEL_CUDA_SANDBOX = int(os.environ.get('TUTEL_CUDA_SANDBOX', 0))

def init_affinity_at_program_beginning():
    if TUTEL_CUDA_SANDBOX:
        return
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
    from tutel.impls.communicate import create_groups_from_world
    result = create_groups_from_world(group_count=group_count, include_init=backend)

    # Temp work around for: https://github.com/pytorch/pytorch/issues/56390
    import atexit
    atexit.register(lambda *args: os._exit(0))

    logging.critical(f'Registering device global rank {result.global_rank}: data_rank = {result.data_rank}, model_rank = {result.model_rank}')
    return result
