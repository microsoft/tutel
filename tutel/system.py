# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
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
    from tutel import net as C
    result = C.create_groups_from_world(group_count=group_count, include_init=backend)
    result.is_cuda = (result.local_device.type == 'cuda')

    logging.critical(f'Registering device global rank {result.global_rank}: data_rank = {result.data_rank}, model_rank = {result.model_rank}')
    init_data_model_parallel.default_env = result

    def on_quit():
        sys.stdout.flush()
        sys.stderr.flush()
        # Builtin dist.all_to_all_single in torch is unstable in some versions.
        # Temp work around: https://github.com/pytorch/pytorch/issues/56390
        if getattr(C.simple_all_to_all, '_use_builtins', False):
            os._exit(0)

    import atexit
    atexit.register(lambda *args: on_quit())
    return result

class LocalCache:
    _CACHE = dict()

    @staticmethod
    def reset():
        LocalCache._CACHE = dict()

    @staticmethod
    def set(key, val):
        LocalCache._CACHE[key] = val

    @staticmethod
    def get(key=None):
        if key not in LocalCache._CACHE:
            return [LocalCache._CACHE[x] for x in LocalCache._CACHE]
        return LocalCache._CACHE[key]

def cache():
    return LocalCache

def get_local_session():
    if not hasattr(init_data_model_parallel, 'default_env'):
        raise Exception("Current session is not initialized with: system.init_data_model_parallel() from tutel. Please try with: system.record_time(is_cuda=False)")
    return init_data_model_parallel.default_env

def record_time(is_cuda=None):
    import time
    is_cuda = is_cuda if is_cuda is not None else get_local_session().is_cuda
    if is_cuda:
        import torch
        torch.cuda.synchronize()
    return time.time()

def save(t, path):
    import numpy as np
    npv = t.detach().cpu().numpy()
    np.save(path, npv)

def load(path, device=None):
    import numpy as np
    import torch
    npv = np.load(path)
    return torch.tensor(npv, device=device)

def apply_rank_size_from_pattern(filename, rank, size, create_dir=True):
    if not re.search(r'\{rank\}', filename):
        logging.warning('Keyword `{rank}` is not found in file pattern: %s, which may cause collision in file access.' % filename)

    filename = re.sub(r'\{rank\}', str(rank), re.sub(r'\{size\}', str(size), filename))
    if create_dir:
        filedir = os.path.dirname(filename)
        if filedir:
            try:
                os.makedirs(filedir)
            except FileExistsError:
                pass
    return filename
