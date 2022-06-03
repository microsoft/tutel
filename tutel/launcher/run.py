# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys

def main():
    host_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    host_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_size = int(os.environ.get('LOCAL_SIZE', 1))

    if 'TUTEL_ALLTOALL_ALGO' not in os.environ:
        if host_size >= 64 and local_size >= 8:
            os.environ['TUTEL_ALLTOALL_ALGO'] = '2D'

    master_addr = os.environ['MASTER_ADDR'] if host_size > 1 else 'localhost'
    master_port = int(os.environ.get('MASTER_PORT', 23232))

    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1024'

    try:
        from torch.distributed import run
        launch_mode = ['torch.distributed.run']
    except:
        launch_mode = ['torch.distributed.launch', '--use_env']

    cmd_args = [sys.executable, '-m'] + launch_mode + [
        '--nproc_per_node=%d' % local_size,
        '--nnodes=%d' % host_size,
        '--node_rank=%d' % host_rank,
        '--master_addr=%s' % master_addr,
        '--master_port=%s' % master_port,
        '-m', 'tutel.launcher.execl',
    ] + sys.argv[1:]
    os.execl(cmd_args[0], *cmd_args)

if __name__ == "__main__":
    main()
