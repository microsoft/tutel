# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
from torch.distributed import run

if __name__ == "__main__":

    sys.argv = [sys.argv[0],
            '--nproc_per_node=%s' % os.environ.get('LOCAL_SIZE', 1),
            '--nnodes=%s' % os.environ.get('OMPI_COMM_WORLD_SIZE', 1),
            '--node_rank=%s' % os.environ.get('OMPI_COMM_WORLD_RANK', 0)
    ] + sys.argv[1:]

    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'

    run.main()
