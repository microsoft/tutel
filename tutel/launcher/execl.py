# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, re, sys
import logging
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default=False, action='store_true')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    local_size = int(os.environ['LOCAL_SIZE'])

    if int(os.environ.get('TUTEL_CUDA_SANDBOX', 0)) == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    else:
        os.environ['TUTEL_CUDA_SANDBOX'] = '1'

    skip_numa = int(os.environ.get('OMP_NUM_THREADS', '1')) > 1
    cmd_args = []
    try:
        if skip_numa or not os.path.exists('/usr/bin/numactl'):
            raise
        local_size = int(os.environ['LOCAL_SIZE'])
        cpu_nodes = sorted([str(x[4:]) for x in os.listdir('/sys/devices/system/node') if re.match('node[0-9]+', x)])
        if len(cpu_nodes) <= local_size:
          sel_nodes = cpu_nodes[(local_rank // (local_size // len(cpu_nodes))) % len(cpu_nodes)]
        else:
          sel_nodes = cpu_nodes[local_rank::local_size]
        sel_nodes = ','.join(sel_nodes)

        cmd_args = ['/usr/bin/numactl', '--cpunodebind=%s' % sel_nodes]
    except Exception as ex:
        if local_rank == 0:
            logging.warning('`numactl` is not enabled by tutel.launcher.execl')

    cmd_args += [sys.executable, '-m'] if args.m else []
    cmd_args += args.rest
    os.execl(cmd_args[0], *cmd_args)

if __name__ == "__main__":
    main()
