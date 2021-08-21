#!/bin/bash -e

exec python3 -m torch.distributed.launch --nproc_per_node=${NGPU:-2} ${@:-test_tutel.py}
