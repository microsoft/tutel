#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

exec python3 -m torch.distributed.launch --nproc_per_node=${NGPU:-2} ${FILE:-test_tutel.py} "$@"
