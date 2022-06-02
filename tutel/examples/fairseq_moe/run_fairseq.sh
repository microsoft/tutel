#!/bin/bash -e

cd $(dirname $0)/fairseq

if [[ "$FP16" == "1" ]]; then
    FLAGS=${FLAGS:---fp16 --fp16-init-scale 4 --fp16-no-flatten-grads}
fi

python3 -m torch.distributed.launch --nproc_per_node=8 train.py ${@:-./wikitext-103} \
    --ddp-backend legacy_ddp \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 500000 --log-format json --log-interval 100 \
    ${FLAGS} \
    --save-dir ./fairseq_checkpoints
