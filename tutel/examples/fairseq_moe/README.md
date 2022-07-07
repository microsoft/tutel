# Training WikiText-103 on fairseq with Tutel:
## Install Tutel
```shell
git clone https://github.com/microsoft/tutel --branch main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py
```

## Install fairseq
```shell
cd ./tutel/tutel/examples/fairseq_moe
git clone https://github.com/facebookresearch/fairseq --branch main
cd fairseq/ && git checkout b5e7b250913120409b872a940fbafec4d43c7b13
# This patch is an example to train Fairseq MoE transformers.
# Note that the current patch only works for `legacy_ddp` backend, and `--checkpoint-activations` must be disabled.
git apply ../fairseq_patch.diff
python3 -m pip install omegaconf==2.0.5 hydra-core==1.0.7
python3 -m pip install --no-deps --editable .
```

## Prepare the dataset
Download [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):
```shell
curl -LO https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip && unzip wikitext-103-v1.zip
```
Preprocess the data:
```shell
fairseq-preprocess \
    --only-source \
    --trainpref wikitext-103/wiki.train.tokens \
    --validpref wikitext-103/wiki.valid.tokens \
    --testpref wikitext-103/wiki.test.tokens \
    --destdir ./wikitext-103 \
    --workers 20

```

## Train a Model with Tutel moe (MOE is moe-freq)
```shell

# Example of Training with 8GPUs (FP32)
MOE=1 L_AUX_WT=0.01 ../run_fairseq.sh ./wikitext-103

# Example of Training with 8GPUs (FP16)
FP16=1 NO_OVERFLOW=0 MOE=1 L_AUX_WT=0.01 ../run_fairseq.sh ./wikitext-103

```
