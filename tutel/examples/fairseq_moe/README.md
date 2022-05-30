```
This patch is an example to make Fairseq Legacy DDP use MoE transformer simply.
The patch replaces all transformer FFN layers into Tutel MoE layers (but load-balance loss is not contributed to model loss).

# Install Tutel
git clone https://github.com/microsoft/tutel --branch main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py

# Prepare Fairseq
cd ./tutel/tutel/examples/fairseq_moe
git clone https://github.com/facebookresearch/fairseq --branch main
cd fairseq/ && git checkout b5e7b250913120409b872a940fbafec4d43c7b13

# Keep in Fairseq Root and Apply Patch for Once
git apply ../fairseq_patch.diff
python3 -m pip install --editable .

# Prepare Fairseq Dataset following: https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md

# Example of Fairseq MoE FP32 Training using 8 Local GPU
MOE=1 python3 -m torch.distributed.launch --nproc_per_node=8 ./train.py <dataset-dir> \
    --ddp-backend legacy_ddp \
    ..

# Example of Fairseq MoE FP16 Training using 8 Local GPU
MOE=1 python3 -m torch.distributed.launch --nproc_per_node=8 ./train.py <dataset-dir> \
    --ddp-backend legacy_ddp \
    --fp16 --fp16-init-scale 4 --fp16-no-flatten-grads \
    ..
```
