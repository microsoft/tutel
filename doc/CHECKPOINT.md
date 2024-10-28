
### How to convert checkpoint files for different distributed world sizes:
```sh
# Firstly, using 2 GPUs to train a model with 16 global experts (each GPU holds 8 local experts), saving checkpoint files in the end:
python3 -m torch.distributed.run --nproc_per_node=2 -m tutel.examples.helloworld --num_local_experts=8 --checkpoint=./states/{rank}-of-{size}.ckpt --device=cuda

# Secondly, convert the checkpoint files (based on 2 GPUs) into a single checkpoint file containing all parameters:
python3 -m tutel.checkpoint.gather --inputs=./states/{rank}-of-{size}.ckpt --input_size=2 --output ./model-all-in-one.ckpt

# Optionally, you can test the All-in-One checkpoint using single CPU device, note that there will be 16 experts locally:
python3 -m tutel.examples.helloworld --num_local_experts=16 --checkpoint=./model-all-in-one.ckpt --device=cpu --eval

# Next, convert the All-in-One checkpoint file that adapts to distributed training using 8 GPUs:
python3 -m tutel.checkpoint.scatter --input=./model-all-in-one.ckpt --output_size=8 --outputs=./adapted-for-8-gpus/{rank}-of-{size}.ckpt

# Then, using generated checkpoint files to train/eval using 8 GPUs, note that there will be 2 local experts this time:
python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.helloworld --num_local_experts=2 --checkpoint=./adapted-for-8-gpus/{rank}-of-{size}.ckpt --device=cuda

# Similarly, the convertion tool also supports X global experts adapting to Y GPUs, where Y % X == 0, making num_local_experts to be -Y / X.
python3 -m tutel.checkpoint.scatter --input=./model-all-in-one.ckpt --output_size=32 --outputs=./adapted-for-32-gpus/{rank}-of-{size}.ckpt

ssh <node-ip-0> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --num_local_experts=-2 --checkpoint=./adapted-for-32-gpus/{rank}-of-{size}.ckpt --device=cuda
ssh <node-ip-1> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=4 --node_rank=1 --master_addr=<node-ip-0> -m tutel.examples.helloworld --num_local_experts=-2 --checkpoint=./adapted-for-32-gpus/{rank}-of-{size}.ckpt --device=cuda
ssh <node-ip-2> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=4 --node_rank=2 --master_addr=<node-ip-0> -m tutel.examples.helloworld --num_local_experts=-2 --checkpoint=./adapted-for-32-gpus/{rank}-of-{size}.ckpt --device=cuda
ssh <node-ip-3> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=4 --node_rank=3 --master_addr=<node-ip-0> -m tutel.examples.helloworld --num_local_experts=-2 --checkpoint=./adapted-for-32-gpus/{rank}-of-{size}.ckpt --device=cuda
```

### SWIN-Transmformer maintains a special checkpoint format. How to convert SWIN-Transformer checkpoint files for different distributed world sizes:

Reference Link: [Pretrained Model](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#evaluating-swin-moe)

```sh
# Download pretrained Swin-Transformer checkpoint for 32 GPUs:
curl -LO https://github.com/SwinTransformer/storage/releases/download/v2.0.2/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.zip
unzip swin_moe_small_patch4_window12_192_32expert_32gpu_22k.zip

# Make an All-in-one checkpoint for Single-GPU:
python3 -m tutel.checkpoint.gather --namespace model --default_num_global_experts=32 --inputs=./swin_moe_small_patch4_window12_192_32expert_32gpu_22k/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.pth.rank{rank} --input_size=32 --output ./swin-moe-small-all-in-one.ckpt

# Example of convertion from All-in-One checkpoint to multiple checkpoints for 2 GPUs:
python3 -m tutel.checkpoint.scatter --namespace model --input=./swin-moe-small-all-in-one.ckpt --output_size=2 --outputs=./new_swin_moe_small_for_2_gpus/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.pth.rank{rank}

# Copy the remaining parameters from SWIN-Transformer:
cp ./swin_moe_small_patch4_window12_192_32expert_32gpu_22k/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.pth.master ./new_swin_moe_small_for_2_gpus/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.pth.master
```

