# Tutel

Tutel MoE: An Optimized Mixture-of-Experts Implementation.

- Supported Framework: Pytorch (recommend: >= 1.10)
- Supported GPUs: CUDA(fp64/fp32/fp16/bfp16), ROCm(fp64/fp32/fp16)
- Supported CPU: fp64/fp32


How to setup Tutel MoE for Pytorch and [run examples](tutel/examples), or [enable fairseq with MoE](tutel/examples/fairseq_moe):
```
* Recommended Pytorch (minimize version == 1.8.0):
        #   Pytorch for NVIDIA CUDA >= 10.2:
        python3 -m pip install --user torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
        #   Pytorch for NVIDIA CUDA >= 11.3:
        python3 -m pip install --user torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
        #   Pytorch for AMD ROCm == 4.2:
        python3 -m pip install --user torch==1.10.0+rocm4.2 torchvision==0.11.1+rocm4.2 -f https://download.pytorch.org/whl/torch_stable.html
        #   Pytorch for CPU:
        python3 -m pip install --user torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
        #   Windows Pytorch for NVIDIA CUDA >= 10.2:
        python3 -m pip install torch==1.10.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102

* Install Tutel Online:

        $ python3 -m pip uninstall tutel -y
        $ python3 -m pip install --verbose --user --upgrade git+https://github.com/microsoft/tutel@main

* Build Tutel from Source:

        $ git clone https://github.com/microsoft/tutel --branch main

        $ python3 -m pip uninstall tutel -y
        $ python3 ./tutel/setup.py install --user

* Quick Test on Single-GPU:

        $ python3 -m tutel.examples.helloworld --batch_size=16               # Test Tutel-optimized MoE + manual distribution
        $ python3 -m tutel.examples.helloworld_ddp --batch_size=16           # Test Tutel-optimized MoE + Pytorch DDP distribution (requires: Pytorch >= 1.8.0)
        $ python3 -m tutel.examples.helloworld_ddp_tutel --batch_size=16     # Test Tutel-optimized MoE + Tutel DDP distribution (ZeRO on optimizors)
        $ python3 -m tutel.examples.helloworld_amp --batch_size=16           # Test Tutel-optimized MoE with AMP data type + manual distribution
        $ python3 -m tutel.examples.helloworld_deepspeed --batch_size=16     # Test Deepspeed (0.5.6) MoE + manual distribution
        $ python3 -m tutel.examples.helloworld_from_scratch                  # Test Custom MoE implementation from scratch
        $ python3 -m tutel.examples.moe_mnist                                # Test MoE layer in end-to-end MNIST dataset
        $ python3 -m tutel.examples.moe_cifar10                              # Test MoE layer in end-to-end CIFAR10 dataset

        (If building from source, the following method also works:)
        $ python3 ./tutel/examples/helloworld.py --batch_size=16
        ..

* Switch Test using single-node 8 GPUs:

        $ python3 -m torch.distributed.launch --nproc_per_node=8 -m tutel.examples.helloworld_switch --batch_size=16

* Run Tutel MoE in Distributed Mode:

        (Method A - Torch launcher for `Multi-Node x Multi-GPU`:)
        $ ssh <node-ip-0> python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16
        $ ssh <node-ip-1> python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16

        (Method B - Tutel launcher for `Multi-Node x Multi-GPU`, requiring package `openmpi-bin`:)
        # << Single Node >>
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.helloworld_ddp_tutel --batch_size=16
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.moe_mnist
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.moe_cifar10
        ...

        # << Cross Nodes >>
        $ mpiexec -bind-to none -host <node-ip-0>,<node-ip-1>,.. -x MASTER_ADDR=<node-ip-0> -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.helloworld --batch_size=16

        # << For CPU-based Launch>>
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=1 -x OMP_NUM_THREADS=1024 python3 -m tutel.launcher.run -m tutel.examples.helloworld --batch_size=16 --device cpu

```

#### How to convert checkpoint files that adapt to different distributed world sizes:
```
# Firstly, using 2 GPUs to train a model with 16 global experts (each GPU holds 8 local experts), saving checkpoint files in the end:
mpiexec -bind-to none -host localhost -x LOCAL_SIZE=2 python3 -m tutel.launcher.run -m tutel.examples.helloworld --num_local_experts=8 --checkpoint=./states/{rank}-of-{size}.ckpt --device=cuda

# Secondly, convert the checkpoint files (based on 2 GPUs) into a single checkpoint file containing all parameters:
python3 -m tutel.checkpoint.gather --inputs=./states/{rank}-of-{size}.ckpt --input_size=2 --output ./model-synthetis.ckpt

# Optionally, you can test the synthetis checkpoint using single CPU device, note that there will be 16 experts locally:
python3 -m tutel.examples.helloworld --num_local_experts=16 --checkpoint=./model-synthetis.ckpt --device=cpu --eval

# Next, convert the synthetis checkpoint file that adapts to distributed training using 8 GPUs:
python3 -m tutel.checkpoint.scatter --input=./model-synthetis.ckpt --output_size=8 --outputs=./adapted-for-8-gpus/{rank}-of-{size}.ckpt

# Then, using generated checkpoint files to train/eval using 8 GPUs, note that there will be 2 local experts this time:
mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.helloworld --num_local_experts=2 --checkpoint=./adapted-for-8-gpus/{rank}-of-{size}.ckpt --device=cuda

# Similarly, the convertion tool also supports X global experts adapting to Y GPUs, where Y % X == 0, making num_local_experts to be -Y / X.
python3 -m tutel.checkpoint.scatter --input=./model-synthetis.ckpt --output_size=32 --outputs=./adapted-for-32-gpus/{rank}-of-{size}.ckpt
mpiexec -bind-to none -host localhost -x LOCAL_SIZE=32 python3 -m tutel.launcher.run -m tutel.examples.helloworld --num_local_experts=-2 --checkpoint=./adapted-for-32-gpus/{rank}-of-{size}.ckpt --device=cuda

```

#### How to import Tutel-optimized MoE in Pytorch:
```
# Input Example:
import torch
x = torch.ones([6, 1024], device='cuda:0')

# Create MoE:
from tutel import moe as tutel_moe
moe_layer = tutel_moe.moe_layer(
    gate_type={'type': 'top', 'k': 2},
    model_dim=x.shape[-1],
    experts={
        'count_per_node': 2,
        'type': 'ffn', 'hidden_size_per_expert': 2048, 'activation_fn': lambda x: torch.nn.functional.relu(x)
    },
    scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
)

# Cast to GPU
moe_layer = moe_layer.to('cuda:0')

# In distributed model, you need further skip doing allreduce on global parameters that have `skip_allreduce` mask, 
# e.g.
#    for p in moe_layer.parameters():
#        if hasattr(p, 'skip_allreduce'):
#            continue
#        dist.all_reduce(p.grad)


# Forward MoE:
y = moe_layer(x)

print(y)
```

#### Usage of MOELayer:
```
* Usage of MOELayer Args:

        gate_type        : dict-type gate description, e.g. {'type': 'top', 'k': 2, 'capacity_factor': -1.5, ..},
                              or a list of dict-type gate descriptions, e.g. [{'type': 'top', 'k', 2}, {'type': 'top', 'k', 2}],
                              the value of k in top-gating can be also negative, like -2, which indicates one GPU will hold 1/(-k) parameters of an expert
                              capacity_factor X can be positive (factor = X), zero (factor = max(needed_volumes)) or negative (factor = min(-X, max(needed_volumes))).
        model_dim        : the number of channels for MOE's input tensor
        experts          : a dict-type config for builtin expert network
        scan_expert_func : allow users to specify a lambda function to iterate each experts param, e.g. `scan_expert_func = lambda name, param: setattr(param, 'expert', True)`
        result_func      : allow users to specify a lambda function to format the MoE output and aux_loss, e.g. `result_func = lambda output: (output, output.l_aux)`
        group            : specify the explicit communication group of all_to_all
        seeds            : a tuple containing a tripple of int to specify manual seed of (shared params, local params, others params after MoE's)
        a2a_ffn_overlap_degree : the value to control a2a overlap depth, 1 by default for no overlap, 2 for overlap a2a with half gemm, ..
        parallel_type    : the parallel method to compute MoE, valid types: 'auto', 'data', 'model'
        pad_samples      : whether do auto padding on newly-coming input data to maximum data size in history

* Usage of dict-type Experts Config:

        count_per_node   : the number of local experts per device (by default, the value is 1 if not specified)
        type             : available built-in experts implementation, e.g: ffn
        hidden_size_per_expert : the hidden size between two linear layers for each expert (used for type == 'ffn' only)
        activation_fn    : the custom-defined activation function between two linear layers (used for type == 'ffn' only)
```

#### For Deepspeed MoE Acceleration (Deepspeed MoE Top-1 Gate has integrated Tutel acceleration):
```sh
# Without Tutel optimization:
python3 -m tutel.examples.helloworld_deepspeed --top=1

# With Tutel optimization:
python3 -m tutel.examples.helloworld_deepspeed --top=1 --use_tutel
```

## Reference
You can consult this [paper](https://arxiv.org/pdf/2206.03382.pdf) below to get to know more technical details about Tutel:
```
@article {tutel,
author = {Changho Hwang and Wei Cui and Yifan Xiong and Ziyue Yang and Ze Liu and Han Hu and Zilong Wang and Rafael Salas and Jithin Jose and Prabhat Ram and Joe Chau and Peng Cheng and Fan Yang and Mao Yang and Yongqiang Xiong},
title = {Tutel: Adaptive Mixture-of-Experts at Scale},
year = {2022},
month = jun,
journal = {CoRR},
volume= {abs/2206.03382},
url = {https://arxiv.org/pdf/2206.03382.pdf},
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
