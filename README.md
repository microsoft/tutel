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

* Install Tutel Online:

        $ python3 -m pip uninstall tutel -y
        $ python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main

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

How to import Tutel-optimized MoE in Pytorch:
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

Usage of MOELayer:
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

For Deepspeed MoE Acceleration (Deepspeed MoE Top-1 Gate has integrated Tutel acceleration):
```sh
# Without Tutel optimization:
python3 -m tutel.examples.helloworld_deepspeed --top=1

# With Tutel optimization:
python3 -m tutel.examples.helloworld_deepspeed --top=1 --use_tutel
```


### Single-GPU Throughput (batches/sec) with default settings on NVIDIA A100 (40GB):
| batch-size | helloworld (top2) | helloworld_ddp (top2) | helloworld_deepspeed (top2) |
| :--------: | :--------: | :------------: | :------------------: |
| 8  | 672.75 | 672.24 | 188.27 |
| 16 | 715.86 | 714.95 | 115.43 |
| 24 | 725.95 | 725.04 | 81.02 |
| 32 | 729.02 | 729.02 | OOM |
| 64 | 687.92 | 686.31 | OOM |
| 128 | 619.75 | 619.03 | OOM |
| 256 | 577.08 | 577.49 | OOM |

How to reproduce these results:
```shell
$ python3 -m tutel.examples.helloworld --batch_size=<batch_size>
$ python3 -m tutel.examples.helloworld_ddp --batch_size=<batch_size>
$ python3 -m tutel.examples.helloworld_deepspeed --batch_size=<batch_size>
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
