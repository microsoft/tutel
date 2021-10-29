# Tutel

Tutel MoE: An Optimized Mixture-of-Experts Implementation.

- Supported Framework: Pytorch
- Supported GPUs: CUDA(fp32 + fp16), ROCm(fp32 + fp16)

How to setup Tutel MoE for Pytorch:
```
* Install Online:

        $ python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x

* Build from Source:

        $ git clone https://github.com/microsoft/tutel --branch v0.1.x

        $ python3 ./tutel/setup.py install --user
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
    }
)

# Cast to GPU
moe_layer = moe_layer.to('cuda:0')

# Forward MoE:
y = moe_layer(x)

print(y)
```

Full Examples & Usage:
```
* Single-GPU Test:

        $ python3 -m tutel.examples.helloworld

        (If full source code exists:)
        $ python3 ./tutel/examples/helloworld.py

* Running MoE Hello World Model by torch.distributed.all_reduce:

        $ python3 -m torch.distributed.launch --nproc_per_node=2 -m tutel.examples.helloworld

        (For New Pytorch:)
        $ python3 -m torch.distributed.run --nproc_per_node=2 -m tutel.examples.helloworld

* Running MoE Hello World Model by torch.nn.parallel.DistributedDataParallel (requires torch >= 1.8.0):

        $ python3 -m torch.distributed.launch --nproc_per_node=2 -m tutel.examples.helloworld_ddp

        (For New Pytorch:)
        $ python3 -m torch.distributed.run --nproc_per_node=2 -m tutel.examples.helloworld_ddp

* Usage of MOELayer Args:

        gate_type        : dict-type gate description, e.g. {'type': 'top', 'k': 2, ..}
        model_dim        : the number of channels for MOE's input tensor
        experts          : a dict-type config for builtin expert network, or a torch.nn.Module-type custom expert network
        scan_expert_func : allow users to specify a lambda function to iterate each experts param, e.g. `scan_expert_func = lambda name, param: setattr(param, 'expert', True)`
        result_func      : allow users to specify a lambda function to format the MoE output and aux_loss, e.g. `result_func = lambda output: (output, output.l_aux)`
        group            : specify the explicit communication group of all_to_all
        seeds            : a tuple containing a tripple of int to specify manual seed of (shared params, local params, others params after MoE's)

* Usage of dict-type Experts Config:

        count_per_node   : the number of local experts per device (by default, the value is 1 if not specified)
        type             : available built-in experts implementation, e.g: ffn
        hidden_size_per_expert : the hidden size between two linear layers for each expert (used for type == 'ffn' only)
        activation_fn    : the custom-defined activation function between two linear layers (used for type == 'ffn' only)
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
