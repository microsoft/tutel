# Project Tutel

Tutel MoE: An Optimized Mixture-of-Experts Implementation.

- Supported Framework: Pytorch
- Supported GPUs: CUDA(fp32 + fp16), ROCm(fp32)

How to setup Tutel MoE for Pytorch:
```
* Install Online:

        $ python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x

* Build from Source:

        $ git clone https://github.com/microsoft/tutel
        $ python3 ./tutel/setup.py install --user
```

How to use Tutel-optimized MoE in Pytorch:
```
* Tutel MoE Example:

        moe_layer = MOELayer('Top2Gate', model_dim, experts={
            'count_per_node': 2,
            'type': 'ffn', 'hidden_size_per_expert': 1024, 'activation_fn': lambda x: F.relu(x), ..
        })
        y = moe_layer(x)

* Usage of MOELayer Args:

        gate             : the string type of MOE gate, e.g: Top1Gate, Top2Gate, Top3Gate, Top4Gate
        model_dim        : the number of channels for MOE's input tensor
        experts          : a dict-type config for builtin expert network, or a torch.nn.Module-type custom expert network
        fp32_gate        : option of enabling mixed precision for gate network
        scan_expert_func : allow users to specify a lambda function to iterate each experts param, e.g. `scan_expert_func = lambda name, param: setattr(param, 'expert', True)`
        result_func      : allow users to specify a lambda function to format the MoE output and aux_loss, e.g. `result_func = lambda output: (output, output.l_aux)`
        group            : specify the explicit communication group of all_to_all
        seeds            : a tuple containing a pair of int to specify manual seed of (shared params, local params)

* Usage of dict-type Experts Config:

        count_per_node   : the number of local experts per device (by default, the value is 1 if not specified)
        type             : available built-in experts implementation, e.g: ffn
        hidden_size_per_expert : the hidden size between two linear layers for each expert (used for type == 'ffn' only)
        activation_fn    : the custom-defined activation function between two linear layers (used for type == 'ffn' only)

* Running MoE Hello World Model by torch.distributed.all_reduce:

        $ python3 -m torch.distributed.launch --nproc_per_node=1 ./examples/helloworld.py

* Running MoE Hello World Model by torch.nn.parallel.DistributedDataParallel (requires torch >= 1.8.0):

        $ python3 -m torch.distributed.launch --nproc_per_node=1 ./examples/helloworld_ddp.py
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
