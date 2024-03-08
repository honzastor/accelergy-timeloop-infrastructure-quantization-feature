# Author: Jan Klhufek (iklhufek@fit.vut.cz)
# This code has been inspired by the code provided within the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple

# Get list of PyTorch model names
model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))

def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for loading PyTorch model and parsing its layers into parsed shapes format.

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="parse_model",
                                     description="Parser of PyTorch models into Timeloop layer description format")
    # Configuration
    parser.add_argument('-m', '--model_file', type=str, required=True,
                        help='relative path to model file')
    parser.add_argument("-a", "--arch", metavar="ARCH", default="", choices=model_names,
                        help="model architecture for model instantiation when loading just a state_dict: " + " | ".join(model_names))
    parser.add_argument('-i', '--input_size', type=str, default="224,224,3",
                        help='input size in format W,H,C')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size')
    # Saving
    parser.add_argument('-o', '--outfile', type=str, default=f"parsed_model_layers",
                        help='output file name')
    # Miscs
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose output")
    return parser.parse_args()


def get_output_size(W: int, H: int, kernel_size: int, stride: int, padding: int) -> Tuple[int, int]:
    """
    Computes the output shape for a convolutional layer of a model.

    Args:
        W (int): Input width.
        H (int): Input height.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.

    Returns:
        Tuple[int, int]: The computed output width and height.
    """
    W_out = int((W - kernel_size + 2 * padding) / stride) + 1
    H_out = int((H - kernel_size + 2 * padding) / stride) + 1
    # dilation = 1
    # W_out = int((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    # H_out = int((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return W_out, H_out


def get_layers_pytorch(model: torch.nn.Module, input_size: Tuple[int, int, int], batch_size: int) -> List[Tuple[int, ...]]:
    """
    Extracts the dimensions of convolutional, linear, and pooling layers from a PyTorch model summary.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_size (Tuple[int, int, int]): The input size as (W, H, C).
        batch_size (int): The batch size.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing the dimensions of each convolutional layer.
    """
    layers = []
    W, H, C = input_size
    N = batch_size

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            M = m.out_channels
            S = m.kernel_size[0]
            R = m.kernel_size[1]
            Wpad = m.padding[0]
            Hpad = m.padding[1]
            Wstride = m.stride[0]
            Hstride = m.stride[1]

            layer = (W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer)

            # print(f"in_size: {W}x{H}")
            W, H = get_output_size(W, H, S, Wstride, Wpad)
            # print(f"out_size: {W}x{H}")
            C = M

        # IGNORE FC
        """
        elif isinstance(m, torch.nn.Linear):
            M = m.out_features
            layer = (1, 1, C, N, M, 1, 1, 0, 0, 1, 1)
            layers.append(layer)
            C = M
        """

        # IGNORE POOLING
        """
        elif isinstance(m, torch.nn.MaxPool2d) or isinstance(m, torch.nn.AvgPool2d):
            if isinstance(m.kernel_size, int):
                S = R = m.kernel_size
            else:
                S, R = m.kernel_size
            if isinstance(m.stride, int):
                Wstride = Hstride = m.stride
            else:
                Wstride, Hstride = m.stride
            if isinstance(m.padding, int):
                Wpad = Hpad = m.padding
            else:
                Wpad, Hpad = m.padding

            W, H = get_output_size(W, H, S, Wstride, Wpad)
            layer = (W, H, C, N, C, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer)
        """
    return layers


def parse_pytorch_model(input_size: str, model_file: str, batch_size: int, out_dir: str, out_file: str, architecture: str = "", verbose: bool = False) -> None:
    """
    Parses a PyTorch model and writes its layer dimensions into a YAML file for Timeloop processing.

    Args:
        input_size (str): Input size as a string in format "W,H,C".
        model_file (str): Path to the PyTorch model file.
        batch_size (int): Batch size.
        out_dir (str): Output directory for the YAML file.
        out_file (str): Output file name.
        architecture (str): Specify the model architecture to instantiate if loading just the state_dict. Defaults to "".
        verbose (bool): Enables verbose output. Defaults to False.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model file `{model_file}` found.")

    # Ensure the model file is a PyTorch model
    model_parts = model_file.split("/")[-1].split(".")
    if len(model_parts) > 2:
        model_ext = ".".join(model_parts[-2:])
    else:
        model_ext = model_parts[-1]
    assert model_ext in ["pth", "pt", "pth.tar", "pt.tar"], "Unrecognized model file extension. Expected .pt, .pth, .pt.tar or .pth.tar for PyTorch model."

    input_size = tuple((int(d) for d in str.split(input_size, ",")))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that the model contains state_dict
    loaded_obj = torch.load(model_file)

    # Check if loaded_obj is a model or a state_dict
    if isinstance(loaded_obj, dict):  # Assume it's a state_dict, load it into the model
        assert architecture in model_names, "Model architecture must be provided to load state_dict!"
        chkpt_state_dict = loaded_obj['state_dict'] if 'state_dict' in loaded_obj else loaded_obj
        qat_model = any('quant' in key for key in chkpt_state_dict.keys())
        fc_tensor_id_from_last = 8 if qat_model else 2
        chkpt_fc_classes = (list(chkpt_state_dict.values())[-fc_tensor_id_from_last]).shape[0]
        model = models.__dict__[architecture]().to(device)

        # Adjust the number of output connections (classes) for the last (classifier) layer
        last_module = list(model.children())[-1]
        if isinstance(last_module, nn.Linear):
            last_module.out_features = chkpt_fc_classes
        elif isinstance(last_module, nn.Sequential):
            last_submodule = list(last_module.children())[-1]
            if isinstance(last_submodule, nn.Linear):
                last_submodule.out_features = chkpt_fc_classes
    else:
        raise TypeError("The loaded object is not a state_dict.")

    cnn_layers = get_layers_pytorch(model, input_size, batch_size)

    if verbose:
        print("# Model: " + str(model_file.split(".")[0]))
        print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
        print("cnn_layers = [")
        for layer in cnn_layers:
            print("    " + str(layer) + ",")
        print("]")

    with open(os.path.join(out_dir, out_file + ".yaml"), "w") as f:
        f.write(f"api: pytorch\n")
        f.write(f"model: " + architecture + "\n")
        f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
        f.write("layers:\n")
        for layer in cnn_layers:
            f.write("  - [")
            f.write(", ".join(str(p) for p in layer))
            f.write("]\n")


def main() -> None:
    args = parse_args()

    if args.verbose:
        print('Begin processing')
        print('API name: pytorch')
        print('Model name: ' + str(args.model))
        print('Input size: ' + str(args.input_size))

    out_dir = "parsed_models"
    os.makedirs("parsed_models", exist_ok=True)
    # Process PyTorch model and return layer dimensions
    parse_pytorch_model(args.input_size, args.model_file, args.batch_size, out_dir, args.outfile, args.arch, args.verbose)


if __name__ == "__main__":
    main()
