# Author: Jan Klhufek (iklhufek@fit.vut.cz)
# This code has been inspired by the code provided within the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import os
import argparse
from argparse import RawTextHelpFormatter
import torch
import torchvision.models as models

from typing import Tuple, List

# Get list of PyTorch model names
model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))

def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for creating PyTorch model and parsing its layers into parsed shapes format.

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="create_model_pytorch",
                                     description="Creator of PyTorch models into Timeloop layer description format")
    # Configuration
    parser.add_argument("-m", "--model", type=str, metavar="MODEL", default="alexnet", choices=model_names,
                        help="model choices: " + " | ".join(model_names) + " (default: alexnet)")
    parser.add_argument("-i", "--input_size", type=str, default="224,224,3",
                        help="input size in format W,H,C")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-c", "--num_classes", type=int, default=1000,
                        help="number of classes for classification")
    # Saving
    parser.add_argument("-o", "--outfile", type=str, default=f"created_model_layers",
                        help="output file name")
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

        elif isinstance(m, torch.nn.Linear):
            M = m.out_features
            layer = (1, 1, C, N, M, 1, 1, 0, 0, 1, 1)
            layers.append(layer)
            C = M

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


def create_pytorch_model(input_size: Tuple[int, int, int], model_name: str, batch_size: int, out_dir: str, out_file: str, num_classes: int = 1000, verbose: bool = False) -> None:
    """
    Creates a PyTorch model and writes its layer dimensions into
    a description for a YAML file used by Timeloop.

    Args:
        input_size (Tuple[int, int, int]): Input size as (W, H, C).
        model_name (str): Name of the model.
        batch_size (int): Batch size.
        out_dir (str): Output directory for the YAML file.
        out_file (str): Output file name.
        num_classes (int): Number of classes for the classification task. Defaults to 1000.
        verbose (bool): Enables verbose output. Defaults to False.
    """
    assert model_name in model_names, f"Provided model named '{model_name}' is not in the list of supported models: " + " | ".join(model_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[model_name](num_classes=num_classes).to(device)

    cnn_layers = get_layers_pytorch(model, input_size, batch_size)

    if verbose:
        print("# Model: " + str(model_name))
        print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
        print("cnn_layers = [")
        for layer in cnn_layers:
            print("    " + str(layer) + ",")
        print("]")

    with open(os.path.join(out_dir, out_file + ".yaml"), "w") as f:
        f.write(f"api: pytorch\n")
        f.write(f"model: {model_name}\n")
        f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
        f.write("layers:\n")
        for layer in cnn_layers:
            f.write("  - [")
            f.write(", ".join(str(p) for p in layer))
            f.write("]\n")


def main() -> None:
    args = parse_args()
    # Process parsed arguments
    input_size = tuple((int(d) for d in str.split(args.input_size, ",")))

    if args.verbose:
        print("Begin processing")
        print("API name: PyTorch")
        print("Model name: " + str(args.model))
        print("Input size: " + str(input_size))

    out_dir = "parsed_models"
    os.makedirs("parsed_models", exist_ok=True)
    # Process PyTorch model and return layer dimensions
    create_pytorch_model(input_size, args.model, args.batch_size, out_dir, args.outfile, args.num_classes, args.verbose)


if __name__ == "__main__":
    main()
