# Author: Jan Klhufek (iklhufek@fit.vut.cz)
# This code has been inspired by the code provided within the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models as keras_models
from typing import List, Tuple

# Get list of Keras model names
model_names = [m for m in dir(keras_models) if not m.startswith('_') and callable(getattr(keras_models, m))]


def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for loading Keras model and parsing its layers into parsed shapes format.

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="parse_model",
                                     description="Parser of Keras models into Timeloop layer description format")
    # Configuration
    parser.add_argument('-m', '--model_file', type=str, required=True,
                        help='relative path to model file')
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
    if isinstance(stride, tuple):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(padding, tuple):
        padding_height, padding_width = padding
    else:
        padding_height = padding_width = padding

    W_out = int((W - kernel_size + 2 * padding_width) / stride_width) + 1
    H_out = int((H - kernel_size + 2 * padding_height) / stride_height) + 1
    # dilation = 1
    # W_out = int((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    # H_out = int((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return W_out, H_out


def get_layers_keras(model: tf.keras.Model, input_size: Tuple[int, int, int], batch_size: int) -> List[Tuple[int, ...]]:
    """
    Extracts the dimensions of convolutional, linear, and pooling layers from a Keras model summary.

    Args:
        model (tf.keras.Model): The Keras model.
        input_size (Tuple[int, int, int]): The input size as (W, H, C).
        batch_size (int): The batch size.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing the dimensions of each convolutional layer.
    """
    layers = []
    W, H, C = input_size
    N = batch_size

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            M = layer.filters if not isinstance(layer, tf.keras.layers.DepthwiseConv2D) else C
            S = layer.kernel_size[0]
            R = layer.kernel_size[1]
            if layer.padding == 'same':
                Wpad, Hpad = S // 2, R // 2
            else:  # 'valid' padding
                Wpad, Hpad = 0, 0
            Wstride, Hstride = layer.strides

            W_out, H_out = get_output_size(W, H, S, Wstride, Wpad)
            layer_dims = (W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer_dims)
            W, H, C = W_out, H_out, M

        elif isinstance(layer, tf.keras.layers.Dense):
            M = layer.units
            # For Dense layers, the spatial dimensions are considered as 1, and the input channels are the flattened features
            layer_dims = (1, 1, C, N, M, 1, 1, 0, 0, 1, 1)
            layers.append(layer_dims)
            W, H, C = 1, 1, M

        # IGNORE POOLING
        """
        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.AveragePooling2D):
            pool_size = layer.pool_size[0] if isinstance(layer.pool_size, tuple) else layer.pool_size
            strides = layer.strides if layer.strides is not None else (pool_size, pool_size)
            Wstride, Hstride = strides if isinstance(strides, tuple) else (strides, strides)
            Wpad, Hpad = (0, 0)  # Typically, pooling layers do not use padding

            W, H = get_output_size(W, H, pool_size, Wstride, Wpad)
            layer_dims = (W, H, C, N, C, pool_size, pool_size, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer_dims)
        """

    return layers


def parse_keras_model(input_size: str, model_file: str, batch_size: int, out_dir: str, out_file: str, verbose: bool = False) -> None:
    """
    Parses a Keras model and writes its layer dimensions into a YAML file for Timeloop processing.

    Args:
        input_size (str): Input size as a string in format "W,H,C".
        model_file (str): Path to the Keras model file.
        batch_size (int): Batch size.
        out_dir (str): Output directory for the YAML file.
        out_file (str): Output file name.
        verbose (bool): Enables verbose output. Defaults to False.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model file `{model_file}` found.")

    # Load the Keras model
    model = load_model(model_file)

    input_size = tuple(map(int, input_size.split(',')))
    layers = get_layers_keras(model, input_size, batch_size)

    if verbose:
        print("# Model: " + str(model_file.split(".")[0]))
        print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
        print("cnn_layers = [")
        for layer in cnn_layers:
            print("    " + str(layer) + ",")
        print("]")

    with open(os.path.join(out_dir, out_file + ".yaml"), "w") as f:
        f.write("api: keras\n")
        f.write("model: " + model_file.split("/")[-1] + "\n")
        f.write("# Layer details\n")
        for layer in layers:
            f.write("  - [")
            f.write(", ".join(str(p) for p in layer))
            f.write("]\n")


def main() -> None:
    args = parse_args()

    if args.verbose:
        print('Begin processing')
        print('API name: keras')
        print('Model name: ' + str(args.model))
        print('Input size: ' + str(args.input_size))

    out_dir = "parsed_models"
    os.makedirs("parsed_models", exist_ok=True)
    # Process Keras model and return layer dimensions
    parse_keras_model(args.input_size, args.model_file, args.batch_size, out_dir, args.outfile, args.arch, args.verbose)


if __name__ == "__main__":
    main()
