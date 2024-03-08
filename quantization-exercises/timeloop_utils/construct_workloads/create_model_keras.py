# Author: Jan Klhufek (iklhufek@fit.vut.cz)
# This code has been inspired by the code provided within the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import os
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
from tensorflow.keras import applications as keras_apps
from typing import Tuple, List

# Get list of Keras model names in applications
model_names = sorted(name for name in dir(keras_apps)
                     if not name.startswith("__")
                     and callable(getattr(keras_apps, name)))

def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for creating Keras model and parsing its layers into parsed shapes format.

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="create_model_keras",
                                     description="Creator of Keras models into Timeloop layer description format")
    # Configuration
    parser.add_argument("-m", "--model", type=str, metavar="MODEL", default="MobileNet", choices=model_names,
                        help="model choices: " + " | ".join(model_names) + " (default: MobileNet)")
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

        # IGNORE FC
        """
        elif isinstance(layer, tf.keras.layers.Dense):
            M = layer.units
            # For Dense layers, the spatial dimensions are considered as 1, and the input channels are the flattened features
            layer_dims = (1, 1, C, N, M, 1, 1, 0, 0, 1, 1)
            layers.append(layer_dims)
            W, H, C = 1, 1, M
        """

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


# Create Timeloop layer description from created keras model
def create_keras_model(input_size: Tuple[int, int, int], model_name: str, batch_size: int, out_dir: str, out_file: str, num_classes: int = 1000, verbose: bool = False) -> None:
    """
    Creates a Keras model and writes its layer dimensions into
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
    model_func = getattr(keras_apps, model_name, None)
    assert model_func is not None, f"Provided model name '{model_name}' is not in the list of supported models: " + " | ".join(model_names)
    model = model_func(include_top=True, weights=None, input_shape=input_size, classes=num_classes)
    layers = get_layers_keras(model, input_size, batch_size)

    if verbose:
        print("# Model: " + str(model_name))
        print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
        print("cnn_layers = [")
        for layer in layers:
            print("    " + str(layer) + ",")
        print("]")

    with open(os.path.join(out_dir, out_file + ".yaml"), "w") as f:
        f.write(f"api: keras\n")
        f.write(f"model: {model_name}\n")
        f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
        f.write("layers:\n")
        for layer in layers:
            f.write("  - [")
            f.write(", ".join(str(p) for p in layer))
            f.write("]\n")


def main() -> None:
    args = parse_args()
    # Process parsed arguments
    input_size = tuple(map(int, args.input_size.split(',')))

    if args.verbose:
        print("Begin processing")
        print("API name: Keras")
        print("Model name: " + str(args.model))
        print("Input size: " + str(input_size))

    out_dir = "parsed_models"
    os.makedirs(out_dir, exist_ok=True)
    # Process Keras model and return layer dimensions
    create_keras_model(input_size, args.model, args.batch_size, out_dir, args.outfile, args.num_classes, args.verbose)

if __name__ == "__main__":
    main()
