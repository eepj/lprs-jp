# Import torch and torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

# Import additional libraries
import argparse
import mplfonts
import numpy as np
import os
import random
import sys
from typing import Sequence

from models.cnn import LP_CharacterRecognitionCNN
from utils import *

# Set Noto Sans CJK JP font used to render plots
mplfonts.use_font("Noto Sans JP")

# Define and set random states
RANDOM_STATE_DEFAULT = 42

# Define default hyperparameters
EPOCH_DEFAULT = 100
LR_DEFAULT = 1e-3


def main():
    # Parse the input arguments
    parser = argparse.ArgumentParser()

    # Path to the dataset image direcotry and label csv
    parser.add_argument("--dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--export", required=True)

    # Define model architecture
    parser.add_argument("--layers", nargs="+", type=int, required=True)

    # Define image size
    parser.add_argument("-x", required=True, type=int)
    parser.add_argument('-y', required=True, type=int)

    # Training hyperparameters
    parser.add_argument("--epoch", type=int, default=EPOCH_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)

    # Hardware acceleratin device
    parser.add_argument("--device", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])

    # Augmentation pipeline options
    parser.add_argument("--zoom", action="store_true", default=False)
    g = parser.add_argument_group('Zoom options')
    # Add the zoom-specific arguments to the group
    g.add_argument("--zoom_min", type=float, required="--zoom" in sys.argv)
    g.add_argument("--zoom_max", type=float, required="--zoom" in sys.argv)
    g.add_argument("--zoom_ratio", type=float, required="--zoom" in sys.argv)

    parser.add_argument("--rotation", type=float, default=False)

    # Flag to visualize training history and evalutaion results, default `False`
    parser.add_argument("--visualize", action="store_true", default=False)

    # Random seed
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)

    # Parse arguments
    args = parser.parse_args()

    # Set the random seed
    random_state = args.seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    # If --zoom is set and any required args are missing, raise error
    if args.zoom and None in [args.zoom_min, args.zoom_max, args.zoom_ratio]:
        parser.error("Augmentation zoom parameters not set")

    # Automatically select hardware acceleration device based on availability
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() \
            else "cpu"
    else:
        device = args.device

    # Get paths
    dir_path = os.path.expanduser(args.dir)
    csv_path = os.path.expanduser(args.csv)

    # Perform a 60-20-20 stratified data split on the dataset
    datasets, maps = stratify_dataset(csv_path, dir_path, random_state)
    train_set, val_set, test_set = datasets
    _, i2s = maps

    image_shape = (args.y, args.x)

    # Determine the augmentation pipeline to use
    pipeline = [
        transforms.ColorJitter((0.8, 1.2), (0.8, 1.2),
                               (0.8, 1.2), (-0.3, 0.3)),
        transforms.RandomChannelPermutation(),
        transforms.RandomInvert()
    ]

    # Random resized crop
    if args.zoom:
        pipeline = pipeline + [
            transforms.RandomResizedCrop(image_shape, (args.zoom_min, args.zoom_max),
                                         ratio=(1.0, args.zoom_ratio), antialias=True)
        ]

    # Rotation
    if args.rotation:
        pipeline = pipeline + [
            transforms.RandomRotation(args.rotation)
        ]

    # Perspective and noise
    pipeline = pipeline + [
        transforms.RandomPerspective(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x + torch.normal(0, 1, x.size())),
    ]

    print(f"Augmentation pipeline:")
    for p in pipeline:
        print("-", p)

    # Check if --layer is a sequence
    if not isinstance(args.layers, Sequence):
        parser.error("Sequence of int expected for --layers argument")

    # Define the model
    model = LP_CharacterRecognitionCNN(3, len(i2s), image_shape,
                                       args.layers, device,
                                       transforms.Compose(pipeline))
    # Get dataloaders
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)

    # Train the model using the provided hyperparameters
    model.train_loop(train_loader, val_loader, args.epoch, args.lr)

    # Evalulate the model on the test set
    result = evaluate_model(model, test_set)

    # Export the model
    export_path = os.path.abspath(args.export)
    torch.save(model.state_dict(), export_path)
    print(f"Model exported as {export_path}")

    # Visualize the result
    if args.visualize:
        visualize_history(model)
        visualize_result(result, i2s, "Misclassified Instances")


if __name__ == "__main__":
    main()
