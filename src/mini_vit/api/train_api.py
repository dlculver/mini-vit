"""API for training tasks for ViT."""

# TODO(dominic): Currently, I hard code in some configs, need to figure out a better
# pattern later. 
import argparse
import logging
from rich.logging import RichHandler

import torchvision.datasets as vision_datasets
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from mini_vit.train import train
from mini_vit.model import VisionTransformer
from mini_vit.data import get_cifar10_dataset

RANDOM_SEED = 42 # for reproducibility

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles formatting, so keep this simple
    datefmt="[%Y-%m-%d %H:%M:%S]",  # Custom date format
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False)  # Rich console output
    ],
)
logger = logging.getLogger(__name__)

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Training API for ViT")

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run the training on (default: mps if available, else cpu"
    )

    parser.set_defaults(func=run_training)
    return parser

def run_training(args):
    """Run the training process with specified arguments."""
    logger.info(f"Starting training for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}")

    # instantiate the model
    vit = VisionTransformer(
    in_channels=3,
    patch_shape=(8, 8),
    image_shape=(32, 32),
    embedding_dim=256,
    num_heads=8,
    ff_dim=512,
    dropout=0.1,
    qkv_bias=True,
    num_layers=8,
    num_classes=10,
    pool="cls"
)

    logger.info("Vision Transformer model instantiated.")

    # fetch training data, create validation set, and create dataloaders
    cifar = get_cifar10_dataset(
        split="train",
    )
    cifar_train, cifar_val = random_split(
        cifar,
        [45000, 5000],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(
        cifar_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True, # TODO(dominic): Is this something we need?
    )
    val_loader = DataLoader(
        cifar_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, # TODO(dominic): Is this something we need?
    )

    logger.info("Data loaders for training and validation created.")

    # start training
    train(
        model=vit,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    logger.info(f"CIFAR-10 training and validation datasets created with sizes {len(cifar_train)} and {len(cifar_val)} respectively.")

def main():
    """Main function to run the training API."""
    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)