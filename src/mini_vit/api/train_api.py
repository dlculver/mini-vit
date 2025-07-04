"""API for training tasks for ViT."""

# TODO(dominic): Currently, I hard code in some configs, need to figure out a better
# pattern later.
import argparse
import logging
from rich.logging import RichHandler

from torch.utils.data import DataLoader, random_split
import torch

from mini_vit.train import train
from mini_vit.model import VisionTransformer
from mini_vit.data import get_cifar10_dataset

RANDOM_SEED = 42  # for reproducibility

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

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    # Model parameters
    parser.add_argument(
        "--in-channels",
        type=int,
        default=3,
        help="Number of input channels (default: 3 for RGB images)",
    )
    parser.add_argument(
        "--patch-shape",
        type=int,
        default=4,
        help="Shape of the patches to be extracted from the input images (default: 4)",
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        default=32,
        help="Shape of the input images (default: 32)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=192,
        help="Dimension of the embedding (default: 192)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=3,
        help="Number of attention heads (default: 3)",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=768,
        help="Dimension of the feedforward layer (default: 768)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--qkv-bias",
        action="store_true",
        help="Use bias in QKV layers (default: False)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Number of transformer layers (default: 8)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes (default: 10 for CIFAR-10)",
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["cls", "mean"],
        default="cls",
        help="Pooling method to use ('cls' for class token, 'mean' for mean pooling, default: 'cls')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run the training on (default: mps if available, else cpu",
    )

    parser.set_defaults(func=run_training)
    return parser


def run_training(args):
    """Run the training process with specified arguments."""
    logger.info(
        f"Starting training for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}"
    )
    logger.info(f"Using device: {args.device}")
    logger.info(
        f"Model configuration: in_channels={args.in_channels}, patch_shape={args.patch_shape}, image_shape={args.image_shape}, embedding_dim={args.embedding_dim}, num_heads={args.num_heads}, ff_dim={args.ff_dim}, dropout={args.dropout}, qkv_bias={args.qkv_bias}, depth={args.depth}, num_classes={args.num_classes}, pool={args.pool}"
    )

    # instantiate the model
    vit = VisionTransformer(
        in_channels=args.in_channels,
        patch_shape=(args.patch_shape, args.patch_shape),
        image_shape=(args.image_shape, args.image_shape),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        qkv_bias=args.qkv_bias,
        num_layers=args.depth,
        num_classes=args.num_classes,
        pool=args.pool,
    )

    logger.info("Vision Transformer model instantiated.")

    # fetch training data, create validation set, and create dataloaders
    cifar = get_cifar10_dataset(
        split="train",
    )
    cifar_train, cifar_val = random_split(
        cifar, [45000, 5000], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(
        cifar_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        cifar_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
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
    logger.info(
        f"CIFAR-10 training and validation datasets created with sizes {len(cifar_train)} and {len(cifar_val)} respectively."
    )


def main():
    """Main function to run the training API."""
    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)
