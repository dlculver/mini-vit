"""API entry point for data related functionalities."""

import argparse
import logging
from rich.logging import RichHandler


from mini_vit.data import get_cifar10_dataloaders

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
    parser = argparse.ArgumentParser(description="CIFAR-10 Data Loader API")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add in subparsers
    add_cifar_subparser(subparsers)

    return parser


def add_cifar_subparser(subparsers) -> None:
    """Add CIFAR-10 specific subparser."""
    cifar_parser = subparsers.add_parser(
        "cifar", help="CIFAR-10 dataset related operations"
    )

    cifar_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to load (default: train)",
    )

    cifar_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the dataloader (default: 64)",
    )

    cifar_parser.add_argument(
        "--shuffle", action="store_true", help="Whether to shuffle the dataset"
    )

    cifar_parser.set_defaults(func=run_cifar_loader)


def run_cifar_loader(args):
    """Run CIFAR-10 dataloader based on parsed arguments."""
    dataloader = get_cifar10_dataloaders(
        split=args.split, batch_size=args.batch_size, shuffle=args.shuffle
    )
    logger.info(
        f"CIFAR-10 {args.split} DataLoader created with {len(dataloader)} batches."
    )
    return dataloader


def main():
    """Main function to run the data API."""
    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)
