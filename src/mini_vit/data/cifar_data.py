"""Module for loading CIFAR datasets for training and testing."""

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

import logging
from rich.logging import RichHandler

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


# TODO(dominic): Do we need to add any additional transforms here?
# Should we also make this ore dynamic, e.g. fewer for experimental architectures etc.?
TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914009, 0.48215896, 0.4465308), (0.24703279, 0.24348423, 0.26158753)
        ),
    ]
)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914009, 0.48215896, 0.4465308),
            (0.24703279, 0.24348423, 0.26158753),
        ),
    ]
)


def get_cifar10_dataset(
    split: str = "train",
):
    """Download the CIFAR-10 dataset, if necessary, and return the dataset object."""
    if split not in ["train", "test"]:
        logger.error(f"The split is not 'train' or 'test': got {split}")
        raise ValueError("Split must be either 'train' or 'test'")

    # Ensure there is a data directory
    download = False
    if not os.path.exists("./cifar_data"):
        download = True
        logger.info("CIFAR-10 data directory does not exist. Downloading...")

    dataset = datasets.CIFAR10(
        root="./cifar_data",
        train=(split == "train"),
        download=download,
        transform=TRAIN_TRANSFORMS if split == "train" else TRANSFORMS,
    )
    if download:
        logger.info("Data downloaded successfully")
    else:
        logger.info("Using existing CIFAR-10 data")
    return dataset


def get_cifar10_dataloaders(
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    transforms=TRANSFORMS,
    num_workers: int = 0,
) -> DataLoader:
    """
    Get CIFAR-10 dataset and returns a dataloader.

    Args:
    ----
        split (str): Dataset split to load, either "train" or "test".
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
    -------
        DataLoader: Dataloader for the specified CIFAR-10 split.

    """
    if split not in ["train", "test"]:
        logger.error(f"The split is not 'train' or 'test': got {split}")
        raise ValueError("Split must be either 'train' or 'test'")

    # Ensure there is a data directory
    download = False
    if not os.path.exists("./cifar_data"):
        download = True
        logger.info("CIFAR-10 data directory does not exist. Downloading...")

    dataset = datasets.CIFAR10(
        root="./cifar_data",
        train=(split == "train"),
        download=download,
        transform=transforms,
    )
    if download:
        logger.info("Data downloaded successfully")
    else:
        logger.info("Using existing CIFAR-10 data")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
