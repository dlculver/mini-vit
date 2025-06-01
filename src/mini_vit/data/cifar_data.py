"""Module for loading CIFAR datasets for training and testing."""

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

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
        raise ValueError("split must be either 'train' or 'test'")

    # Ensure there is a data directory
    download = False
    if not os.path.exists("./cifar_data"):
        download = True

    dataset = datasets.CIFAR10(
        root="./cifar_data",
        train=(split == "train"),
        download=download,
        transform=transforms,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
