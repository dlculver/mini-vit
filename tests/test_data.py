"""Pytests for the data module."""

import pytest
from mini_vit.data.cifar_data import get_cifar10_dataloaders


@pytest.fixture
def cifar10_train_loader():
    """Fixture for CIFAR10 train dataloader."""
    return get_cifar10_dataloaders()


def test_cifar_dataloader(cifar10_train_loader):
    """Tests for the CIFAR dataloader."""
    dl = get_cifar10_dataloaders(split="train", batch_size=32)
    for images, labels in dl:
        assert images.size(0) == 32
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Tensor type: {type(images)}")
        print(images)
        break  # Just show the first batch for demonstration
