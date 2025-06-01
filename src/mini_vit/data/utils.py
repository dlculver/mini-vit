"""Utility functions for data processing and augmentation."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_cifar(tensor_list: list[torch.Tensor], ncols=4):
    """
    Visualize a list of CIFAR-10 images.

    Args:
    ----
        tensor_list (list[torch.Tensor]): List of tensors representing images.
        ncols (int): Number of columns in the grid.
    """
    nrows = int(np.ceil(len(tensor_list) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))
    for i, tensor in enumerate(tensor_list):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(tensor.permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.show()
