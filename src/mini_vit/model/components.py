"""Components used in the ViT architecture."""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import Tuple


class PatchEmbedding(nn.Module):
    """Transform input images into patch embeddings.

    This module splits an image into fixed-size patches, flattens them, and projects
    them into a specified embedding dimension. Each patch is processed independently.

    Attributes:
        patch_size: Tuple[int, int], the height and width of each patch
        num_patches: int, total number of patches for the input image
        projection_dim: int, the output dimension for each patch embedding
    """

    def __init__(
        self,
        in_channels: int,
        patch_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
        projection_dim: int,
    ):
        super().__init__()
        patch_height, patch_width = patch_shape
        image_height, image_width = image_shape

        if not all(im % p == 0 for im, p in zip(image_shape, patch_shape)):
            raise ValueError("Image dimensions must be divisible by patch size")

        self.patch_size = patch_shape
        self.projection_dim = projection_dim
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = in_channels * patch_height * patch_width
        normalized_shape = (self.num_patches, patch_dim)

        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.RMSNorm(normalized_shape=normalized_shape, eps=1e-4),
            nn.Linear(patch_dim, projection_dim),
            nn.RMSNorm(normalized_shape=(self.num_patches, projection_dim), eps=1e-4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch embedding.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, num_patches, projection_dim)
        """
        return self.layers(x)

    @property
    def output_size(self) -> Tuple[int, int]:
        """Returns the output size as (num_patches, projection_dim)."""
        return (self.num_patches, self.projection_dim)
