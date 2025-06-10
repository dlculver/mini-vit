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
            raise ValueError(
                "Image dimensions must be divisible by patch size")

        self.patch_size = patch_shape
        self.projection_dim = projection_dim
        self.num_patches = (image_height // patch_height) * \
            (image_width // patch_width)

        patch_dim = in_channels * patch_height * patch_width
        normalized_shape = [self.num_patches, patch_dim]

        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.RMSNorm(normalized_shape=normalized_shape, eps=1e-4),
            nn.Linear(patch_dim, projection_dim),
            nn.RMSNorm(normalized_shape=[
                       self.num_patches, projection_dim], eps=1e-4),
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


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, dropout: float, num_heads: int, qkv_bias: bool
    ):
        super().__init__()
        assert d_out % num_heads == 0, (
            f"d_out must be divisible by num_heads: {d_out} % {num_heads} != 0"
        )
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        # linear layers
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_o = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        b, num_patches, d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        # reshape: bs, num_patches, d_out -> b, num_patches, num_heads, head_dim
        keys = keys.view(b, num_patches, self.num_heads, self.head_dim)
        queries = queries.view(b, num_patches, self.num_heads, self.head_dim)
        values = values.view(b, num_patches, self.num_heads, self.head_dim)

        # transpose: b, num_patches, num_heads, head_dim -> b, num_heads, num_patches, head_dim
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_weights = torch.softmax(
            attn_scores / torch.sqrt(keys.shape[-1]), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(
            1, 2
        )  # b, num_patches, num_heads, head_dim

        # put your heads together
        context_vector = context_vector.contiguous().view(b, num_patches, self.d_out)

        return self.W_o(context_vector)
