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
        normalized_shape = [self.num_patches, patch_dim]

        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.RMSNorm(normalized_shape=normalized_shape, eps=1e-4),
            nn.Linear(patch_dim, projection_dim),
            nn.RMSNorm(normalized_shape=[self.num_patches, projection_dim], eps=1e-4),
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
    """Multi-head attention mechanism that allows the model to jointly attend to information
    from different representation subspaces at different positions.

    This implementation splits the input into multiple heads, computes scaled dot-product
    attention for each head independently, and then concatenates the results and projects
    them through a final linear layer.

    Attributes:
        d_out (int): Output dimension of the model
        head_dim (int): Dimension of each attention head
        num_heads (int): Number of parallel attention heads
        qkv_bias (bool): Whether to include bias in the query, key, value projections
    """

    def __init__(
        self, d_in: int, d_out: int, dropout: float, num_heads: int, qkv_bias: bool
    ):
        """Initialize the Multi-Head Attention module.

        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            dropout (float): Dropout probability
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to include bias in QKV projections
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi-head attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, d_out)
        """
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
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(
            1, 2
        )  # b, num_patches, num_heads, head_dim

        # put your heads together
        context_vector = context_vector.contiguous().view(b, num_patches, self.d_out)

        return self.W_o(context_vector)


class MLP(nn.Module):
    def __init__(self, dim_embed: int, mlp_factor: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_embed, mlp_factor * dim_embed),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_factor * dim_embed, dim_embed),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        num_heads: int,
        qkv_bias: bool,
        mlp_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO(dominic): Is this the correct normalized shape to use? Passing an int is interpreted in normalizing over the last dimension
        # This makes sense, since you don't want to mix the patches together (we assume patches are meant to be iid??)
        self.norm1 = nn.RMSNorm(dim_embed, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim_embed, eps=1e-6)

        self.mha = MultiHeadAttention(
            d_in=dim_embed,
            d_out=dim_embed,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.mlp = MLP(dim_embed=dim_embed, mlp_factor=mlp_factor)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        shotcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shotcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        return x + shortcut
