"""Components used in the ViT architecture."""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from typing import Tuple


class PatchEmbedding(nn.Module):
    """Transform input images into patch embeddings.

    This module splits an image into fixed-size patches, flattens them, and projects
    them into a specified embedding dimension. Each patch is processed independently.

    Attributes:
        in_channels: int, the number of channels in the input
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

        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.RMSNorm(normalized_shape=patch_dim, eps=1e-4),
            nn.Linear(patch_dim, projection_dim),
            nn.RMSNorm(normalized_shape=projection_dim, eps=1e-4),
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
        d_in (int): Input dimension of the MHA
        d_out (int): Output dimension of the MHA
        dropout (float): dropout probability
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

        attn_scores = queries @ keys.transpose(2, 3) / self.head_dim**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
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
    """Transformer block that consists of multi-head self-attention and a feed-forward network.

    This block applies layer normalization, multi-head attention, and a feed-forward network
    with residual connections.

    Attributes:
        d_in (int): Input dimension
        d_out (int): Output dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Dimension of the feed-forward network
        dropout (float): Dropout probability
        qkv_bias (bool): Whether to include bias in QKV projections
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        qkv_bias: bool,
    ):
        """Initialize the Transformer block.

        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Dimension of the feed-forward network
            dropout (float): Dropout probability
            qkv_bias (bool): Whether to include bias in QKV projections
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_in, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_out),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.RMSNorm(normalized_shape=d_in, eps=1e-4)
        self.norm2 = nn.RMSNorm(normalized_shape=d_in, eps=1e-4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, d_out)
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return x + shortcut


class TransformerEncoder(nn.Module):
    """
    Transformer based encoder which consists of multiple transformer blocks.

    This module stacks multiple transformer blocks to create a deep NN.

    Attributes:
    ----------
        d_int (int): input dimension
        d_out (int): output dimension
        num_heads (int): number of attention heads
        ff_dim (int): dimension of the hidden feed-forward network
        dropout (float): dropout probability
        qkv_bias (bool): whether to include bias in QKV projections
        num_layers (int): number of transformer blocks
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        qkv_bias: bool,
        num_layers: int,
    ):
        """Initialize the Transformer encoder.
        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Dimension of the feed-forward network
            dropout (float): Dropout probability
            qkv_bias (bool): Whether to include bias in QKV projections
            num_layers (int): Number of transformer blocks
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_in=d_in,
                    d_out=d_out,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, d_in)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, d_out)
        """
        # Pass the input through each transformer block
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """
    Transformer based mdoel for image classification.

    Based on the the paper: https://arxiv.org/pdf/2010.11929

    The model patches the input image into fixed-size patches, flattens each patch,
    and passes these through an embedding layer. The resulting embeddings are passed
    through a transformer based encoder, and the output of the encoder is passed through a
    a classification head.

    The implementation allows for two types of pooling:
    1. CLS pooling: The first token of the transformer encoder is used as the representation
       of the entire image.
    2. Mean pooling: The mean of all tokens is used as the representation of the entire image.
    In the former case, we add a learnable CLS token to the input sequence and concat it
    with the patch embeddings. In the latter case, we simply take the mean of all tokens
    in the sequence.

    Attributes:
    ----------
        in_channels (int): Number of input channels
        patch_shape (tuple[int, int]): Height and width of each patch
        image_shape (tuple[int, int]): Height and width of the input image
        embedding_dim (int): Dimension of the patch embeddings
        num_heads (int): Number of attention heads
        ff_dim (int): Dimension of the feed-forward network
        dropout (float): Dropout probability
        qkv_bias (bool): Whether to include bias in QKV projections
        num_layers (int): Number of transformer blocks
        num_classes (int): Number of output classes
        pool (str): Pooling method ('cls' or 'mean')

    """

    def __init__(
        self,
        in_channels: int,
        patch_shape: tuple[int, int],
        image_shape: tuple[int, int],
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        qkv_bias: bool,
        num_layers: int,
        num_classes: int,
        pool: str = "cls",
    ):
        super().__init__()

        patch_height, patch_width = patch_shape
        image_height, image_width = image_shape
        if not all(im % p == 0 for im, p in zip(image_shape, patch_shape)):
            raise ValueError("Image dimensions must be divisible by patch size")

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        assert pool in ["cls", "mean"], (
            f"pool must be one of ['cls', 'mean'], but got {pool}"
        )
        self.pool = pool

        # if using CLS pooling, create a CLS token "embedding"
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.num_patches += 1

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))

        self.patch_emb = PatchEmbedding(
            in_channels=in_channels,
            patch_shape=patch_shape,
            image_shape=image_shape,
            projection_dim=embedding_dim,
        )

        self.transformer = TransformerEncoder(
            d_in=embedding_dim,
            d_out=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            qkv_bias=qkv_bias,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vision Transformer.

        Pass the tensor through the patch embedding layer, transformer encoder,
        and classification head.
        The input tensor is expected to have shape (batch_size, channels, height, width).
        The output tensor will have shape (batch_size, num_classes).

        If the pooling method is 'cls', the first token of the transformer encoder
        is used as the representation of the entire image. In this case, we concatenate
        the CLS token with the patch embeddings. If the pooling method is 'mean',
        the mean of all tokens is used as the representation of the entire image.
        The CLS token is learnable and is added to the input sequence.s

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # x: b, c, h, w
        x = self.patch_emb(x)
        b, num_patches, _ = x.shape

        # repeat the CLS token for each member of the batch if using CLS pooling
        if self.pool == "cls":
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_emb
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
        x = self.mlp(x)
        return x
