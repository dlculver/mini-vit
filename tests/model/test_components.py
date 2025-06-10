"""Module for testing model components"""

import pytest
import torch

from mini_vit.model.components import PatchEmbedding, MultiHeadAttention


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "batch_size",
        "in_channels",
        "patch_shape",
        "image_shape",
        "projection_dim",
    ),
    [
        (1, 3, (16, 16), (32, 32), 768),
        (1, 3, (16, 16), (64, 64), 768),
        (1, 3, (16, 16), (128, 128), 768),
    ],
)
def test_patch_embedding(
    batch_size, in_channels, patch_shape, image_shape, projection_dim
):
    # Initialize PatchEmbedding class
    patch_emb = PatchEmbedding(
        in_channels=in_channels,
        patch_shape=patch_shape,
        image_shape=image_shape,
        projection_dim=projection_dim,
    )

    # random test input
    test_input = torch.randn(batch_size, in_channels, image_shape[0], image_shape[1])

    output = patch_emb(test_input)

    assert output.shape == (batch_size, patch_emb.num_patches, 768)


@pytest.mark.unit
@pytest.mark.parametrize(
    "d_in, d_out, dropout, num_heads, qkv_bias",
    [
        (768, 768, 0.1, 8, True),
        (512, 512, 0.2, 4, False),
        (256, 256, 0.3, 2, True),
    ],
)
def test_multihead_attn_initialize(d_in, d_out, dropout, num_heads, qkv_bias):
    """Test initialization of MultiHeadAttention class."""
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
    )

    assert mha.d_out == d_out
    assert mha.num_heads == num_heads
    assert mha.qkv_bias == qkv_bias
    assert mha.W_q is not None
    assert mha.W_k is not None
    assert mha.W_v is not None
    assert mha.W_o is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "batch_size, num_patches, d_in",
    [
        (1, 16, 768),
        (2, 32, 512),
        (3, 64, 256),
    ],
)
def test_multihead_attn_forward(batch_size, num_patches, d_in):
    """Test forward pass of MultiHeadAttention class."""
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=768,
        dropout=0.1,
        num_heads=8,
        qkv_bias=True,
    )

    # random test input
    test_input = torch.randn(batch_size, num_patches, d_in)

    output = mha(test_input)

    assert output.shape == (batch_size, num_patches, 768)
