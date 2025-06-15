"""Module for testing model components"""

import pytest
import torch

from mini_vit.model.components import (
    PatchEmbedding,
    MultiHeadAttention,
    TransformerBlock,
    TransformerEncoder,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "batch_size",
        "in_channels",
        "patch_shape",
        "image_shape",
        "projection_dim",
        "expected_num_patches",
    ),
    [
        (1, 3, (16, 16), (32, 32), 768, 4),
        (1, 3, (16, 16), (64, 64), 768, 16),
        (1, 3, (16, 16), (128, 128), 768, 64),
    ],
)
def test_patch_embedding(
    batch_size,
    in_channels,
    patch_shape,
    image_shape,
    projection_dim,
    expected_num_patches,
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

    assert output.shape == (batch_size, expected_num_patches, 768)


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "batch_size, num_patches, embed_dim, num_heads, ff_dim, dropout, qkv_bias, expected_shape",
    [
        (1, 4, 64, 8, 128, 0.1, False, torch.Size([1, 4, 64])),
        (4, 8, 128, 8, 256, 0.1, False, torch.Size([4, 8, 128])),
        (16, 8, 256, 8, 512, 0.1, False, torch.Size([16, 8, 256])),
    ],
)
def test_transformer_block_forward(
    batch_size,
    num_patches,
    embed_dim,
    num_heads,
    ff_dim,
    dropout,
    qkv_bias,
    expected_shape,
):
    """Test forward pass of TransformerBlock."""
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        qkv_bias=qkv_bias,
    )
    x = torch.randn(batch_size, num_patches, embed_dim)
    output = block(x)

    assert (
        output.shape == expected_shape
    )  # should be batch_size, num_patches, embed_dim
    # Output should be different from input
    assert not torch.allclose(output, x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "batch_size, num_patches, embed_dim, num_heads, ff_dim, dropout, qkv_bias, num_layers, expected_shape",
    [
        (1, 4, 64, 4, 128, 0.1, False, 6, torch.Size([1, 4, 64])),
        (4, 8, 128, 4, 256, 0.1, False, 12, torch.Size([4, 8, 128])),
        (8, 16, 256, 4, 512, 0.1, False, 24, torch.Size([8, 16, 256])),
    ],
)
def test_transformer_encoder_forward(
    batch_size,
    num_patches,
    embed_dim,
    num_heads,
    ff_dim,
    dropout,
    qkv_bias,
    num_layers,
    expected_shape,
):
    """Test forward pass of TransformerEncoder."""
    encoder = TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        qkv_bias=qkv_bias,
        num_layers=num_layers,
    )

    x = torch.randn(batch_size, num_patches, embed_dim)
    output = encoder(x)

    assert output.shape == expected_shape
    assert not torch.allclose(output, x)


# @pytest.mark.unit
# @pytest.mark.parametrize(
#     "n_layers, dim_embed, num_heads, qkv_bias, mlp_factor, dropout, batch_size, num_patches",
#     [
#         (2, 768, 8, True, 4, 0.1, 1, 16),
#         (4, 512, 4, False, 4, 0.2, 2, 32),
#         (6, 256, 2, True, 2, 0.1, 4, 64),
#     ],
# )
# def test_transformer_encoder(
#     n_layers,
#     dim_embed,
#     num_heads,
#     qkv_bias,
#     mlp_factor,
#     dropout,
#     batch_size,
#     num_patches,
# ):
#     """Test forward pass of TransformerEncoder."""
#     encoder = TransformerEncoder(
#         n_layers=n_layers,
#         dim_embed=dim_embed,
#         num_heads=num_heads,
#         qkv_bias=qkv_bias,
#         mlp_factor=mlp_factor,
#         dropout=dropout,
#     )
#
#     x = torch.randn(batch_size, num_patches, dim_embed)
#     output = encoder(x)
#
#     assert output.shape == x.shape
#     # Output should be different from input
#     assert not torch.allclose(output, x)
#
#
# @pytest.mark.unit
# @pytest.mark.parametrize(
#     "image_shape, patch_shape, channels, n_layers, dim_embed, num_heads, num_classes, qkv_bias, batch_size",
#     [
#         ((32, 32), (16, 16), 3, 2, 768, 8, 10, True, 1),
#         ((64, 64), (16, 16), 3, 4, 512, 4, 10, False, 2),
#         ((128, 128), (16, 16), 3, 6, 256, 2, 10, True, 4),
#     ],
# )
# def test_vision_transformer(
#     image_shape,
#     patch_shape,
#     channels,
#     n_layers,
#     dim_embed,
#     num_heads,
#     num_classes,
#     qkv_bias,
#     batch_size,
# ):
#     """Test forward pass of VisionTransformer."""
#     model = VisionTransformer(
#         image_shape=image_shape,
#         patch_shape=patch_shape,
#         channels=channels,
#         n_layers=n_layers,
#         dim_embed=dim_embed,
#         num_heads=num_heads,
#         num_classes=num_classes,
#         qkv_bias=qkv_bias,
#     )
#
#     x = torch.randn(batch_size, channels, image_shape[0], image_shape[1])
#     output = model(x)
#
#     assert output.shape == (batch_size, num_classes)
#     assert torch.is_floating_point(output)
#     # Check if output is proper logits (not normalized)
#     assert not torch.allclose(output.sum(dim=1), torch.ones(batch_size))
