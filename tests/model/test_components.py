"""Module for testing model components"""

import pytest
import torch

from mini_vit.model.components import PatchEmbedding


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
