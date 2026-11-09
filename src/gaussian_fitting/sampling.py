"""Batch sampling strategies and loss selection shared by training and hparam search."""

import math
from typing import Callable

import torch

from config.loss import SSIMLoss


def get_loss_and_sampling(loss: str) -> tuple[torch.nn.Module, Callable]:
    """Returns the loss module and the matching batch sampler for a loss name."""
    losses = {
        "mse": (torch.nn.MSELoss(), sample_voxels),
        "l1": (torch.nn.L1Loss(), sample_voxels),
        "ssim": (SSIMLoss(data_range=1.0), sample_patches),
    }

    return losses[loss]


def sample_voxels(
        spatial_shape, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, tuple]:
    """Sample random voxels.

    Args:
        spatial_shape: The shape of the image.
        batch_size: How many voxels to sample.
        device: _.

    Returns:
        The sampled voxels and their shape
    """
    indices = torch.randperm(math.prod(spatial_shape), device=device)[:batch_size]

    return indices, (-1, 1)


def sample_patches(
        spatial_shape, batch_size: int, device: torch.device, patch_size: int = 48
) -> tuple[torch.Tensor, tuple]:
    """Sample random contiguous patches.

    Args:
        spatial_shape: The shape of the image.
        batch_size: How many voxels to sample.
        device: _.
        patch_size: Cubed size of patch that is sampled.

    Returns:
        The n sampled voxels patches and their shape

    Note:
        # TODO: add some sort of probability readjustment for edges and voxels close to edges. Currently undersampled. Padding?
    """
    n = max(1, round(batch_size / patch_size ** 3))
    D, H, W = spatial_shape

    starts = torch.stack(
        [
            torch.randint(0, s - patch_size + 1, (n,), device=device)
            for s in spatial_shape
        ],
        dim=-1,
    )
    in_patch = torch.arange(patch_size, device=device)
    in_patch_d, in_patch_h, in_patch_w = torch.meshgrid(
        in_patch, in_patch, in_patch, indexing="ij"
    )

    indices = (
                      (starts[:, 0, None, None, None] + in_patch_d) * H
                      + (starts[:, 1, None, None, None] + in_patch_h)
              ) * W + (starts[:, 2, None, None, None] + in_patch_w)

    return indices.reshape(-1), (n, 1, patch_size, patch_size, patch_size)
