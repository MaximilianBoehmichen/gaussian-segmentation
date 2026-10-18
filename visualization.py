from pathlib import Path

import numpy as np
import torchvision.utils as vutils
import torch
from matplotlib import pyplot as plt

from knn import StructuredGridKNN

import torchvision.utils as vutils
import nibabel as nib
from pathlib import Path
from typing import Tuple, Any


def save_visualization(gs,
                       img_tensor: torch.Tensor,
                       knn_index: StructuredGridKNN,
                       epoch: int,
                       run_dir: Path,
                       device: torch.device) -> None:
    """
    Evaluates the middle Z-slice of the volume, normalizes it (clamping max to 20k),
    and saves a side-by-side comparison (Ground Truth, Prediction, Error).

    Args:
        gs: GaussianIntensityField. The model to evaluate.
        img_tensor: torch.Tensor. The raw ground truth volume data.
                    Expected shape: (C, Z, Y, X).
        knn_index: StructuredGridKNN. The spatial index for querying neighbors.
        epoch: int. The current training epoch.
        run_dir: Path. The directory where the PNG should be saved.
        device: torch.device. The compute device.
    """
    gs.eval()
    with torch.no_grad():
        C, Z, Y, X = img_tensor.shape
        z_mid = Z // 2

        # 1. Create coordinates for the middle Z-slice
        y_coords = torch.arange(Y, device=device)
        x_coords = torch.arange(X, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        zz = torch.full_like(yy, z_mid)

        # Shape: (Y * X, 3)
        slice_coords = torch.stack([zz, yy, xx], dim=-1).float()
        slice_coords_flat = slice_coords.view(-1, gs.D)

        # 2. Query KNN and perform forward pass
        slice_knn = knn_index.query(slice_coords_flat.detach())
        pred_intensities_flat = gs.forward(slice_coords_flat, slice_knn)

        # 3. Reshape predictions and fetch ground truth
        # Shape: (C, Y, X)
        pred_slice = pred_intensities_flat.view(Y, X, gs.out_dim).permute(2, 0, 1)
        gt_slice = img_tensor[:, z_mid, :, :]

        # 4. Normalize based on the 20k constraint for visualization
        MAX_INTENSITY = img_tensor.amax()
        gt_vis = (gt_slice.clamp(0, MAX_INTENSITY) / MAX_INTENSITY)
        pred_vis = (pred_slice.clamp(0, MAX_INTENSITY) / MAX_INTENSITY)
        error_vis = torch.abs(gt_vis - pred_vis)

        # 5. Format channels for standard image saving (RGB max)
        def format_channels(tensor_slice: torch.Tensor) -> torch.Tensor:
            if tensor_slice.shape[0] > 3:
                return tensor_slice[:3, :, :]
            elif tensor_slice.shape[0] == 2:
                zero_channel = torch.zeros((1, Y, X), device=device)
                return torch.cat([tensor_slice, zero_channel], dim=0)
            return tensor_slice

        gt_vis = format_channels(gt_vis)
        pred_vis = format_channels(pred_vis)
        error_vis = format_channels(error_vis)

        batch_vis = torch.stack([gt_vis, pred_vis, error_vis], dim=0)

        save_path = run_dir / f"recon_epoch_{epoch:04d}.png"
        vutils.save_image(batch_vis, save_path, nrow=3, normalize=False)
    print(f'Saved visualization to: {save_path}')

    gs.train()


from typing import Tuple, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib


def save_reconstructed_volume(
        gs: nn.Module,
        knn_index: Any,
        image_shape: Tuple[int, int, int, int],
        save_path: Union[str, Path],
        device: torch.device,
        chunk_size: int = 1_000_000
) -> None:
    """
    Evaluates the full spatial grid in chunks to prevent OOM errors,
    and saves the reconstructed volume to a NIfTI file. Properly handles
    dimensionality for ITK-SNAP compatibility.

    Args:
        gs (nn.Module): The trained model.
        knn_index (Any): The spatial index structure.
        image_shape (Tuple[int, int, int, int]): Expected tensor shape of the original
            volume in the format (C, Z, Y, X).
        save_path (Union[str, Path]): Path to save the '.nii.gz' file.
        device (torch.device): Compute device for tensor allocations.
        chunk_size (int): Number of voxels to process simultaneously.
    """
    print("Exporting full reconstructed volume to NIfTI...")
    gs.eval()
    C, Z, Y, X = image_shape

    z_coords: torch.Tensor = torch.arange(Z, device=device, dtype=torch.float32)
    y_coords: torch.Tensor = torch.arange(Y, device=device, dtype=torch.float32)
    x_coords: torch.Tensor = torch.arange(X, device=device, dtype=torch.float32)

    # grid shape: (Z, Y, X, 3)
    grid: torch.Tensor = torch.stack(torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij'), dim=-1)

    # coords_flat shape: (Z * Y * X, 3)
    coords_flat: torch.Tensor = grid.view(-1, 3)

    num_voxels: int = coords_flat.shape[0]

    # out_flat shape: (num_voxels, C)
    out_flat: torch.Tensor = torch.zeros((num_voxels, C), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, num_voxels, chunk_size):
            end_idx: int = min(i + chunk_size, num_voxels)

            # chunk_coords shape: (N, 3)
            chunk_coords: torch.Tensor = coords_flat[i:end_idx]

            chunk_knn = knn_index.query(chunk_coords)

            # chunk_pred shape: (N, C)
            chunk_pred: torch.Tensor = gs.forward(chunk_coords, chunk_knn)

            out_flat[i:end_idx] = chunk_pred

    # Reshape back to spatial dimensions: (Z, Y, X, C)
    vol_reconstructed: torch.Tensor = out_flat.view(Z, Y, X, C)

    # Permuting from (Z, Y, X, C) -> (X, Y, Z, C)
    vol_nifti_format: np.ndarray = vol_reconstructed.permute(2, 1, 0, 3).contiguous().cpu().numpy()

    # ITK-SNAP dimension fix
    if C == 1:
        # (X, Y, Z, 1) causes crashes. Squeeze to strictly (X, Y, Z)
        vol_nifti_format = np.squeeze(vol_nifti_format, axis=-1)
    else:
        # For C > 1, ITK-SNAP expects multi-component images in the 5th dimension: (X, Y, Z, T, C)
        vol_nifti_format = np.expand_dims(vol_nifti_format, axis=3)  # Shape becomes (X, Y, Z, 1, C)

    # Create and save NIfTI image (identity affine is fine for raw voxel grids)
    nii_img = nib.Nifti1Image(vol_nifti_format, affine=np.eye(4))
    nib.save(nii_img, str(save_path))
    print(f"Saved reconstructed volume to {save_path}")


def save_interpolated_volume(
        gs: nn.Module,
        knn_index: Any,
        image_shape: Tuple[int, int, int, int],
        save_path: Union[str, Path],
        device: torch.device,
        scale_factors: Tuple[int, int, int] = (4, 1, 1),
        chunk_size: int = 1_000_000
) -> None:
    """
    Evaluates the spatial grid at a specified sub-voxel resolution in chunks
    to prevent OOM errors, and saves the interpolated volume to a NIfTI file.

    Args:
        gs (nn.Module): The trained model.
        knn_index (Any): The spatial index structure.
        image_shape (Tuple[int, int, int, int]): Expected tensor shape of the original
            volume in the format (C, Z, Y, X).
        save_path (Union[str, Path]): Path to save the '.nii.gz' file.
        device (torch.device): Compute device for tensor allocations.
        scale_factors (Tuple[int, int, int]): Upsampling factor for (Z, Y, X).
        chunk_size (int): Number of voxels to process simultaneously.
    """
    print(f"Exporting interpolated volume to NIfTI with scale factors (Z, Y, X) = {scale_factors}...")
    gs.eval()
    C, Z, Y, X = image_shape
    sz, sy, sx = scale_factors

    z_coords: torch.Tensor = torch.arange(0, Z, 1.0 / sz, device=device, dtype=torch.float32)
    y_coords: torch.Tensor = torch.arange(0, Y, 1.0 / sy, device=device, dtype=torch.float32)
    x_coords: torch.Tensor = torch.arange(0, X, 1.0 / sx, device=device, dtype=torch.float32)

    Z_new, Y_new, X_new = len(z_coords), len(y_coords), len(x_coords)

    # grid shape: (Z_new, Y_new, X_new, 3)
    grid: torch.Tensor = torch.stack(torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij'), dim=-1)

    # coords_flat shape: (Z_new * Y_new * X_new, 3)
    coords_flat: torch.Tensor = grid.view(-1, 3)

    num_voxels: int = coords_flat.shape[0]

    # out_flat shape: (num_voxels, C)
    out_flat: torch.Tensor = torch.zeros((num_voxels, C), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, num_voxels, chunk_size):
            end_idx: int = min(i + chunk_size, num_voxels)

            # chunk_coords shape: (N, 3)
            chunk_coords: torch.Tensor = coords_flat[i:end_idx]

            chunk_knn = knn_index.query(chunk_coords)

            # chunk_pred shape: (N, C)
            chunk_pred: torch.Tensor = gs.forward(chunk_coords, chunk_knn)

            out_flat[i:end_idx] = chunk_pred

    # Reshape back to (Z_new, Y_new, X_new, C)
    vol_reconstructed: torch.Tensor = out_flat.view(Z_new, Y_new, X_new, C)

    # Permute to (X_new, Y_new, Z_new, C)
    vol_nifti_format: np.ndarray = vol_reconstructed.permute(2, 1, 0, 3).contiguous().cpu().numpy()

    # # ITK-SNAP dimension fix
    # if C == 1:
    #     vol_nifti_format = np.squeeze(vol_nifti_format, axis=-1)
    # else:
    #     vol_nifti_format = np.expand_dims(vol_nifti_format, axis=3)  # Shape becomes (X_new, Y_new, Z_new, 1, C)

    # Update affine matrix to shrink the voxel spacing relative to the upsampling factor.
    affine: np.ndarray = np.diag([1.0 / sx, 1.0 / sy, 1.0 / sz, 1.0])
    print(vol_nifti_format.shape)

    nii_img = nib.Nifti1Image(vol_nifti_format, affine=affine)
    nib.save(nii_img, str(save_path))
    print(f"Saved interpolated volume to {save_path}. New shape: {(C, Z_new, Y_new, X_new)}")


def plot_intensity_histogram(
        image: torch.Tensor | np.ndarray,
        bins: int = 30,
        sample_size: int = 1_000_000
) -> None:
    """Plots a histogram of pixel intensities with frequency values above each bar.

    Args:
        image (torch.Tensor | np.ndarray): Input image of shape (C, H, W), (H, W, C), etc.
        bins (int): Number of histogram bins (default: 30 for readable labels).
        sample_size (int): Max number of elements to sample for memory efficiency.
    """
    if isinstance(image, torch.Tensor):
        flat = image.detach().cpu().reshape(-1)
        if flat.numel() > sample_size:
            indices = torch.randint(0, flat.numel(), (sample_size,))
            flat = flat[indices]
        data = flat.numpy()
    else:
        flat = image.ravel()
        if flat.size > sample_size:
            flat = np.random.choice(flat, size=sample_size, replace=False)
        data = flat

    fig, ax = plt.subplots(figsize=(10, 5))

    # plt.hist returns the counts, bin edges, and patches (the bars)
    counts, bin_edges, patches = ax.hist(
        data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7
    )

    # Automatically add the frequency labels on top of each bar
    ax.bar_label(patches, padding=3, rotation=45, fontsize=8, fmt='%d')

    # Give the top of the plot extra headroom so the labels aren't cut off
    ax.set_ylim(0, counts.max() * 1.15)

    ax.set_title('Pixel Intensity Histogram with Bin Frequencies')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()