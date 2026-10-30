import argparse
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

torch.set_float32_matmul_precision('high')


@dataclass
class Stage2AConfig:
    """Configuration hyperparameters for Phase 2A (Scan-Specific Backbone Fitting)."""
    sample_prop: float = 0.25                    # Proportion of voxels (0 < x <= 1) sampled per fitting step
    max_fit_steps: int = 256                    # Total pixel-fitting steps T_f (powers of 2: 2, 4, ..., 256)
    min_checkpoint_step: int = 0                # Minimum power-of-2 step at which to save a Stage 2A checkpoint
    fit_lr: float = 1e-4                        # Adam learning rate for scan-specific f_theta fitting
    seed: int = 42

    # Visualization & Full-Volume Reconstruction Debugging
    enable_visualization: bool = False          # If True, saves middle-slice PNG and NIfTI volume in checkpoint dir
    max_vis_subjects: int = 1                   # Number of subjects to export PNG/NIfTI for (-1 exports all subjects)
    eval_full_psnr_at_ckpts: bool = True        # Compute exact full-volume PSNR at power-of-2 checkpoint steps
    recon_chunk_size: int = 500_000             # Voxel chunk size for full-volume inference and PSNR calculation

    # SIREN Weight Regularization during scan fitting
    l2_reg_weight: float = 1e-6                 # Gentle L2 penalty on SIREN linear weights
    bound_excess_weight: float = 1e-3           # Penalty on weights exceeding (max_weight_bound_mult * init_bound)
    max_weight_bound_mult: float = 3.0          # Allowed multiple of theoretical SIREN initialization bound
    hard_clamp_siren_weights: bool = False      # If True, hard-clamps SIREN weights after each step

    # Reconstruction loss hyperparameters for 3D volumes
    bg_recon_weight: float = 0.1                # Down-weighting factor for background voxels
    bg_threshold: float = 1e-5                  # Intensity threshold below which voxels are treated as background

    # Data caching, storage & hardware settings
    clip_percentile: float = 99.95              # Upper percentile clipping before uint8 quantization
    pin_cached_memory: bool = True              # Pin uint8 image tensors in page-locked CPU RAM
    use_amp: bool = True                        # Use bfloat16 autocast
    save_fp16_theta: bool = False               # If True, stores per-subject theta_j^* in float16 to halve disk size
    skip_existing_checkpoints: bool = True      # Skip Stage 1 checkpoints that already have a completed Phase 2A run


@dataclass
class CachedImageUint8:
    """Container holding a single subject's full-resolution intensity volume in CPU RAM as uint8."""
    subject_id: str
    spatial_shape: Tuple[int, int, int]
    affine: np.ndarray        # Original (4, 4) NIfTI affine matrix for accurate volume export
    img_uint8: torch.Tensor   # Shape: (N_voxels, out_dim), dtype: torch.uint8, values in [0, 255]


def is_power_of_two_step(step_idx: int, min_ckpt_step: int, max_steps: int) -> bool:
    """
    Determines whether `step_idx` is a power-of-two checkpoint step (e.g., 2, 4, 8, 16, 32, 64, 128, 256)
    or the final fitting step.

    Args:
        step_idx (int): Current 1-based fitting step index.
        min_ckpt_step (int): Minimum step index before power-of-two saving begins (e.g., 2).
        max_steps (int): Maximum fitting steps configured for Phase 2A.

    Returns:
        bool: True if a `.pth` checkpoint should be saved at this step.
    """
    is_pow2 = (step_idx >= min_ckpt_step) and ((step_idx & (step_idx - 1)) == 0)
    return is_pow2 or (step_idx == max_steps)


class SineLayer(nn.Module):
    """Linear layer followed by a periodic sine activation with SIREN initialization and bound tracking."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ) -> None:
        """
        Initializes the SineLayer and records its theoretical initialization bound.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, includes a learnable bias parameter.
            is_first (bool): If True, applies first-layer SIREN uniform initialization.
            omega_0 (float): Frequency scaling factor for the sine activation.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if self.is_first:
            self.init_bound = 1.0 / self.in_features
        else:
            self.init_bound = math.sqrt(6.0 / self.in_features) / self.omega_0

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights uniformly within [-init_bound, +init_bound]."""
        with torch.no_grad():
            self.linear.weight.uniform_(-self.init_bound, self.init_bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-self.init_bound, self.init_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the scaled sine activation to the linear projection.

        Args:
            x (torch.Tensor): Input coordinate or feature tensor of shape (N, in_features).

        Returns:
            torch.Tensor: Activated feature tensor of shape (N, out_features).
        """
        return torch.sin(self.omega_0 * self.linear(x))


class MetaSegINR(nn.Module):
    """
    MetaSeg Implicit Neural Representation consisting of an L-layer SIREN reconstruction
    backbone f_theta and a shallow segmentation head g_phi.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        num_classes: int = 5,
        num_hidden_layers: int = 5,
        hidden_dim: int = 414,
        omega_0: float = 30.0
    ) -> None:
        """
        Initializes the MetaSegINR architecture.

        Args:
            in_dim (int): Spatial coordinate dimensionality (3 for 3D MRI).
            out_dim (int): Output intensity channels (1 for grayscale T1 MRI).
            num_classes (int): Number of segmentation classes C.
            num_hidden_layers (int): Number of SIREN hidden layers L.
            hidden_dim (int): Hidden layer width h.
            omega_0 (float): SIREN frequency scaling factor.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.omega_0 = omega_0

        layers: List[SineLayer] = [
            SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.feature_layers = nn.Sequential(*layers)

        self.recon_out = nn.Linear(hidden_dim, out_dim)
        self.recon_init_bound = math.sqrt(6.0 / hidden_dim) / omega_0
        with torch.no_grad():
            self.recon_out.weight.uniform_(-self.recon_init_bound, self.recon_init_bound)
            self.recon_out.bias.zero_()

        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_theta_parameters(self) -> List[nn.Parameter]:
        """Returns parameters belonging to the reconstruction network f_theta."""
        return list(self.feature_layers.parameters()) + list(self.recon_out.parameters())

    def get_phi_parameters(self) -> List[nn.Parameter]:
        """Returns parameters belonging to the segmentation head g_phi."""
        return list(self.seg_head.parameters())

    def get_theta_state_dict(self, half_precision: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns a CPU copy of the state dictionary for f_theta parameters.

        Args:
            half_precision (bool): If True, casts tensors to float16 to reduce storage size.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Nested state dictionary for 'feature_layers' and 'recon_out'.
        """
        target_dtype = torch.float16 if half_precision else torch.float32
        return {
            "feature_layers": {
                k: v.detach().cpu().to(dtype=target_dtype).clone()
                for k, v in self.feature_layers.state_dict().items()
            },
            "recon_out": {
                k: v.detach().cpu().to(dtype=target_dtype).clone()
                for k, v in self.recon_out.state_dict().items()
            },
        }

    def compute_siren_regularization(
        self,
        l2_weight: float = 1e-6,
        bound_excess_weight: float = 1e-3,
        max_bound_mult: float = 3.0
    ) -> torch.Tensor:
        """
        Computes L2 and bound-excess regularization on the SIREN backbone weights.

        Args:
            l2_weight (float): Coefficient for standard L2 weight penalty.
            bound_excess_weight (float): Coefficient for bound-excess hinge penalty.
            max_bound_mult (float): Multiplier of the theoretical SIREN bound allowed before penalty.

        Returns:
            torch.Tensor: Scalar regularization loss of shape ().
        """
        device = self.recon_out.weight.device
        reg_loss = torch.zeros((), device=device, dtype=torch.float32)

        for module in self.feature_layers:
            if isinstance(module, SineLayer):
                w = module.linear.weight
                if l2_weight > 0.0:
                    reg_loss = reg_loss + l2_weight * torch.mean(w.pow(2))
                if bound_excess_weight > 0.0:
                    limit = max_bound_mult * module.init_bound
                    excess = F.relu(w.abs() - limit)
                    reg_loss = reg_loss + bound_excess_weight * torch.mean(excess.pow(2))

        w_out = self.recon_out.weight
        if l2_weight > 0.0:
            reg_loss = reg_loss + l2_weight * torch.mean(w_out.pow(2))
        if bound_excess_weight > 0.0:
            limit_out = max_bound_mult * self.recon_init_bound
            excess_out = F.relu(w_out.abs() - limit_out)
            reg_loss = reg_loss + bound_excess_weight * torch.mean(excess_out.pow(2))

        return reg_loss

    @torch.no_grad()
    def clamp_siren_weights(self, max_bound_mult: float = 3.0) -> None:
        """
        In-place hard-clamps SIREN linear weights to [-max_bound_mult * init_bound, +max_bound_mult * init_bound].

        Args:
            max_bound_mult (float): Maximum allowed multiple of each layer's initial SIREN bound.
        """
        for module in self.feature_layers:
            if isinstance(module, SineLayer):
                limit = max_bound_mult * module.init_bound
                module.linear.weight.clamp_(-limit, limit)
        limit_out = max_bound_mult * self.recon_init_bound
        self.recon_out.weight.clamp_(-limit_out, limit_out)

    def forward_recon(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Predicts only image intensities I(x) = f_theta(x) for Phase 2A pixel fitting.

        Args:
            coords (torch.Tensor): Normalized coordinates of shape (N, in_dim) in [-1, 1].

        Returns:
            torch.Tensor: Predicted pixel intensities of shape (N, out_dim).
        """
        features = self.feature_layers(coords)
        return self.recon_out(features)


def weighted_reconstruction_loss(
    pred_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    bg_recon_weight: float = 0.1,
    bg_threshold: float = 1e-5
) -> Tuple[torch.Tensor, float]:
    """
    Computes per-voxel weighted L2 reconstruction loss (down-weighting background voxels)
    alongside unweighted MSE for PSNR reporting.

    Args:
        pred_intensity (torch.Tensor): Predicted intensities of shape (N, out_dim).
        target_intensity (torch.Tensor): Ground truth intensities of shape (N, out_dim) in [0, 1].
        bg_recon_weight (float): Multiplicative weight applied to background pixels.
        bg_threshold (float): Intensity threshold below which target pixels are treated as background.

    Returns:
        Tuple[torch.Tensor, float]:
            - weighted_mse: Differentiable scalar weighted MSE loss of shape ().
            - unweighted_mse_val: Detached float unweighted MSE across the batch.
    """
    sq_err = (pred_intensity.float() - target_intensity.float()).pow(2)
    unweighted_mse_val = float(sq_err.mean().item())
    weights = torch.where(target_intensity <= bg_threshold, bg_recon_weight, 1.0)
    weighted_mse = torch.mean(weights * sq_err)
    return weighted_mse, unweighted_mse_val


@torch.no_grad()
def reconstruct_full_volume_and_psnr(
    model: MetaSegINR,
    coords_flat: torch.Tensor,
    img_u8_gpu: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    amp_enabled: bool,
    chunk_size: int = 500_000,
    return_volume: bool = False
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Evaluates the fine-tuned INR across 100% of the volume coordinates in chunks to compute
    the exact full-volume MSE and PSNR (in dB), and optionally returns the 3D numpy volume.

    Args:
        model (MetaSegINR): Fine-tuned MetaSegINR model on GPU.
        coords_flat (torch.Tensor): Flattened coordinate grid of shape (N_voxels, D) in [-1, 1].
        img_u8_gpu (torch.Tensor): Target uint8 intensities on GPU of shape (N_voxels, out_dim).
        spatial_shape (Tuple[int, int, int]): 3D spatial shape (Z, Y, X) of the volume.
        amp_enabled (bool): Whether bfloat16 autocast is active.
        chunk_size (int): Number of coordinates evaluated per forward chunk.
        return_volume (bool): If True, materializes and returns the reconstructed 3D numpy volume.

    Returns:
        Tuple[float, float, Optional[np.ndarray]]:
            - full_mse: Exact full-volume unweighted MSE.
            - full_psnr_db: Exact full-volume PSNR in dB (assuming max intensity = 1.0).
            - recon_vol_np: Reconstructed 3D numpy array of shape (Z, Y, X) in [0, 1] if
                `return_volume` is True, else None.
    """
    model.eval()
    num_voxels = coords_flat.shape[0]
    sq_err_sum = 0.0
    vol_chunks: List[torch.Tensor] = []

    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        b_coords = coords_flat[start:end]
        b_target = img_u8_gpu[start:end].to(dtype=torch.float32) * (1.0 / 255.0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            pred = model.forward_recon(b_coords).float()

        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        sq_err_sum += F.mse_loss(pred_clamped, b_target, reduction='sum').item()

        if return_volume:
            vol_chunks.append(pred_clamped.squeeze(-1).cpu())

    model.train()
    full_mse = sq_err_sum / float(num_voxels)
    full_psnr_db = 10.0 * math.log10(1.0 / full_mse) if full_mse > 0.0 else 100.0

    recon_vol_np: Optional[np.ndarray] = None
    if return_volume:
        recon_vol_np = torch.cat(vol_chunks, dim=0).view(*spatial_shape).numpy().astype(np.float32)

    return full_mse, full_psnr_db, recon_vol_np


def save_debug_visualization_and_volume(
    recon_vol_np: np.ndarray,
    gt_img_uint8_cpu: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    affine: np.ndarray,
    log_dir: Path,
    subject_id: str,
    step_idx: int,
    full_psnr_db: float
) -> Tuple[Path, Path]:
    """
    Saves the reconstructed 3D volume as a `.nii.gz` file and the orthogonal middle slices
    (Ground Truth vs. INR Reconstruction) as a `.png` image inside the checkpoint directory.

    Args:
        recon_vol_np (np.ndarray): Reconstructed 3D volume array of shape (Z, Y, X) in [0, 1].
        gt_img_uint8_cpu (torch.Tensor): Ground truth flattened uint8 tensor of shape (N_voxels, 1) on CPU.
        spatial_shape (Tuple[int, int, int]): Volume spatial dimensions (Z, Y, X).
        affine (np.ndarray): Original (4, 4) NIfTI affine matrix.
        ckpt_dir (Path): Target checkpoint directory where `.pth`, `.png`, and `.nii.gz` are saved.
        subject_id (str): Identifier of the subject.
        step_idx (int): Current Phase 2A fitting step index.
        full_psnr_db (float): Full-volume reconstruction PSNR in dB.

    Returns:
        Tuple[Path, Path]:
            - slice_png_path: Path to the saved middle-slice PNG image.
            - nifti_path: Path to the saved NIfTI volume.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"stage2a_step_{step_idx:04d}_{subject_id}"
    slice_png_path = log_dir / f"{prefix}_mid_slice.png"
    nifti_path = log_dir / f"{prefix}_recon_vol.nii.gz"

    # 1. Save Reconstructed Volume as NIfTI (reverse transpose applied during loading)
    recon_nifti_data = np.transpose(recon_vol_np)
    nib.save(nib.Nifti1Image(recon_nifti_data, affine), str(nifti_path))

    # 2. Save Middle Slice Image (Ground Truth vs. Reconstructed across Axial/Coronal/Sagittal mid-planes)
    gt_vol_np = (gt_img_uint8_cpu.view(*spatial_shape).numpy().astype(np.float32)) * (1.0 / 255.0)
    z_mid, y_mid, x_mid = spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2

    gt_slices = [gt_vol_np[z_mid, :, :], gt_vol_np[:, y_mid, :], gt_vol_np[:, :, x_mid]]
    recon_slices = [recon_vol_np[z_mid, :, :], recon_vol_np[:, y_mid, :], recon_vol_np[:, :, x_mid]]
    plane_titles = [f"Mid-Plane 0 (idx={z_mid})", f"Mid-Plane 1 (idx={y_mid})", f"Mid-Plane 2 (idx={x_mid})"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Phase 2A Reconstruction | Subject: {subject_id} | Step: {step_idx} | Full-Vol PSNR: {full_psnr_db:.2f} dB",
        fontsize=13
    )

    for col in range(3):
        axes[0, col].imshow(gt_slices[col], cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[0, col].set_title(f"GT - {plane_titles[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(recon_slices[col], cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[1, col].set_title(f"INR Recon - {plane_titles[col]}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(slice_png_path, dpi=150)
    plt.close(fig)

    return slice_png_path, nifti_path


def resolve_training_subject_dirs(
    dataset_dir: Path,
    stage1_ckpt: Dict[str, Any],
    phase1_dir: Path
) -> List[Path]:
    """
    Resolves the exact list of training subject directories used in Stage 1 by reading
    `train_subject_ids` from the Stage 1 checkpoint or `stage1_config.yml`.

    Args:
        dataset_dir (Path): Root dataset path containing the `train` subdirectory.
        stage1_ckpt (Dict[str, Any]): Loaded Stage 1 checkpoint dictionary.
        phase1_dir (Path): Path to the `phase1` directory of the run.

    Returns:
        List[Path]: Ordered list of training subject directory paths matching Stage 1.
    """
    train_root = dataset_dir / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Training split directory does not exist: {train_root}")

    subject_ids: Optional[List[str]] = stage1_ckpt.get("train_subject_ids", None)

    if subject_ids is None:
        yml_path = phase1_dir / "stage1_config.yml"
        if yml_path.exists():
            with open(yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
                subject_ids = yml_data.get("train_subject_ids", None)

    if subject_ids is not None:
        resolved = [train_root / sid for sid in subject_ids if (train_root / sid).is_dir()]
        if len(resolved) != len(subject_ids):
            print(f"Warning: {len(subject_ids) - len(resolved)} subject IDs from Stage 1 were not found on disk.")
        return resolved

    all_valid = [
        p for p in sorted(train_root.iterdir())
        if p.is_dir() and (p / "aligned_orig.nii.gz").exists()
    ]
    s1_cfg = stage1_ckpt.get("stage1_config", {})
    max_subj = int(s1_cfg.get("max_train_subjects", -1))
    if max_subj > 0:
        all_valid = all_valid[:max_subj]
    return all_valid


def preload_images_as_uint8(
    subject_dirs: List[Path],
    clip_percentile: float = 99.95,
    pin_memory: bool = True
) -> List[CachedImageUint8]:
    """
    Loads all training subjects' intensity volumes (`aligned_orig.nii.gz`) into CPU RAM
    as flattened `torch.uint8` tensors along with their original NIfTI affine matrices.

    Args:
        subject_dirs (List[Path]): List of subject directories to preload.
        clip_percentile (float): Upper intensity percentile clipped prior to [0, 255] scaling.
        pin_memory (bool): If True and CUDA is available, pins uint8 tensors in page-locked CPU RAM.

    Returns:
        List[CachedImageUint8]: List of preloaded intensity volumes stored in CPU RAM.
    """
    cached_list: List[CachedImageUint8] = []
    total_bytes = 0
    use_pin = pin_memory and torch.cuda.is_available()

    print(f"Preloading {len(subject_dirs)} intensity volumes into CPU RAM as uint8...")
    for subj_dir in tqdm.tqdm(subject_dirs, desc="Caching images (uint8)"):
        img_path = subj_dir / "aligned_orig.nii.gz"
        nii_obj = nib.load(str(img_path))
        affine = nii_obj.affine.copy()

        img_np = np.transpose(nii_obj.get_fdata()).astype(np.float32)
        img_np = np.squeeze(img_np)
        spatial_shape: Tuple[int, int, int] = (int(img_np.shape[0]), int(img_np.shape[1]), int(img_np.shape[2]))

        min_val = float(img_np.min())
        max_val = float(np.percentile(img_np, clip_percentile)) if clip_percentile < 100.0 else float(img_np.max())
        if max_val > min_val:
            img_norm = np.clip((img_np - min_val) / (max_val - min_val), 0.0, 1.0)
        else:
            img_norm = np.zeros_like(img_np, dtype=np.float32)

        img_uint8_np = np.rint(img_norm * 255.0).astype(np.uint8)
        img_tensor = torch.from_numpy(img_uint8_np).reshape(-1, 1).contiguous()

        if use_pin:
            img_tensor = img_tensor.pin_memory()

        total_bytes += img_tensor.numel() * img_tensor.element_size()
        cached_list.append(
            CachedImageUint8(
                subject_id=subj_dir.name,
                spatial_shape=spatial_shape,
                affine=affine,
                img_uint8=img_tensor
            )
        )

    print(f"Cached {len(cached_list)} full-res images in CPU RAM ({total_bytes / (1024 ** 2):.2f} MB total).")
    return cached_list


class CoordinateGridCache:
    """Caches flattened [-1, 1]^D coordinate grids on GPU by spatial shape."""

    def __init__(self, device: torch.device) -> None:
        """
        Initializes the coordinate grid cache.

        Args:
            device (torch.device): Compute device on which coordinate grids are stored.
        """
        self.device = device
        self._cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def get_coords(self, spatial_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Retrieves or constructs a flattened coordinate grid normalized to [-1, 1]^D.

        Args:
            spatial_shape (Tuple[int, ...]): Spatial dimensions (Z, Y, X) of the volume.

        Returns:
            torch.Tensor: Flattened coordinate tensor of shape (N_voxels, D) in [-1, 1].
        """
        if spatial_shape not in self._cache:
            axes = [torch.linspace(-1.0, 1.0, steps=s, device=self.device) for s in spatial_shape]
            grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
            self._cache[spatial_shape] = grid.view(-1, len(spatial_shape)).contiguous()
        return self._cache[spatial_shape]


def fit_single_subject_trajectory(
    model: MetaSegINR,
    theta_params: List[nn.Parameter],
    init_theta_buffers: List[torch.Tensor],
    subj: CachedImageUint8,
    coords_flat: torch.Tensor,
    img_u8_gpu: torch.Tensor,
    batch_size: int,
    checkpoint_steps: List[int],
    s2a_ckpt_dir: Path,
    should_visualize_subj: bool,
    cfg: Stage2AConfig,
    amp_enabled: bool
) -> Tuple[
    Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    Dict[int, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Fits `f_theta` starting from `theta^*` on a single subject's intensity volume for
    `cfg.max_fit_steps` steps using stochastic voxel batches, recording metrics and VRAM
    at every step, computing exact full-volume PSNR at checkpoint steps, and optionally
    saving middle-slice PNGs and NIfTI volumes to `s2a_ckpt_dir`.

    Args:
        model (MetaSegINR): Active MetaSegINR model on GPU.
        theta_params (List[nn.Parameter]): List of f_theta parameters to optimize.
        init_theta_buffers (List[torch.Tensor]): Pre-allocated GPU buffers holding `theta^*`.
        subj (CachedImageUint8): Cached subject metadata and CPU uint8 tensor.
        coords_flat (torch.Tensor): Normalized coordinates of shape (N_voxels, D) in [-1, 1].
        img_u8_gpu (torch.Tensor): Target uint8 intensities on GPU of shape (N_voxels, out_dim).
        batch_size (int): Number of voxels sampled randomly per step.
        checkpoint_steps (List[int]): Sorted list of power-of-two steps at which to snapshot weights.
        s2a_ckpt_dir (Path): Directory where `.pth`, `.png`, and `.nii.gz` files are saved.
        should_visualize_subj (bool): Whether to export PNG/NIfTI files for this subject.
        cfg (Stage2AConfig): Phase 2A configuration parameters.
        amp_enabled (bool): Whether bfloat16 autocast is active.

    Returns:
        Tuple containing:
            - step_snapshots: Dict mapping `step_idx -> theta_j_state_dict` on CPU.
            - step_full_psnrs: Dict mapping `step_idx -> exact full-volume PSNR (dB)`.
            - weighted_losses: Array of shape (max_fit_steps,) with weighted MSE per step.
            - unweighted_mses: Array of shape (max_fit_steps,) with batch unweighted MSE per step.
            - reg_losses: Array of shape (max_fit_steps,) with SIREN regularization loss per step.
            - step_times: Array of shape (max_fit_steps,) with elapsed optimization time (sec) per step.
            - fwd_vrams: Array of shape (max_fit_steps,) with peak forward VRAM (MB) per step.
            - bwd_vrams: Array of shape (max_fit_steps,) with peak backward VRAM (MB) per step.
    """
    device = coords_flat.device
    is_cuda = device.type == 'cuda'
    num_voxels = coords_flat.shape[0]
    ckpt_step_set = set(checkpoint_steps)

    with torch.no_grad():
        torch._foreach_copy_([p.data for p in theta_params], init_theta_buffers)

    optimizer = torch.optim.Adam(theta_params, lr=cfg.fit_lr)
    model.train()

    T = cfg.max_fit_steps
    weighted_losses = np.zeros(T, dtype=np.float64)
    unweighted_mses = np.zeros(T, dtype=np.float64)
    reg_losses = np.zeros(T, dtype=np.float64)
    step_times = np.zeros(T, dtype=np.float64)
    fwd_vrams = np.zeros(T, dtype=np.float64)
    bwd_vrams = np.zeros(T, dtype=np.float64)

    step_snapshots: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    step_full_psnrs: Dict[int, float] = {}

    for step_idx in range(1, T + 1):
        t_step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        if batch_size < num_voxels:
            idx = torch.randint(0, num_voxels, (batch_size,), device=device)
            b_coords = coords_flat[idx]
            b_img = img_u8_gpu[idx].to(dtype=torch.float32) * (1.0 / 255.0)
        else:
            b_coords = coords_flat
            b_img = img_u8_gpu.to(dtype=torch.float32) * (1.0 / 255.0)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            pred_img = model.forward_recon(b_coords)
            l_recon, unwe_mse = weighted_reconstruction_loss(
                pred_img, b_img, cfg.bg_recon_weight, cfg.bg_threshold
            )
            l_reg = model.compute_siren_regularization(
                l2_weight=cfg.l2_reg_weight,
                bound_excess_weight=cfg.bound_excess_weight,
                max_bound_mult=cfg.max_weight_bound_mult
            )
            l_total = l_recon + l_reg

        if is_cuda:
            fwd_vrams[step_idx - 1] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            torch.cuda.reset_peak_memory_stats(device)

        l_total.backward()
        optimizer.step()

        if cfg.hard_clamp_siren_weights:
            model.clamp_siren_weights(max_bound_mult=cfg.max_weight_bound_mult)

        if is_cuda:
            bwd_vrams[step_idx - 1] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)

        step_times[step_idx - 1] = time.perf_counter() - t_step_start
        weighted_losses[step_idx - 1] = float(l_recon.item())
        unweighted_mses[step_idx - 1] = unwe_mse
        reg_losses[step_idx - 1] = float(l_reg.item())

        if step_idx in ckpt_step_set:
            step_snapshots[step_idx] = model.get_theta_state_dict(half_precision=cfg.save_fp16_theta)

            if cfg.eval_full_psnr_at_ckpts or should_visualize_subj:
                _, full_psnr_db, recon_vol_np = reconstruct_full_volume_and_psnr(
                    model=model,
                    coords_flat=coords_flat,
                    img_u8_gpu=img_u8_gpu,
                    spatial_shape=subj.spatial_shape,
                    amp_enabled=amp_enabled,
                    chunk_size=cfg.recon_chunk_size,
                    return_volume=should_visualize_subj
                )
                step_full_psnrs[step_idx] = full_psnr_db

                if should_visualize_subj and recon_vol_np is not None:
                    save_debug_visualization_and_volume(
                        recon_vol_np=recon_vol_np,
                        gt_img_uint8_cpu=subj.img_uint8,
                        spatial_shape=subj.spatial_shape,
                        affine=subj.affine,
                        log_dir=s2a_ckpt_dir.parent / 'images',
                        subject_id=subj.subject_id,
                        step_idx=step_idx,
                        full_psnr_db=full_psnr_db
                    )
            else:
                batch_psnr = 10.0 * math.log10(1.0 / unwe_mse) if unwe_mse > 0.0 else 100.0
                step_full_psnrs[step_idx] = batch_psnr

    return (
        step_snapshots,
        step_full_psnrs,
        weighted_losses,
        unweighted_mses,
        reg_losses,
        step_times,
        fwd_vrams,
        bwd_vrams,
    )


def process_single_stage1_checkpoint(
    s1_ckpt_path: Path,
    s1_ckpt_data: Dict[str, Any],
    phase2_root_dir: Path,
    cached_train: List[CachedImageUint8],
    coord_cache: CoordinateGridCache,
    cfg: Stage2AConfig,
    device: torch.device
) -> None:
    """
    Executes Phase 2A for a single Stage 1 checkpoint across all training subjects,
    creating `<phase2_root_dir>/<s1_ckpt_stem>/` with power-of-two checkpoints,
    optional debug visualizations/NIfTIs, and per-step CSV metric logs.

    Args:
        s1_ckpt_path (Path): Path to the source Stage 1 `.pth` checkpoint.
        s1_ckpt_data (Dict[str, Any]): Loaded Stage 1 checkpoint payload.
        phase2_root_dir (Path): Path to `<run_dir>/phase2/`.
        cached_train (List[CachedImageUint8]): Preloaded uint8 training intensity volumes.
        coord_cache (CoordinateGridCache): GPU coordinate grid cache.
        cfg (Stage2AConfig): Phase 2A configuration parameters.
        device (torch.device): Target compute device.
    """
    s1_stem = s1_ckpt_path.stem
    s2a_out_dir = phase2_root_dir / s1_stem
    s2a_ckpt_dir = s2a_out_dir / "checkpoints"
    csv_log_path = s2a_out_dir / "stage2a_metrics_log.csv"
    subj_csv_path = s2a_out_dir / "stage2a_subject_summary.csv"

    checkpoint_steps = [
        s for s in range(1, cfg.max_fit_steps + 1)
        if is_power_of_two_step(s, cfg.min_checkpoint_step, cfg.max_fit_steps)
    ]

    if cfg.skip_existing_checkpoints and csv_log_path.exists() and not cfg.enable_visualization:
        all_exist = all((s2a_ckpt_dir / f"stage2a_step_{s:04d}.pth").exists() for s in checkpoint_steps)
        if all_exist:
            print(f"Skipping {s1_stem}: all Phase 2A checkpoints already exist in {s2a_ckpt_dir}")
            return

    s2a_ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = s1_ckpt_data["model_config"]
    model = MetaSegINR(
        in_dim=int(model_cfg["in_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        num_classes=int(model_cfg["num_classes"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        omega_0=float(model_cfg["omega_0"])
    ).to(device)

    model.load_state_dict(s1_ckpt_data["model_state_dict"])

    for p in model.get_phi_parameters():
        p.requires_grad_(False)
    theta_params = model.get_theta_parameters()
    for p in theta_params:
        p.requires_grad_(True)

    init_theta_buffers: List[torch.Tensor] = [p.data.clone() for p in theta_params]

    spatial_shape = cached_train[0].spatial_shape
    num_voxels = int(np.prod(spatial_shape))
    batch_size = max(1, int(round(num_voxels * cfg.sample_prop)))
    amp_enabled = cfg.use_amp and device.type == 'cuda'

    num_subjects = len(cached_train)
    T = cfg.max_fit_steps

    all_weighted_losses = np.zeros((num_subjects, T), dtype=np.float64)
    all_unweighted_mses = np.zeros((num_subjects, T), dtype=np.float64)
    all_reg_losses = np.zeros((num_subjects, T), dtype=np.float64)
    all_step_times = np.zeros((num_subjects, T), dtype=np.float64)
    all_fwd_vrams = np.zeros((num_subjects, T), dtype=np.float64)
    all_bwd_vrams = np.zeros((num_subjects, T), dtype=np.float64)

    bundled_thetas_by_step: Dict[int, Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = {
        s: {} for s in checkpoint_steps
    }
    full_psnrs_by_step: Dict[int, List[float]] = {s: [] for s in checkpoint_steps}
    subject_summary_rows: List[Dict[str, Any]] = []

    print(f"\n--- Running Phase 2A for Stage 1 Checkpoint: {s1_stem} "
          f"(Loops: {s1_ckpt_data.get('dataset_loops')}, Outer Steps: {s1_ckpt_data.get('outer_steps')}) ---")

    t_ckpt_start = time.perf_counter()

    for subj_idx, subj in enumerate(
        tqdm.tqdm(cached_train, desc=f"Phase 2A [{s1_stem}] Subjects")
    ):
        coords_flat = coord_cache.get_coords(subj.spatial_shape)
        cur_batch_size = min(batch_size, coords_flat.shape[0])
        img_u8_gpu = subj.img_uint8.to(device, non_blocking=True)

        should_vis = cfg.enable_visualization and (
            cfg.max_vis_subjects < 0 or subj_idx < cfg.max_vis_subjects
        )

        (
            subj_snapshots,
            subj_full_psnrs,
            w_losses,
            u_mses,
            r_losses,
            s_times,
            f_vrams,
            b_vrams,
        ) = fit_single_subject_trajectory(
            model=model,
            theta_params=theta_params,
            init_theta_buffers=init_theta_buffers,
            subj=subj,
            coords_flat=coords_flat,
            img_u8_gpu=img_u8_gpu,
            batch_size=cur_batch_size,
            checkpoint_steps=checkpoint_steps,
            s2a_ckpt_dir=s2a_ckpt_dir,
            should_visualize_subj=should_vis,
            cfg=cfg,
            amp_enabled=amp_enabled
        )

        all_weighted_losses[subj_idx] = w_losses
        all_unweighted_mses[subj_idx] = u_mses
        all_reg_losses[subj_idx] = r_losses
        all_step_times[subj_idx] = s_times
        all_fwd_vrams[subj_idx] = f_vrams
        all_bwd_vrams[subj_idx] = b_vrams

        for s_step, theta_sd in subj_snapshots.items():
            bundled_thetas_by_step[s_step][subj.subject_id] = theta_sd
            step_mse = float(u_mses[s_step - 1])
            batch_psnr = 10.0 * math.log10(1.0 / step_mse) if step_mse > 0.0 else 100.0
            exact_full_psnr = subj_full_psnrs[s_step]
            full_psnrs_by_step[s_step].append(exact_full_psnr)

            subject_summary_rows.append({
                "stage1_checkpoint": s1_stem,
                "stage1_dataset_loops": int(s1_ckpt_data.get("dataset_loops", 0)),
                "stage1_outer_steps": int(s1_ckpt_data.get("outer_steps", 0)),
                "subject_id": subj.subject_id,
                "fit_step": s_step,
                "weighted_recon_loss": float(w_losses[s_step - 1]),
                "unweighted_mse": step_mse,
                "batch_psnr_db": round(batch_psnr, 4),
                "full_vol_psnr_db": round(exact_full_psnr, 4),
                "reg_loss": float(r_losses[s_step - 1]),
                "cumulative_subj_opt_time_sec": round(float(np.sum(s_times[:s_step])), 4),
            })

        if should_vis:
            vis_psnr_str = ", ".join(
                [f"step {s}: {subj_full_psnrs[s]:.2f} dB" for s in checkpoint_steps]
            )
            tqdm.tqdm.write(
                f"   [Vis Saved: {subj.subject_id}] Full-Vol PSNR -> {vis_psnr_str}"
            )

    if device.type == 'cuda':
        torch.cuda.synchronize()
        reserved_vram_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
    else:
        reserved_vram_mb = 0.0

    # Aggregate per-step metrics across all subjects
    per_step_batch_psnrs = 10.0 * np.log10(1.0 / np.maximum(all_unweighted_mses, 1e-12))
    mean_w_loss = np.mean(all_weighted_losses, axis=0)
    std_w_loss = np.std(all_weighted_losses, axis=0)
    mean_u_mse = np.mean(all_unweighted_mses, axis=0)
    mean_batch_psnr = np.mean(per_step_batch_psnrs, axis=0)
    std_batch_psnr = np.std(per_step_batch_psnrs, axis=0)
    mean_reg = np.mean(all_reg_losses, axis=0)
    total_step_time_across_subjs = np.sum(all_step_times, axis=0)
    cumulative_opt_time_by_step = np.cumsum(total_step_time_across_subjs)
    max_fwd_vram_by_step = np.max(all_fwd_vrams, axis=0)
    max_bwd_vram_by_step = np.max(all_bwd_vrams, axis=0)

    metrics_history: List[Dict[str, Any]] = []
    ckpt_step_set = set(checkpoint_steps)

    for step_idx in range(1, T + 1):
        i = step_idx - 1
        is_ckpt = step_idx in ckpt_step_set
        ckpt_fname = f"stage2a_step_{step_idx:04d}.pth" if is_ckpt else ""
        mean_full_psnr_val = (
            round(float(np.mean(full_psnrs_by_step[step_idx])), 4)
            if is_ckpt and full_psnrs_by_step[step_idx] else np.nan
        )
        std_full_psnr_val = (
            round(float(np.std(full_psnrs_by_step[step_idx])), 4)
            if is_ckpt and full_psnrs_by_step[step_idx] else np.nan
        )

        metrics_history.append({
            "stage1_checkpoint": s1_stem,
            "stage1_dataset_loops": int(s1_ckpt_data.get("dataset_loops", 0)),
            "stage1_outer_steps": int(s1_ckpt_data.get("outer_steps", 0)),
            "stage1_opt_time_sec": float(s1_ckpt_data.get("elapsed_optimization_time_sec", 0.0)),
            "fit_step": step_idx,
            "total_subject_steps": step_idx * num_subjects,
            "elapsed_stage2a_opt_time_sec": round(float(cumulative_opt_time_by_step[i]), 4),
            "step_opt_time_sec": round(float(total_step_time_across_subjs[i]), 4),
            "mean_weighted_recon_loss": float(mean_w_loss[i]),
            "std_weighted_recon_loss": float(std_w_loss[i]),
            "mean_unweighted_mse": float(mean_u_mse[i]),
            "mean_batch_psnr_db": round(float(mean_batch_psnr[i]), 4),
            "std_batch_psnr_db": round(float(std_batch_psnr[i]), 4),
            "mean_full_vol_psnr_db": mean_full_psnr_val,
            "std_full_vol_psnr_db": std_full_psnr_val,
            "mean_reg_loss": float(mean_reg[i]),
            "fwd_peak_vram_mb": round(float(max_fwd_vram_by_step[i]), 2),
            "bwd_peak_vram_mb": round(float(max_bwd_vram_by_step[i]), 2),
            "reserved_vram_mb": round(reserved_vram_mb, 2),
            "sample_prop": cfg.sample_prop,
            "checkpoint_file": ckpt_fname,
        })

    pd.DataFrame(metrics_history).to_csv(csv_log_path, index=False)
    pd.DataFrame(subject_summary_rows).to_csv(subj_csv_path, index=False)

    # Save bundled Stage 2A checkpoints at each power-of-two step
    for s_step in checkpoint_steps:
        i = s_step - 1
        ckpt_file = s2a_ckpt_dir / f"stage2a_step_{s_step:04d}.pth"
        mean_full_psnr_step = float(np.mean(full_psnrs_by_step[s_step]))
        payload: Dict[str, Any] = {
            "stage1_checkpoint_name": s1_stem,
            "stage1_checkpoint_path": str(s1_ckpt_path),
            "stage1_dataset_loops": int(s1_ckpt_data.get("dataset_loops", 0)),
            "stage1_outer_steps": int(s1_ckpt_data.get("outer_steps", 0)),
            "stage1_elapsed_opt_time_sec": float(s1_ckpt_data.get("elapsed_optimization_time_sec", 0.0)),
            "fit_steps": s_step,
            "elapsed_stage2a_opt_time_sec": float(cumulative_opt_time_by_step[i]),
            "mean_weighted_recon_loss": float(mean_w_loss[i]),
            "mean_batch_psnr_db": float(mean_batch_psnr[i]),
            "mean_full_vol_psnr_db": mean_full_psnr_step,
            "fwd_peak_vram_mb": float(max_fwd_vram_by_step[i]),
            "bwd_peak_vram_mb": float(max_bwd_vram_by_step[i]),
            "reserved_vram_mb": reserved_vram_mb,
            "compression_factor": float(s1_ckpt_data.get("compression_factor", 10.0)),
            "sample_prop": cfg.sample_prop,
            "save_fp16_theta": cfg.save_fp16_theta,
            "spatial_shape": spatial_shape,
            "train_subject_ids": [s.subject_id for s in cached_train],
            "model_config": model_cfg,
            "stage1_config": s1_ckpt_data.get("stage1_config", {}),
            "stage2a_config": asdict(cfg),
            "stage1_model_state_dict": s1_ckpt_data["model_state_dict"],
            "stage1_theta_state_dict": s1_ckpt_data["theta_state_dict"],
            "subject_thetas": bundled_thetas_by_step[s_step],
        }
        torch.save(payload, ckpt_file)
        print(
            f"   -> Saved {ckpt_file.name} | "
            f"Mean Full-Vol PSNR: {mean_full_psnr_step:.2f} dB | "
            f"Mean Batch PSNR: {mean_batch_psnr[i]:.2f} dB"
        )

    total_wall = time.perf_counter() - t_ckpt_start
    final_full_psnr = float(np.mean(full_psnrs_by_step[checkpoint_steps[-1]]))
    print(
        f"Completed Phase 2A for {s1_stem} in {total_wall:.1f}s | "
        f"Step {T} Mean Full-Vol PSNR: {final_full_psnr:.2f} dB | "
        f"VRAM (Fwd/Bwd): {max_fwd_vram_by_step[-1]:.0f}/{max_bwd_vram_by_step[-1]:.0f} MB"
    )


def run_stage2a_pipeline(
    dataset_dir: Path,
    run_dir: Path,
    cfg: Stage2AConfig,
    device: torch.device,
    specific_stage1_ckpt: Optional[str] = None
) -> None:
    """
    Discovers all Stage 1 checkpoints in `<run_dir>/phase1/checkpoints/`, preloads the exact
    Stage 1 training subjects into CPU RAM as `uint8`, and executes Phase 2A for each checkpoint.

    Args:
        dataset_dir (Path): Root dataset path containing the `train` subdirectory.
        run_dir (Path): Timestamped run directory containing the `phase1/` folder.
        cfg (Stage2AConfig): Phase 2A hyperparameter configuration.
        device (torch.device): Target compute device.
        specific_stage1_ckpt (Optional[str]): Optional filename filter to process a single Stage 1 checkpoint.
    """
    if not (0.0 < cfg.sample_prop <= 1.0):
        raise ValueError(f"sample_prop must satisfy 0 < sample_prop <= 1.0, got {cfg.sample_prop}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    phase1_dir = run_dir / "phase1"
    phase1_ckpt_dir = phase1_dir / "checkpoints"
    if not phase1_ckpt_dir.exists():
        raise FileNotFoundError(f"Phase 1 checkpoints directory not found: {phase1_ckpt_dir}")

    if specific_stage1_ckpt is not None:
        candidate_path = phase1_ckpt_dir / specific_stage1_ckpt
        if not candidate_path.exists():
            raise FileNotFoundError(f"Specified Stage 1 checkpoint not found: {candidate_path}")
        s1_ckpt_paths = [candidate_path]
    else:
        s1_ckpt_paths = sorted(phase1_ckpt_dir.glob("stage1_loop_*.pth"))

    if not s1_ckpt_paths:
        raise FileNotFoundError(f"No 'stage1_loop_*.pth' checkpoints found in {phase1_ckpt_dir}")

    first_ckpt = torch.load(s1_ckpt_paths[0], map_location="cpu", weights_only=False)
    train_subject_dirs = resolve_training_subject_dirs(dataset_dir, first_ckpt, phase1_dir)
    if not train_subject_dirs:
        raise FileNotFoundError(f"Could not resolve any training subjects in {dataset_dir / 'train'}")

    cached_train = preload_images_as_uint8(
        train_subject_dirs,
        clip_percentile=cfg.clip_percentile,
        pin_memory=cfg.pin_cached_memory
    )

    phase2_root_dir = run_dir / "phase2"
    phase2_root_dir.mkdir(parents=True, exist_ok=True)

    config_file_path = phase2_root_dir / "stage2a_config.yml"
    with open(config_file_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    checkpoint_steps = [
        s for s in range(1, cfg.max_fit_steps + 1)
        if is_power_of_two_step(s, cfg.min_checkpoint_step, cfg.max_fit_steps)
    ]

    print("==================================================================")
    print(f"Phase 2A Scan-Specific Backbone Fitting Configuration")
    print(f"Run Root Directory         : {run_dir}")
    print(f"Phase 1 Checkpoints Found  : {len(s1_ckpt_paths)} ({[p.name for p in s1_ckpt_paths]})")
    print(f"Phase 2 Root Directory     : {phase2_root_dir}")
    print(f"Active Training Subjects   : {len(cached_train)}")
    print(f"Stochastic Batch Proportion: {cfg.sample_prop:.4f}")
    print(f"Max Fitting Steps (T_f)    : {cfg.max_fit_steps}")
    print(f"Power-of-2 Checkpoints     : {checkpoint_steps}")
    print(f"Enable Visualization       : {cfg.enable_visualization} (Max Vis Subjects: {cfg.max_vis_subjects})")
    print(f"Store Subject Thetas FP16  : {cfg.save_fp16_theta}")
    print("==================================================================")

    coord_cache = CoordinateGridCache(device=device)

    def _handle_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        for s1_path in s1_ckpt_paths:
            s1_data = torch.load(s1_path, map_location="cpu", weights_only=False)
            process_single_stage1_checkpoint(
                s1_ckpt_path=s1_path,
                s1_ckpt_data=s1_data,
                phase2_root_dir=phase2_root_dir,
                cached_train=cached_train,
                coord_cache=coord_cache,
                cfg=cfg,
                device=device
            )
    except KeyboardInterrupt:
        print("\nPhase 2A interrupted by user/signal. Completed Stage 1 checkpoint folders are safely saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 2A Scan-Specific Backbone Fitting for Full-Resolution 3D MetaSeg. "
                    "CLI flags default to None and fall back to Stage2AConfig dataclass defaults."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/vol/miltank/projects/L2R/OASIS_L2R",
        help="Path to the root of the dataset containing the train subdirectory."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped run directory (e.g., .../OASIS_METASEG/cf_10/2026-10-21_18-00-00)."
    )
    parser.add_argument(
        "--specific_stage1_ckpt",
        type=str,
        default=None,
        help="Optional specific Stage 1 checkpoint filename (e.g., stage1_loop_0064.pth) to process alone."
    )
    parser.add_argument(
        "--enable_visualization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save middle-slice PNGs and reconstructed NIfTI volumes in the checkpoint directory."
    )
    parser.add_argument(
        "--max_vis_subjects",
        type=int,
        default=None,
        help="Number of subjects to save PNG/NIfTI visualizations for (-1 saves all subjects, default 1)."
    )
    parser.add_argument(
        "--sample_prop",
        type=float,
        default=None,
        help="Proportion of voxels (0 < sample_prop <= 1) sampled per fitting step."
    )
    parser.add_argument(
        "--max_fit_steps",
        type=int,
        default=None,
        help="Maximum fitting steps T_f per scan (overrides Stage2AConfig.max_fit_steps, default 256)."
    )
    parser.add_argument(
        "--min_checkpoint_step",
        type=int,
        default=None,
        help="Minimum power-of-2 step to save checkpoints (overrides Stage2AConfig.min_checkpoint_step, default 2)."
    )
    parser.add_argument(
        "--fit_lr",
        type=float,
        default=None,
        help="Adam learning rate for scan fitting (overrides Stage2AConfig.fit_lr)."
    )
    parser.add_argument(
        "--l2_reg_weight",
        type=float,
        default=None,
        help="L2 weight regularization coefficient (overrides Stage2AConfig.l2_reg_weight)."
    )
    parser.add_argument(
        "--bound_excess_weight",
        type=float,
        default=None,
        help="Bound-excess regularization weight (overrides Stage2AConfig.bound_excess_weight)."
    )
    parser.add_argument(
        "--max_weight_bound_mult",
        type=float,
        default=None,
        help="Allowed multiple of SIREN initialization bound (overrides Stage2AConfig.max_weight_bound_mult)."
    )
    parser.add_argument(
        "--hard_clamp_siren_weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hard-clamp SIREN weights (overrides Stage2AConfig.hard_clamp_siren_weights)."
    )
    parser.add_argument(
        "--save_fp16_theta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save per-subject theta_j^* weights in float16 to halve disk footprint."
    )
    parser.add_argument(
        "--skip_existing_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Stage 1 checkpoints whose Phase 2A outputs are already complete."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 2A on {device}...")

    dataset_path = Path(args.dataset_path)
    run_dir_path = Path(args.run_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")
    if not run_dir_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")

    cfg_overrides = {
        k: v for k, v in vars(args).items()
        if k in Stage2AConfig.__dataclass_fields__ and v is not None
    }
    stage2a_cfg = Stage2AConfig(**cfg_overrides)

    run_stage2a_pipeline(
        dataset_dir=dataset_path,
        run_dir=run_dir_path,
        cfg=stage2a_cfg,
        device=device,
        specific_stage1_ckpt=args.specific_stage1_ckpt
    )