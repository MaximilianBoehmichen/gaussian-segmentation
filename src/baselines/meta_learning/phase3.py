"""
========================================================================================
Phase 3: Test-Time Inference and Full-Volume Evaluation Pipeline
========================================================================================

1. THEORETICAL CONTEXT & OBJECTIVE
----------------------------------
In the MetaSeg framework, Stage 3 executes test-time inference ("Fit Pixels, Get Labels")[cite: 2]:
  1. Given an unseen test/validation subject, the meta-learned SIREN reconstruction
     backbone is initialized at theta^* (from Stage 1) and fine-tuned for T_f steps
     using strictly the pixel reconstruction loss L_recon[cite: 4, 5].
  2. Ground-truth segmentation masks are never seen during this adaptation[cite: 2, 4].
  3. Once fitted, the penultimate features f_{theta^*}^{L-1}(x) are extracted across the volume
     and passed through the frozen segmentation head g_{phi^*} (optimized in Stage 2B) in a
     single feed-forward operation to decode per-voxel segmentation labels[cite: 3, 4].

Evaluating all configurations across:
  - 9 Stage 1 meta-learning loops (T_o in {4, 8, ..., 1024})[cite: 4, 5],
  - 8 Stage 2A pixel-fitting horizons (T_f in {2, 4, ..., 256})[cite: 4, 5],
  - 7 Stage 2B head training loops (E_head in {2, 4, ..., 128}), and
  - N_val evaluation subjects,
presents a naive combinatorial complexity of 9 * 8 * 7 * N_val = 504 * N_val independent
test-time optimizations and full-volume reconstructions.

2. CORE SYSTEMS & ALGORITHMIC OPTIMIZATIONS
-------------------------------------------
This script reduces that computational burden by ~56x without altering the mathematical
formulation of the model or evaluations through two key optimizations:

  A. Unified Trajectory Sharing per (Stage 1 Checkpoint, Subject):
     During test-time pixel fitting, the segmentation head g_{phi^*} is completely unused;
     only the backbone f_theta is updated on L_recon[cite: 2, 4]. Because all Phase 2B heads for a given
     Stage 1 checkpoint share the exact same initialization theta^*, running independent
     pixel-fitting trajectories for every head is redundant[cite: 4].
     Instead, for each validation subject and Stage 1 checkpoint, the script executes
     EXACTLY ONE continuous 256-step fitting trajectory starting from theta^*. As this
     trajectory reaches power-of-two steps (T_f in {2, 4, ..., 256}), the current state
     is evaluated against the heads trained for that specific T_f horizon[cite: 4, 5].
     This also guarantees that all Stage 2B head variants (loops 2..128) are evaluated
     on identical fine-tuned penultimate features, eliminating stochastic mini-batch
     variance when comparing head training durations.

  B. Single-Pass Multi-Head Feature Forwarding:
     At each checkpoint step T_f, the full 3D volume (6.88M voxels) is reconstructed in
     memory-safe coordinate chunks (recon_chunk_size = 500,000). For each chunk:
       - The 5-layer SIREN backbone features `feats = f_theta(coords)` and reconstructed
         intensities `pred = recon_out(feats)` are computed ONCE.
       - While `feats` resides in GPU cache/VRAM, it is sequentially passed through all
         loaded Phase 2B segmentation heads (E_head in {2, ..., 128}) for that T_f.
       - Intersections and cardinalities for all classes and all heads are accumulated
         in parallel before discarding chunk activations.
     This avoids re-evaluating the 5 SIREN layers 7 separate times per chunk.

3. MEMORY MANAGEMENT & NUMERICAL STABILITY
------------------------------------------
  - Zero-Disk-I/O RAM Caching: Validation intensity volumes and ground-truth segmentation
    masks are preloaded once into CPU RAM as `torch.uint8` tensors (with page-locked
    pinned memory), cutting PCIe transfer volume and CPU RAM usage by ~87.5% compared to
    float32/int64 representations.
  - VRAM Safeguards: Backbone intermediate activations are dropped after feature extraction;
    loaded segmentation heads occupy <5 MB of total VRAM, and bfloat16 AMP halves activation
    memory overhead during chunked inference.
  - Reproducibility & Geometric Integrity: Random seeds are deterministically pinned per
    (Stage 1 checkpoint, subject_idx) pair. Original 4x4 NIfTI affine matrices are cached
    and preserved so that any exported debug volumes match physical scan coordinate spaces.

4. LOGGING & OUTPUT STRUCTURE
-----------------------------
Results are written under `<run_dir>/phase3/`:
  - `stage3_all_subject_evaluations.csv`: Granular record containing every evaluation
    instance (Stage 1 checkpoint, Stage 2A step, Stage 2B loop, subject_id, full-volume
    reconstruction MSE, exact full-volume PSNR, macro Dice all-class, macro Dice foreground,
    and individual per-class Dice scores).
  - `stage3_checkpoint_summary.csv`: Aggregated master table reporting mean and standard
    deviation of PSNR and Dice scores across all validation subjects for every checkpoint.
  - Optional debug NIfTI volumes and 4-row orthogonal middle-slice PNG comparisons are saved
    under `<run_dir>/phase3/<s1_stem>/images/` when `--enable_visualization` is flagged.
========================================================================================
"""

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
class Stage3Config:
    """Configuration hyperparameters for Phase 3 (Validation/Test Inference & Evaluation)."""
    eval_split: str = "val"                     # Dataset split to evaluate ('val' or 'test')
    max_eval_subjects: int = -1                 # Limit on validation subjects (-1 evaluates all available subjects)
    sample_prop: Optional[float] = None         # Voxel proportion sampled per step (None = inherit from Stage 2A config)
    fit_lr: Optional[float] = None              # Test-time Adam learning rate (None = inherit from Stage 2A config)
    num_classes: int = 5                        # aligned_seg4.nii.gz classes: 0 (bg), 1, 2, 3, 4
    seed: int = 42

    # Regularization & loss parameters (None = inherit from Stage 2A checkpoint config)
    l2_reg_weight: Optional[float] = None
    bound_excess_weight: Optional[float] = None
    max_weight_bound_mult: Optional[float] = None
    hard_clamp_siren_weights: Optional[bool] = None
    bg_recon_weight: Optional[float] = None
    bg_threshold: Optional[float] = None

    # Full-volume evaluation & optional debug visualization
    recon_chunk_size: int = 500_000             # Voxel chunk size for full-volume PSNR and Dice evaluation
    enable_visualization: bool = False          # If True, saves middle-slice PNG and NIfTI volumes in images/
    max_vis_subjects: int = 1                   # Number of validation subjects to export PNG/NIfTI for (-1 = all)

    # Caching & hardware settings
    clip_percentile: float = 99.95              # Upper intensity percentile clipped prior to uint8 quantization
    pin_cached_memory: bool = True              # Pin uint8 validation volumes/masks in page-locked CPU RAM
    use_amp: bool = True                        # Use bfloat16 autocast
    skip_existing_stage1_dirs: bool = False     # If True, skips Stage 1 folders that already have Phase 3 CSVs


@dataclass
class CachedValSubjectUint8:
    """Container holding a single validation subject's full-res image and mask in CPU RAM as uint8."""
    subject_id: str
    spatial_shape: Tuple[int, int, int]
    affine: np.ndarray        # Original (4, 4) NIfTI affine matrix
    img_uint8: torch.Tensor   # Shape: (N_voxels, out_dim), dtype: torch.uint8, values in [0, 255]
    seg_uint8: torch.Tensor   # Shape: (N_voxels,), dtype: torch.uint8, values in {0..C-1}


@dataclass
class LoadedHeadCheckpoint:
    """Metadata and segmentation head module for a single Phase 2B checkpoint."""
    s2a_stem: str
    s2a_fit_steps: int
    s2b_stem: str
    s2b_head_loops: int
    s2b_head_grad_steps: int
    s2a_opt_time_sec: float
    s2b_opt_time_sec: float
    ckpt_path: Path
    seg_head: nn.Module


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


def build_seg_head(hidden_dim: int, num_classes: int) -> nn.Sequential:
    """
    Constructs a standalone 2-layer MetaSeg segmentation head g_phi.

    Args:
        hidden_dim (int): Penultimate feature dimensionality h.
        num_classes (int): Number of output segmentation classes C.

    Returns:
        nn.Sequential: 2-layer LeakyReLU segmentation head module.
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Linear(hidden_dim, num_classes)
    )


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

        self.seg_head = build_seg_head(hidden_dim, num_classes)

    def get_theta_parameters(self) -> List[nn.Parameter]:
        """Returns parameters belonging to the reconstruction network f_theta."""
        return list(self.feature_layers.parameters()) + list(self.recon_out.parameters())

    def get_phi_parameters(self) -> List[nn.Parameter]:
        """Returns parameters belonging to the segmentation head g_phi."""
        return list(self.seg_head.parameters())

    def load_theta_state_dict(self, theta_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Loads f_theta parameters from a saved theta state dictionary.

        Args:
            theta_state (Dict[str, Dict[str, torch.Tensor]]): State dictionary containing
                'feature_layers' and 'recon_out' weight tensors.
        """
        self.feature_layers.load_state_dict(theta_state["feature_layers"])
        self.recon_out.load_state_dict(theta_state["recon_out"])

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

    def forward_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes penultimate features f_theta^{L-1}(x).

        Args:
            coords (torch.Tensor): Normalized coordinates of shape (N, in_dim) in [-1, 1].

        Returns:
            torch.Tensor: Penultimate features of shape (N, hidden_dim).
        """
        return self.feature_layers(coords)

    def forward_recon(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Predicts only image intensities I(x) = f_theta(x) for test-time pixel fitting.

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
    alongside unweighted MSE.

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


def discover_split_subjects(split_dir: Path) -> List[Path]:
    """
    Finds all valid subject directories inside a split directory (`val` or `test`).

    Args:
        split_dir (Path): Directory path corresponding to the evaluation split.

    Returns:
        List[Path]: Sorted list of subject paths containing both image and segmentation NIfTIs.
    """
    if not split_dir.exists() or not split_dir.is_dir():
        return []

    valid_subjects: List[Path] = []
    for subject_path in sorted(split_dir.iterdir()):
        if not subject_path.is_dir():
            continue
        if (subject_path / "aligned_orig.nii.gz").exists() and (subject_path / "aligned_seg4.nii.gz").exists():
            valid_subjects.append(subject_path)
    return valid_subjects


def preload_val_subjects_as_uint8(
    subject_dirs: List[Path],
    clip_percentile: float = 99.95,
    pin_memory: bool = True
) -> List[CachedValSubjectUint8]:
    """
    Loads all validation/test subjects into CPU RAM as flattened `torch.uint8` tensors
    along with their original NIfTI affine matrices.

    Args:
        subject_dirs (List[Path]): List of validation subject directories to preload.
        clip_percentile (float): Upper intensity percentile clipped prior to [0, 255] scaling.
        pin_memory (bool): If True and CUDA is available, pins uint8 tensors in page-locked CPU RAM.

    Returns:
        List[CachedValSubjectUint8]: List of preloaded validation subjects stored in CPU RAM.
    """
    cached_list: List[CachedValSubjectUint8] = []
    total_bytes = 0
    use_pin = pin_memory and torch.cuda.is_available()

    print(f"Preloading {len(subject_dirs)} evaluation subjects into CPU RAM as uint8...")
    for subj_dir in tqdm.tqdm(subject_dirs, desc="Caching eval volumes (uint8)"):
        img_path = subj_dir / "aligned_orig.nii.gz"
        seg_path = subj_dir / "aligned_seg4.nii.gz"

        nii_img = nib.load(str(img_path))
        affine = nii_img.affine.copy()

        img_np = np.transpose(nii_img.get_fdata()).astype(np.float32)
        seg_np = np.transpose(nib.load(str(seg_path)).get_fdata()).astype(np.uint8)

        img_np = np.squeeze(img_np)
        seg_np = np.squeeze(seg_np)
        spatial_shape: Tuple[int, int, int] = (int(img_np.shape[0]), int(img_np.shape[1]), int(img_np.shape[2]))

        min_val = float(img_np.min())
        max_val = float(np.percentile(img_np, clip_percentile)) if clip_percentile < 100.0 else float(img_np.max())
        if max_val > min_val:
            img_norm = np.clip((img_np - min_val) / (max_val - min_val), 0.0, 1.0)
        else:
            img_norm = np.zeros_like(img_np, dtype=np.float32)

        img_uint8_np = np.rint(img_norm * 255.0).astype(np.uint8)

        img_tensor = torch.from_numpy(img_uint8_np).reshape(-1, 1).contiguous()
        seg_tensor = torch.from_numpy(seg_np).reshape(-1).contiguous()

        if use_pin:
            img_tensor = img_tensor.pin_memory()
            seg_tensor = seg_tensor.pin_memory()

        total_bytes += img_tensor.numel() * img_tensor.element_size()
        total_bytes += seg_tensor.numel() * seg_tensor.element_size()

        cached_list.append(
            CachedValSubjectUint8(
                subject_id=subj_dir.name,
                spatial_shape=spatial_shape,
                affine=affine,
                img_uint8=img_tensor,
                seg_uint8=seg_tensor
            )
        )

    print(f"Cached {len(cached_list)} full-res eval subjects in CPU RAM ({total_bytes / (1024 ** 2):.2f} MB total).")
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


@torch.no_grad()
def evaluate_full_volume_across_heads(
    model: MetaSegINR,
    heads_for_step: List[LoadedHeadCheckpoint],
    coords_flat: torch.Tensor,
    img_u8_gpu: torch.Tensor,
    seg_u8_gpu: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    num_classes: int,
    amp_enabled: bool,
    chunk_size: int = 500_000,
    return_volumes_for_last_head: bool = False
) -> Tuple[float, float, List[Tuple[float, float, List[float]]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reconstructs 100% of the volume in chunks using the fine-tuned `f_theta` once per chunk,
    computing exact full-volume MSE/PSNR and simultaneously evaluating all Phase 2B segmentation
    heads `g_phi` trained for the current `stage2a_fit_steps` horizon.

    Args:
        model (MetaSegINR): Fine-tuned MetaSegINR model on GPU at step `T_f`.
        heads_for_step (List[LoadedHeadCheckpoint]): List of Phase 2B head checkpoints matching `T_f`.
        coords_flat (torch.Tensor): Flattened coordinate grid of shape (N_voxels, D) in [-1, 1].
        img_u8_gpu (torch.Tensor): Target uint8 intensities on GPU of shape (N_voxels, out_dim).
        seg_u8_gpu (torch.Tensor): Target uint8 class labels on GPU of shape (N_voxels,).
        spatial_shape (Tuple[int, int, int]): 3D spatial shape (Z, Y, X) of the volume.
        num_classes (int): Number of segmentation classes C.
        amp_enabled (bool): Whether bfloat16 autocast is active.
        chunk_size (int): Number of coordinates evaluated per forward chunk.
        return_volumes_for_last_head (bool): If True, returns the 3D reconstructed intensity volume
            and the 3D predicted label volume (from the final head in `heads_for_step`).

    Returns:
        Tuple containing:
            - full_mse: Exact full-volume unweighted reconstruction MSE.
            - full_psnr_db: Exact full-volume reconstruction PSNR in dB.
            - head_dice_results: List of `(dice_all, dice_fg, per_class_dices)` for each head in `heads_for_step`.
            - recon_vol_np: 3D float32 array of shape (Z, Y, X) if requested, else None.
            - pred_seg_vol_np: 3D uint8 array of shape (Z, Y, X) if requested, else None.
    """
    model.eval()
    for h_ckpt in heads_for_step:
        h_ckpt.seg_head.eval()

    num_voxels = coords_flat.shape[0]
    num_heads = len(heads_for_step)
    sq_err_sum = 0.0

    # Shape: (num_heads, num_classes)
    intersections = np.zeros((num_heads, num_classes), dtype=np.float64)
    cardinalities = np.zeros((num_heads, num_classes), dtype=np.float64)

    recon_chunks: List[torch.Tensor] = []
    seg_chunks: List[torch.Tensor] = []

    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        b_coords = coords_flat[start:end]
        b_target_img = img_u8_gpu[start:end].to(dtype=torch.float32) * (1.0 / 255.0)
        b_true_seg = seg_u8_gpu[start:end]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            # Compute penultimate SIREN features and reconstruction once for this chunk
            feats = model.forward_features(b_coords)
            pred_img = model.recon_out(feats).float()

        pred_clamped = torch.clamp(pred_img, 0.0, 1.0)
        sq_err_sum += F.mse_loss(pred_clamped, b_target_img, reduction='sum').item()

        if return_volumes_for_last_head:
            recon_chunks.append(pred_clamped.squeeze(-1).cpu())

        # Evaluate all Phase 2B heads trained for this T_f using the already-computed `feats`
        for h_idx, h_ckpt in enumerate(heads_for_step):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                logits = h_ckpt.seg_head(feats)
                pred_cls = torch.argmax(logits, dim=-1).to(dtype=torch.uint8)

            for c in range(num_classes):
                pred_c = (pred_cls == c)
                true_c = (b_true_seg == c)
                intersections[h_idx, c] += float((pred_c & true_c).sum().item())
                cardinalities[h_idx, c] += float(pred_c.sum().item() + true_c.sum().item())

            if return_volumes_for_last_head and (h_idx == num_heads - 1):
                seg_chunks.append(pred_cls.cpu())

    model.train()
    full_mse = sq_err_sum / float(num_voxels)
    full_psnr_db = 10.0 * math.log10(1.0 / full_mse) if full_mse > 0.0 else 100.0

    eps = 1e-6
    head_dice_results: List[Tuple[float, float, List[float]]] = []
    for h_idx in range(num_heads):
        per_cls_dices: List[float] = []
        for c in range(num_classes):
            if cardinalities[h_idx, c] == 0.0:
                per_cls_dices.append(1.0)
            else:
                per_cls_dices.append(
                    float((2.0 * intersections[h_idx, c] + eps) / (cardinalities[h_idx, c] + eps))
                )
        dice_all = float(np.mean(per_cls_dices))
        dice_fg = float(np.mean(per_cls_dices[1:])) if num_classes > 1 else dice_all
        head_dice_results.append((dice_all, dice_fg, per_cls_dices))

    recon_vol_np: Optional[np.ndarray] = None
    pred_seg_vol_np: Optional[np.ndarray] = None
    if return_volumes_for_last_head:
        recon_vol_np = torch.cat(recon_chunks, dim=0).view(*spatial_shape).numpy().astype(np.float32)
        pred_seg_vol_np = torch.cat(seg_chunks, dim=0).view(*spatial_shape).numpy().astype(np.uint8)

    return full_mse, full_psnr_db, head_dice_results, recon_vol_np, pred_seg_vol_np


def save_stage3_debug_visualization(
    recon_vol_np: np.ndarray,
    pred_seg_vol_np: np.ndarray,
    subj: CachedValSubjectUint8,
    log_dir: Path,
    s1_stem: str,
    fit_step: int,
    head_loop: int,
    num_classes: int,
    full_psnr_db: float,
    dice_all: float,
    dice_fg: float
) -> None:
    """
    Saves the Stage 3 reconstructed volume, predicted segmentation mask NIfTI, and a 4-row
    middle-slice comparison PNG (GT Image, Recon Image, GT Seg, Pred Seg).

    Args:
        recon_vol_np (np.ndarray): Reconstructed 3D intensity volume of shape (Z, Y, X) in [0, 1].
        pred_seg_vol_np (np.ndarray): Predicted 3D segmentation volume of shape (Z, Y, X) in {0..C-1}.
        subj (CachedValSubjectUint8): Cached validation subject container.
        log_dir (Path): Output directory for debug visualizations and NIfTI volumes.
        s1_stem (str): Stage 1 checkpoint identifier.
        fit_step (int): Test-time pixel-fitting step T_f.
        head_loop (int): Stage 2B head training loop index.
        num_classes (int): Number of segmentation classes C.
        full_psnr_db (float): Full-volume reconstruction PSNR in dB.
        dice_all (float): Full-volume macro Dice across all classes.
        dice_fg (float): Full-volume macro Dice across foreground classes.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"stage3_{s1_stem}_tf{fit_step:04d}_hl{head_loop:04d}_{subj.subject_id}"
    slice_png_path = log_dir / f"{prefix}_mid_slice.png"
    recon_nii_path = log_dir / f"{prefix}_recon_vol.nii.gz"
    seg_nii_path = log_dir / f"{prefix}_pred_seg4.nii.gz"

    nib.save(nib.Nifti1Image(np.transpose(recon_vol_np), subj.affine), str(recon_nii_path))
    nib.save(nib.Nifti1Image(np.transpose(pred_seg_vol_np), subj.affine), str(seg_nii_path))

    spatial_shape = subj.spatial_shape
    gt_img_np = (subj.img_uint8.view(*spatial_shape).numpy().astype(np.float32)) * (1.0 / 255.0)
    gt_seg_np = subj.seg_uint8.view(*spatial_shape).numpy().astype(np.uint8)

    z_mid, y_mid, x_mid = spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2
    plane_titles = [f"Mid-Plane 0 (idx={z_mid})", f"Mid-Plane 1 (idx={y_mid})", f"Mid-Plane 2 (idx={x_mid})"]

    gt_img_slices = [gt_img_np[z_mid, :, :], gt_img_np[:, y_mid, :], gt_img_np[:, :, x_mid]]
    rc_img_slices = [recon_vol_np[z_mid, :, :], recon_vol_np[:, y_mid, :], recon_vol_np[:, :, x_mid]]
    gt_seg_slices = [gt_seg_np[z_mid, :, :], gt_seg_np[:, y_mid, :], gt_seg_np[:, :, x_mid]]
    pr_seg_slices = [pred_seg_vol_np[z_mid, :, :], pred_seg_vol_np[:, y_mid, :], pred_seg_vol_np[:, :, x_mid]]

    cmap_seg = plt.get_cmap("tab10", num_classes)
    vmax_cls = max(1, num_classes - 1)

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    fig.suptitle(
        f"Stage 3 Eval | {s1_stem} | T_f={fit_step} | HeadLoop={head_loop} | {subj.subject_id}\n"
        f"Full-Vol PSNR: {full_psnr_db:.2f} dB | Dice (All): {dice_all:.4f} | Dice (FG): {dice_fg:.4f}",
        fontsize=13
    )

    for col in range(3):
        axes[0, col].imshow(gt_img_slices[col], cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[0, col].set_title(f"GT Intensity - {plane_titles[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(rc_img_slices[col], cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[1, col].set_title(f"INR Recon - {plane_titles[col]}")
        axes[1, col].axis("off")

        axes[2, col].imshow(gt_seg_slices[col], cmap=cmap_seg, vmin=0, vmax=vmax_cls, interpolation="nearest", origin="lower")
        axes[2, col].set_title(f"GT Seg - {plane_titles[col]}")
        axes[2, col].axis("off")

        axes[3, col].imshow(pr_seg_slices[col], cmap=cmap_seg, vmin=0, vmax=vmax_cls, interpolation="nearest", origin="lower")
        axes[3, col].set_title(f"Pred Seg - {plane_titles[col]}")
        axes[3, col].axis("off")

    fig.tight_layout()
    fig.savefig(slice_png_path, dpi=150)
    plt.close(fig)


def build_summary_dataframe(df_eval: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """
    Aggregates per-subject Stage 3 evaluation rows into mean/std metrics per
    `(stage1_checkpoint, stage2a_checkpoint, stage2b_checkpoint)`.

    Args:
        df_eval (pd.DataFrame): Long-format dataframe containing one row per subject evaluation.
        num_classes (int): Number of segmentation classes C.

    Returns:
        pd.DataFrame: Aggregated checkpoint summary dataframe.
    """
    group_cols = [
        "stage1_checkpoint",
        "stage1_dataset_loops",
        "stage1_outer_steps",
        "stage1_opt_time_sec",
        "stage2a_checkpoint",
        "stage2a_fit_steps",
        "stage2a_opt_time_sec",
        "stage2b_checkpoint",
        "stage2b_head_loops",
        "stage2b_head_grad_steps",
        "stage2b_opt_time_sec",
        "eval_split",
        "compression_factor",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for keys, grp in df_eval.groupby(group_cols, sort=True):
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["num_eval_subjects"] = len(grp)
        row["mean_test_fit_opt_time_sec"] = round(float(grp["test_fit_opt_time_sec"].mean()), 4)
        row["total_test_fit_opt_time_sec"] = round(float(grp["test_fit_opt_time_sec"].sum()), 4)
        row["mean_full_vol_mse"] = float(grp["full_vol_mse"].mean())
        row["mean_full_vol_psnr_db"] = round(float(grp["full_vol_psnr_db"].mean()), 4)
        row["std_full_vol_psnr_db"] = round(float(grp["full_vol_psnr_db"].std(ddof=0)), 4)
        row["mean_full_vol_dice_all"] = round(float(grp["full_vol_dice_all"].mean()), 4)
        row["std_full_vol_dice_all"] = round(float(grp["full_vol_dice_all"].std(ddof=0)), 4)
        row["mean_full_vol_dice_fg"] = round(float(grp["full_vol_dice_fg"].mean()), 4)
        row["std_full_vol_dice_fg"] = round(float(grp["full_vol_dice_fg"].std(ddof=0)), 4)

        for c in range(num_classes):
            col_c = f"dice_class_{c}"
            if col_c in grp.columns:
                row[f"mean_{col_c}"] = round(float(grp[col_c].mean()), 4)
                row[f"std_{col_c}"] = round(float(grp[col_c].std(ddof=0)), 4)

        row["fwd_peak_vram_mb"] = round(float(grp["fwd_peak_vram_mb"].max()), 2)
        row["bwd_peak_vram_mb"] = round(float(grp["bwd_peak_vram_mb"].max()), 2)
        row["reserved_vram_mb"] = round(float(grp["reserved_vram_mb"].max()), 2)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def evaluate_single_stage1_directory(
    s1_phase2_dir: Path,
    phase3_root_dir: Path,
    cached_val: List[CachedValSubjectUint8],
    coord_cache: CoordinateGridCache,
    cfg: Stage3Config,
    device: torch.device,
    specific_stage2a_ckpt: Optional[str] = None,
    specific_stage2b_ckpt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Loads all Phase 2B checkpoints under `<run_dir>/phase2/<s1_stem>/stage2b/`, groups their
    segmentation heads by their Phase 2A fitting step `T_f`, and evaluates them across all
    validation subjects in `cached_val`.

    Args:
        s1_phase2_dir (Path): Path to `<run_dir>/phase2/<s1_stem>`.
        phase3_root_dir (Path): Path to `<run_dir>/phase3/`.
        cached_val (List[CachedValSubjectUint8]): Preloaded validation subjects in CPU RAM.
        coord_cache (CoordinateGridCache): GPU coordinate grid cache.
        cfg (Stage3Config): Stage 3 evaluation configuration.
        device (torch.device): Target compute device.
        specific_stage2a_ckpt (Optional[str]): Optional filter for a specific Stage 2A stem.
        specific_stage2b_ckpt (Optional[str]): Optional filter for a specific Stage 2B checkpoint filename.

    Returns:
        List[Dict[str, Any]]: Per-subject evaluation records across all (2A, 2B, val_subject) combinations
        for this Stage 1 checkpoint.
    """
    s1_stem = s1_phase2_dir.name
    s2b_root = s1_phase2_dir / "stage2b"
    if not s2b_root.is_dir():
        print(f"Skipping {s1_stem}: no 'stage2b/' directory found in {s1_phase2_dir}")
        return []

    s3_s1_out_dir = phase3_root_dir / s1_stem
    s3_subj_csv_path = s3_s1_out_dir / "stage3_subject_evaluations.csv"
    s3_sum_csv_path = s3_s1_out_dir / "stage3_checkpoint_summary.csv"
    s3_img_dir = s3_s1_out_dir / "images"

    if cfg.skip_existing_stage1_dirs and s3_subj_csv_path.exists() and not cfg.enable_visualization:
        print(f"Skipping {s1_stem}: existing Phase 3 evaluation CSV found at {s3_subj_csv_path}")
        return pd.read_csv(s3_subj_csv_path).to_dict(orient="records")

    # Discover all Stage 2A folders and their Stage 2B checkpoints
    s2a_dirs = sorted([d for d in s2b_root.iterdir() if d.is_dir() and d.name.startswith("stage2a_")])
    if specific_stage2a_ckpt is not None:
        target_s2a_stem = Path(specific_stage2a_ckpt).stem
        s2a_dirs = [d for d in s2a_dirs if d.name == target_s2a_stem]

    heads_by_fit_step: Dict[int, List[LoadedHeadCheckpoint]] = {}
    first_ckpt_payload: Optional[Dict[str, Any]] = None
    total_2b_ckpts = 0

    for s2a_dir in s2a_dirs:
        ckpt_dir = s2a_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        s2b_files = sorted(ckpt_dir.glob("stage2b_loop_*.pth"))
        if specific_stage2b_ckpt is not None:
            target_s2b_name = (
                specific_stage2b_ckpt
                if specific_stage2b_ckpt.endswith(".pth")
                else f"{specific_stage2b_ckpt}.pth"
            )
            s2b_files = [f for f in s2b_files if f.name == target_s2b_name]

        for s2b_path in s2b_files:
            payload = torch.load(s2b_path, map_location="cpu", weights_only=False)
            if first_ckpt_payload is None:
                first_ckpt_payload = payload

            model_cfg = payload["model_config"]
            fit_step = int(payload["stage2a_fit_steps"])
            head_loops = int(payload["stage2b_head_loops"])

            head_mod = build_seg_head(
                hidden_dim=int(model_cfg["hidden_dim"]),
                num_classes=int(model_cfg["num_classes"])
            ).to(device)

            if "phi_star_state_dict" in payload:
                head_mod.load_state_dict(payload["phi_star_state_dict"])
            else:
                # Extract seg_head keys from full model_state_dict if needed
                full_sd = payload["model_state_dict"]
                phi_sd = {
                    k.replace("seg_head.", "", 1): v
                    for k, v in full_sd.items()
                    if k.startswith("seg_head.")
                }
                head_mod.load_state_dict(phi_sd)

            for p in head_mod.parameters():
                p.requires_grad_(False)
            head_mod.eval()

            loaded_head = LoadedHeadCheckpoint(
                s2a_stem=s2a_dir.name,
                s2a_fit_steps=fit_step,
                s2b_stem=s2b_path.stem,
                s2b_head_loops=head_loops,
                s2b_head_grad_steps=int(payload.get("stage2b_head_gradient_steps", 0)),
                s2a_opt_time_sec=float(payload.get("stage2a_elapsed_opt_time_sec", 0.0)),
                s2b_opt_time_sec=float(payload.get("elapsed_stage2b_opt_time_sec", 0.0)),
                ckpt_path=s2b_path,
                seg_head=head_mod
            )
            heads_by_fit_step.setdefault(fit_step, []).append(loaded_head)
            total_2b_ckpts += 1

    if first_ckpt_payload is None or not heads_by_fit_step:
        print(f"No valid Phase 2B checkpoints found for {s1_stem}.")
        return []

    s3_s1_out_dir.mkdir(parents=True, exist_ok=True)

    # Extract shared Stage 1 model configuration & hyperparameters
    model_cfg = first_ckpt_payload["model_config"]
    s2a_saved_cfg = first_ckpt_payload.get("stage2a_config", {})

    sample_prop = cfg.sample_prop if cfg.sample_prop is not None else float(s2a_saved_cfg.get("sample_prop", 0.25))
    fit_lr = cfg.fit_lr if cfg.fit_lr is not None else float(s2a_saved_cfg.get("fit_lr", 1e-4))
    l2_reg_w = cfg.l2_reg_weight if cfg.l2_reg_weight is not None else float(s2a_saved_cfg.get("l2_reg_weight", 1e-6))
    bound_ex_w = (
        cfg.bound_excess_weight
        if cfg.bound_excess_weight is not None
        else float(s2a_saved_cfg.get("bound_excess_weight", 1e-3))
    )
    max_bound_m = (
        cfg.max_weight_bound_mult
        if cfg.max_weight_bound_mult is not None
        else float(s2a_saved_cfg.get("max_weight_bound_mult", 3.0))
    )
    hard_clamp = (
        cfg.hard_clamp_siren_weights
        if cfg.hard_clamp_siren_weights is not None
        else bool(s2a_saved_cfg.get("hard_clamp_siren_weights", False))
    )
    bg_weight = cfg.bg_recon_weight if cfg.bg_recon_weight is not None else float(s2a_saved_cfg.get("bg_recon_weight", 0.1))
    bg_thresh = cfg.bg_threshold if cfg.bg_threshold is not None else float(s2a_saved_cfg.get("bg_threshold", 1e-5))

    model = MetaSegINR(
        in_dim=int(model_cfg["in_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        num_classes=int(model_cfg["num_classes"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        omega_0=float(model_cfg["omega_0"])
    ).to(device)

    # Load Stage 1 meta-learned backbone initialization theta^*
    if "theta_star_state_dict" in first_ckpt_payload:
        model.load_theta_state_dict(first_ckpt_payload["theta_star_state_dict"])
    else:
        model.load_state_dict(first_ckpt_payload["model_state_dict"])

    for p in model.get_phi_parameters():
        p.requires_grad_(False)
    theta_params = model.get_theta_parameters()
    for p in theta_params:
        p.requires_grad_(True)

    init_theta_buffers: List[torch.Tensor] = [p.data.clone() for p in theta_params]

    sorted_fit_steps = sorted(heads_by_fit_step.keys())
    max_required_step = max(sorted_fit_steps)

    s1_loops = int(first_ckpt_payload.get("stage1_dataset_loops", 0))
    s1_outer_steps = int(first_ckpt_payload.get("stage1_outer_steps", 0))
    s1_opt_time = float(first_ckpt_payload.get("stage1_elapsed_opt_time_sec", 0.0))
    compression_factor = float(first_ckpt_payload.get("compression_factor", 10.0))
    num_classes = int(model_cfg["num_classes"])

    spatial_shape = cached_val[0].spatial_shape
    num_voxels = int(np.prod(spatial_shape))
    batch_size = max(1, int(round(num_voxels * sample_prop)))
    amp_enabled = cfg.use_amp and device.type == 'cuda'
    is_cuda = device.type == 'cuda'

    print(
        f"\n--- Stage 3 Eval for [{s1_stem}] | "
        f"2A Steps: {sorted_fit_steps} | Total 2B Heads: {total_2b_ckpts} | "
        f"Eval Subjects: {len(cached_val)} ({total_2b_ckpts * len(cached_val)} evaluations) ---"
    )

    eval_records: List[Dict[str, Any]] = []

    for subj_idx, subj in enumerate(
        tqdm.tqdm(cached_val, desc=f"Stage 3 [{s1_stem}] {cfg.eval_split} Subjects")
    ):
        # Seed deterministically per (s1_stem, subject) so trajectory is strictly reproducible
        torch.manual_seed(cfg.seed + subj_idx)

        coords_flat = coord_cache.get_coords(subj.spatial_shape)
        num_subj_voxels = coords_flat.shape[0]
        cur_batch_size = min(batch_size, num_subj_voxels)

        img_u8_gpu = subj.img_uint8.to(device, non_blocking=True)
        seg_u8_gpu = subj.seg_uint8.to(device, non_blocking=True)

        # Reset f_theta to Stage 1 initialization theta^*
        with torch.no_grad():
            torch._foreach_copy_([p.data for p in theta_params], init_theta_buffers)

        optimizer = torch.optim.Adam(theta_params, lr=fit_lr)
        model.train()

        cumulative_subj_fit_time = 0.0
        max_fwd_vram_so_far = 0.0
        max_bwd_vram_so_far = 0.0

        should_vis_subj = cfg.enable_visualization and (
            cfg.max_vis_subjects < 0 or subj_idx < cfg.max_vis_subjects
        )

        for step_idx in tqdm(range(1, max_required_step + 1), desc=f"Fine-tuning subj {subj_idx}:"):
            t_step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            if cur_batch_size < num_subj_voxels:
                idx = torch.randint(0, num_subj_voxels, (cur_batch_size,), device=device)
                b_coords = coords_flat[idx]
                b_img = img_u8_gpu[idx].to(dtype=torch.float32) * (1.0 / 255.0)
            else:
                b_coords = coords_flat
                b_img = img_u8_gpu.to(dtype=torch.float32) * (1.0 / 255.0)

            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                pred_img = model.forward_recon(b_coords)
                l_recon, batch_unwe_mse = weighted_reconstruction_loss(
                    pred_img, b_img, bg_weight, bg_thresh
                )
                l_reg = model.compute_siren_regularization(
                    l2_weight=l2_reg_w,
                    bound_excess_weight=bound_ex_w,
                    max_bound_mult=max_bound_m
                )
                l_total = l_recon + l_reg

            if is_cuda:
                fwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if fwd_mb > max_fwd_vram_so_far:
                    max_fwd_vram_so_far = fwd_mb
                torch.cuda.reset_peak_memory_stats(device)

            l_total.backward()
            optimizer.step()

            if hard_clamp:
                model.clamp_siren_weights(max_bound_mult=max_bound_m)

            if is_cuda:
                bwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if bwd_mb > max_bwd_vram_so_far:
                    max_bwd_vram_so_far = bwd_mb

            cumulative_subj_fit_time += (time.perf_counter() - t_step_start)

            # If this step_idx corresponds to a trained Phase 2A horizon, reconstruct full volume & evaluate heads
            if step_idx in heads_by_fit_step:
                heads_at_step = heads_by_fit_step[step_idx]
                full_mse, full_psnr_db, head_dice_list, recon_vol_np, pred_seg_vol_np = (
                    evaluate_full_volume_across_heads(
                        model=model,
                        heads_for_step=heads_at_step,
                        coords_flat=coords_flat,
                        img_u8_gpu=img_u8_gpu,
                        seg_u8_gpu=seg_u8_gpu,
                        spatial_shape=subj.spatial_shape,
                        num_classes=num_classes,
                        amp_enabled=amp_enabled,
                        chunk_size=cfg.recon_chunk_size,
                        return_volumes_for_last_head=should_vis_subj
                    )
                )

                if is_cuda:
                    reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
                else:
                    reserved_mb = 0.0

                for h_ckpt, (dice_all, dice_fg, per_cls_dices) in zip(heads_at_step, head_dice_list):
                    record: Dict[str, Any] = {
                        "eval_split": cfg.eval_split,
                        "subject_id": subj.subject_id,
                        "stage1_checkpoint": s1_stem,
                        "stage1_dataset_loops": s1_loops,
                        "stage1_outer_steps": s1_outer_steps,
                        "stage1_opt_time_sec": round(s1_opt_time, 4),
                        "stage2a_checkpoint": h_ckpt.s2a_stem,
                        "stage2a_fit_steps": h_ckpt.s2a_fit_steps,
                        "stage2a_opt_time_sec": round(h_ckpt.s2a_opt_time_sec, 4),
                        "stage2b_checkpoint": h_ckpt.s2b_stem,
                        "stage2b_head_loops": h_ckpt.s2b_head_loops,
                        "stage2b_head_grad_steps": h_ckpt.s2b_head_grad_steps,
                        "stage2b_opt_time_sec": round(h_ckpt.s2b_opt_time_sec, 4),
                        "test_fit_steps": step_idx,
                        "test_fit_opt_time_sec": round(cumulative_subj_fit_time, 4),
                        "batch_weighted_recon_loss": round(float(l_recon.item()), 6),
                        "batch_unweighted_mse": round(batch_unwe_mse, 6),
                        "full_vol_mse": full_mse,
                        "full_vol_psnr_db": round(full_psnr_db, 4),
                        "full_vol_dice_all": round(dice_all, 4),
                        "full_vol_dice_fg": round(dice_fg, 4),
                    }
                    for c_idx, c_dice in enumerate(per_cls_dices):
                        record[f"dice_class_{c_idx}"] = round(c_dice, 4)

                    record["fwd_peak_vram_mb"] = round(max_fwd_vram_so_far, 2)
                    record["bwd_peak_vram_mb"] = round(max_bwd_vram_so_far, 2)
                    record["reserved_vram_mb"] = round(reserved_mb, 2)
                    record["compression_factor"] = compression_factor
                    record["sample_prop"] = sample_prop
                    eval_records.append(record)

                if should_vis_subj and recon_vol_np is not None and pred_seg_vol_np is not None:
                    last_h = heads_at_step[-1]
                    last_d_all, last_d_fg, _ = head_dice_list[-1]
                    save_stage3_debug_visualization(
                        recon_vol_np=recon_vol_np,
                        pred_seg_vol_np=pred_seg_vol_np,
                        subj=subj,
                        log_dir=s3_img_dir,
                        s1_stem=s1_stem,
                        fit_step=step_idx,
                        head_loop=last_h.s2b_head_loops,
                        num_classes=num_classes,
                        full_psnr_db=full_psnr_db,
                        dice_all=last_d_all,
                        dice_fg=last_d_fg
                    )

    df_s1_eval = pd.DataFrame(eval_records)
    df_s1_eval.to_csv(s3_subj_csv_path, index=False)

    df_s1_summary = build_summary_dataframe(df_s1_eval, num_classes=num_classes)
    df_s1_summary.to_csv(s3_sum_csv_path, index=False)

    best_row = df_s1_summary.loc[df_s1_summary["mean_full_vol_dice_fg"].idxmax()]
    print(
        f"Completed Stage 3 for {s1_stem} | Saved {len(df_s1_eval):,} subject eval rows -> {s3_subj_csv_path.name}\n"
        f"   Best Config for {s1_stem}: T_f={int(best_row['stage2a_fit_steps'])}, "
        f"HeadLoops={int(best_row['stage2b_head_loops'])} -> "
        f"Mean PSNR: {best_row['mean_full_vol_psnr_db']:.2f} dB | "
        f"Mean Dice (All/FG): {best_row['mean_full_vol_dice_all']:.4f}/{best_row['mean_full_vol_dice_fg']:.4f}"
    )

    return eval_records


def run_stage3_evaluation_pipeline(
    dataset_dir: Path,
    run_dir: Path,
    cfg: Stage3Config,
    device: torch.device,
    specific_stage1_ckpt: Optional[str] = None,
    specific_stage2a_ckpt: Optional[str] = None,
    specific_stage2b_ckpt: Optional[str] = None
) -> None:
    """
    Discovers all Stage 1/2A/2B checkpoints under `<run_dir>/phase2/`, preloads the validation
    (or test) split into CPU RAM as `uint8`, runs test-time pixel fitting and full-volume PSNR/Dice
    evaluation, and writes both per-Stage-1 and master CSV files to `<run_dir>/phase3/`.

    Args:
        dataset_dir (Path): Root dataset path containing `val` and `test` subdirectories.
        run_dir (Path): Timestamped run directory containing `phase2/`.
        cfg (Stage3Config): Stage 3 evaluation configuration.
        device (torch.device): Target compute device.
        specific_stage1_ckpt (Optional[str]): Optional filter for a specific Stage 1 checkpoint stem.
        specific_stage2a_ckpt (Optional[str]): Optional filter for a specific Stage 2A checkpoint stem.
        specific_stage2b_ckpt (Optional[str]): Optional filter for a specific Stage 2B checkpoint filename.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    phase2_root_dir = run_dir / "phase2"
    if not phase2_root_dir.exists():
        raise FileNotFoundError(f"Phase 2 directory not found: {phase2_root_dir}")

    split_dir = dataset_dir / cfg.eval_split
    val_subject_dirs = discover_split_subjects(split_dir)
    if not val_subject_dirs:
        raise FileNotFoundError(f"No valid evaluation subjects found in {split_dir}")

    if cfg.max_eval_subjects > 0:
        val_subject_dirs = val_subject_dirs[:cfg.max_eval_subjects]
        print(f"Capped evaluation subjects to {len(val_subject_dirs)} via max_eval_subjects.")

    cached_val = preload_val_subjects_as_uint8(
        val_subject_dirs,
        clip_percentile=cfg.clip_percentile,
        pin_memory=cfg.pin_cached_memory
    )

    if specific_stage1_ckpt is not None:
        s1_stem_clean = Path(specific_stage1_ckpt).stem
        cand_dir = phase2_root_dir / s1_stem_clean
        if not cand_dir.is_dir():
            raise FileNotFoundError(f"Specified Stage 1 directory not found in phase2: {cand_dir}")
        s1_dirs = [cand_dir]
    else:
        s1_dirs = sorted([
            d for d in phase2_root_dir.iterdir()
            if d.is_dir() and d.name.startswith("stage1_")
        ])

    if not s1_dirs:
        raise FileNotFoundError(f"No 'stage1_*' subdirectories found in {phase2_root_dir}")

    phase3_root_dir = run_dir / "phase3"
    phase3_root_dir.mkdir(parents=True, exist_ok=True)

    config_file_path = phase3_root_dir / "stage3_config.yml"
    with open(config_file_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    print("==================================================================")
    print(f"Phase 3 Test-Time Inference & Full-Volume Evaluation Configuration")
    print(f"Run Root Directory         : {run_dir}")
    print(f"Phase 3 Output Directory   : {phase3_root_dir}")
    print(f"Evaluation Split           : '{cfg.eval_split}' ({len(cached_val)} subjects)")
    print(f"Stage 1 Directories Queued : {len(s1_dirs)} ({[d.name for d in s1_dirs]})")
    print(f"Full-Volume Chunk Size     : {cfg.recon_chunk_size:,} voxels")
    print(f"Enable Visualization       : {cfg.enable_visualization} (Max Vis Subjects: {cfg.max_vis_subjects})")
    print("==================================================================")

    coord_cache = CoordinateGridCache(device=device)
    master_eval_records: List[Dict[str, Any]] = []
    master_subj_csv_path = phase3_root_dir / "stage3_all_subject_evaluations.csv"
    master_sum_csv_path = phase3_root_dir / "stage3_checkpoint_summary.csv"

    def _handle_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        for s1_dir in s1_dirs:
            s1_records = evaluate_single_stage1_directory(
                s1_phase2_dir=s1_dir,
                phase3_root_dir=phase3_root_dir,
                cached_val=cached_val,
                coord_cache=coord_cache,
                cfg=cfg,
                device=device,
                specific_stage2a_ckpt=specific_stage2a_ckpt,
                specific_stage2b_ckpt=specific_stage2b_ckpt
            )
            master_eval_records.extend(s1_records)

            if master_eval_records:
                df_master = pd.DataFrame(master_eval_records)
                df_master.to_csv(master_subj_csv_path, index=False)
                df_master_sum = build_summary_dataframe(df_master, num_classes=cfg.num_classes)
                df_master_sum.to_csv(master_sum_csv_path, index=False)

    except KeyboardInterrupt:
        print("\nPhase 3 interrupted by user/signal. Flushing completed evaluations to master CSVs...")
        if master_eval_records:
            df_master = pd.DataFrame(master_eval_records)
            df_master.to_csv(master_subj_csv_path, index=False)
            df_master_sum = build_summary_dataframe(df_master, num_classes=cfg.num_classes)
            df_master_sum.to_csv(master_sum_csv_path, index=False)

    if master_eval_records:
        print("\n==================================================================")
        print(f"Phase 3 Evaluation Complete! Total subject-checkpoint evaluations: {len(master_eval_records):,}")
        print(f"Master Per-Subject CSV : {master_subj_csv_path}")
        print(f"Master Summary CSV     : {master_sum_csv_path}")
        print("==================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 3 Validation/Test-Time Inference & Full-Volume Evaluation for 3D MetaSeg. "
                    "CLI flags default to None and fall back to Stage3Config dataclass defaults."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/vol/miltank/projects/L2R/OASIS_L2R",
        help="Path to the root of the dataset containing the val and test subdirectories."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped run directory (e.g., .../OASIS_METASEG/cf_10/2026-10-21_18-00-00)."
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        choices=["val", "test"],
        help="Dataset split to evaluate (overrides Stage3Config.eval_split, default 'val')."
    )
    parser.add_argument(
        "--max_eval_subjects",
        type=int,
        default=None,
        help="Optional cap on number of validation/test subjects evaluated (-1 evaluates all)."
    )
    parser.add_argument(
        "--specific_stage1_ckpt",
        type=str,
        default=None,
        help="Optional specific Stage 1 checkpoint stem (e.g., stage1_loop_0064) to evaluate alone."
    )
    parser.add_argument(
        "--specific_stage2a_ckpt",
        type=str,
        default=None,
        help="Optional specific Stage 2A checkpoint stem (e.g., stage2a_step_0064) to evaluate alone."
    )
    parser.add_argument(
        "--specific_stage2b_ckpt",
        type=str,
        default=None,
        help="Optional specific Stage 2B checkpoint filename (e.g., stage2b_loop_0064.pth) to evaluate alone."
    )
    parser.add_argument(
        "--sample_prop",
        type=float,
        default=None,
        help="Override voxel sample proportion during test-time fitting (default: inherit from Stage 2A)."
    )
    parser.add_argument(
        "--fit_lr",
        type=float,
        default=None,
        help="Override Adam learning rate during test-time fitting (default: inherit from Stage 2A)."
    )
    parser.add_argument(
        "--recon_chunk_size",
        type=int,
        default=None,
        help="Number of coordinates per chunk during full-volume reconstruction (default 500,000)."
    )
    parser.add_argument(
        "--enable_visualization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save middle-slice PNGs and reconstructed/predicted NIfTI volumes in images/."
    )
    parser.add_argument(
        "--max_vis_subjects",
        type=int,
        default=None,
        help="Number of validation subjects to export PNG/NIfTI visualizations for (-1 = all, default 1)."
    )
    parser.add_argument(
        "--skip_existing_stage1_dirs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Stage 1 directories that already have completed Phase 3 CSV logs."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 3 Evaluation on {device}...")

    dataset_path = Path(args.dataset_path)
    run_dir_path = Path(args.run_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")
    if not run_dir_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")

    cfg_overrides = {
        k: v for k, v in vars(args).items()
        if k in Stage3Config.__dataclass_fields__ and v is not None
    }
    stage3_cfg = Stage3Config(**cfg_overrides)

    run_stage3_evaluation_pipeline(
        dataset_dir=dataset_path,
        run_dir=run_dir_path,
        cfg=stage3_cfg,
        device=device,
        specific_stage1_ckpt=args.specific_stage1_ckpt,
        specific_stage2a_ckpt=args.specific_stage2a_ckpt,
        specific_stage2b_ckpt=args.specific_stage2b_ckpt
    )
