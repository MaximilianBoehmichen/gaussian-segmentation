import argparse
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
class Stage1Config:
    """Configuration hyperparameters for Stage 1 Meta-Learning of MetaSeg."""
    compression_factor: float = 10.0  # Target compression ratio (e.g., 5.0, 10.0, 20.0)
    sample_prop: float = 0.25  # Proportion of voxels (0 < x <= 1) sampled per inner step
    max_train_subjects: int = -1  # Number of training subjects used (-1 uses all available subjects)
    num_classes: int = 5  # aligned_seg4.nii.gz classes: 0 (bg), 1, 2, 3, 4
    seed: int = 42

    # Architecture (hidden_dim is derived automatically from compression_factor)
    in_dim: int = 3
    out_dim: int = 1
    num_hidden_layers: int = 5
    omega_0: float = 30.0

    # Stage 1 Meta-learning loop settings (Powers of 2: 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    max_dataset_loops: int = 1024  # Number of full passes (epochs) over the training set
    min_checkpoint_loop: int = 0  # Minimum power-of-2 loop index at which to start saving .pth files
    inner_steps: int = 2  # T_i = 2 stochastic inner steps per subject
    inner_lr: float = 1e-4  # Adam LR for inner loop
    outer_lr: float = 1e-4  # AdamW LR for outer loop

    # SIREN Weight Regularization
    l2_reg_weight: float = 1e-6  # Gentle L2 penalty on SIREN linear weights in inner loop
    bound_excess_weight: float = 1e-3  # Penalty on weights exceeding (max_weight_bound_mult * init_bound)
    max_weight_bound_mult: float = 3.0  # Allowed multiple of theoretical SIREN initialization bound
    hard_clamp_siren_weights: bool = False  # If True, hard-clamps SIREN weights after each outer step
    outer_weight_decay: float = 1e-5  # Decoupled weight decay for outer AdamW optimizer

    # Loss hyperparameters for 3D volumes
    focal_gamma: float = 3.0
    bg_recon_weight: float = 0.1
    bg_threshold: float = 1e-5

    # Data caching & hardware settings
    clip_percentile: float = 99.95  # Percentile clipping before uint8 quantization (100.0 = min-max)
    pin_cached_memory: bool = True  # Pin uint8 tensors in CPU page-locked memory for fast DMA
    use_amp: bool = True  # Use bfloat16 autocast


@dataclass
class CachedSubjectUint8:
    """Container holding a single subject's full-resolution volume and mask in CPU RAM as uint8."""
    subject_id: str
    spatial_shape: Tuple[int, int, int]
    img_uint8: torch.Tensor  # Shape: (N_voxels, out_dim), dtype: torch.uint8, values in [0, 255]
    seg_uint8: torch.Tensor  # Shape: (N_voxels,), dtype: torch.uint8, values in {0..C-1}


def is_power_of_two_checkpoint(loop_idx: int, min_ckpt_loop: int, max_loops: int) -> bool:
    """
    Determines whether `loop_idx` is a power-of-two checkpointing loop (e.g., 0, 1, 2, 4, 8, 16, ..., 1024)
    or the final loop of training.

    Args:
        loop_idx (int): Current 0-based dataset loop index (0 = pre-training).
        min_ckpt_loop (int): Minimum loop index before saving begins (e.g., 0).
        max_loops (int): Maximum dataset loops configured for the run.

    Returns:
        bool: True if a `.pth` checkpoint should be saved on this loop.
    """
    if loop_idx < min_ckpt_loop:
        return False
    if loop_idx == 0 or loop_idx == 1:
        return True
    is_pow2 = (loop_idx & (loop_idx - 1)) == 0
    return is_pow2 or (loop_idx == max_loops)


def calculate_siren_width_from_compression(
        num_voxels: int,
        in_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        compression_factor: float
) -> Tuple[int, int, int]:
    """
    Calculates the hidden layer width h of the SIREN reconstruction backbone f_theta
    such that its parameter count matches (num_voxels * out_dim) / compression_factor.
    Excludes the segmentation head g_phi from the parameter budget.

    Args:
        num_voxels (int): Total number of spatial voxels in the volume.
        in_dim (int): Input coordinate dimensionality d (e.g., 3).
        out_dim (int): Output signal channels D (e.g., 1).
        num_hidden_layers (int): Number of hidden sine layers L (must be >= 2).
        compression_factor (float): Desired ratio of original volume parameters to f_theta parameters.

    Returns:
        Tuple[int, int, int]:
            - hidden_dim: Calculated integer hidden dimension h.
            - target_params: Target parameter count for f_theta.
            - actual_params: Exact parameter count of f_theta using hidden_dim.
    """
    if num_hidden_layers < 2:
        raise ValueError("num_hidden_layers must be >= 2 to solve quadratic width formula.")
    if compression_factor <= 0.0:
        raise ValueError("compression_factor must be strictly positive.")

    original_params = num_voxels * out_dim
    target_params = max(1, int(round(original_params / compression_factor)))

    a = float(num_hidden_layers - 1)
    b = float(in_dim + num_hidden_layers + out_dim)
    c = float(out_dim - target_params)

    discriminant = b * b - 4.0 * a * c
    raw_h = (-b + math.sqrt(max(0.0, discriminant))) / (2.0 * a)
    hidden_dim = max(16, int(round(raw_h)))

    actual_params = (
            (num_hidden_layers - 1) * (hidden_dim ** 2)
            + (in_dim + num_hidden_layers + out_dim) * hidden_dim
            + out_dim
    )
    return hidden_dim, target_params, actual_params


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
            hidden_dim (int): Hidden layer width h calculated from compression factor.
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

    def get_theta_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns a CPU copy of the state dictionary for f_theta parameters."""
        return {
            "feature_layers": {k: v.detach().cpu().clone() for k, v in self.feature_layers.state_dict().items()},
            "recon_out": {k: v.detach().cpu().clone() for k, v in self.recon_out.state_dict().items()},
        }

    def compute_siren_regularization(
            self,
            l2_weight: float = 1e-6,
            bound_excess_weight: float = 1e-3,
            max_bound_mult: float = 3.0
    ) -> torch.Tensor:
        """
        Computes a two-part regularization loss on the SIREN backbone weights to prevent
        high-frequency blowup over long meta-training runs without collapsing valid frequencies:
          1. Gentle L2 penalty on linear weights (excludes biases).
          2. Quadratic excess penalty on any weight magnitude exceeding `max_bound_mult * init_bound`.

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

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simultaneously predicts per-coordinate intensities and class logits.

        Args:
            coords (torch.Tensor): Normalized coordinates of shape (N, in_dim) in [-1, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pred_intensity: Predicted pixel intensities of shape (N, out_dim).
                - seg_logits: Predicted class logits of shape (N, num_classes).
        """
        features = self.feature_layers(coords)
        pred_intensity = self.recon_out(features)
        seg_logits = self.seg_head(features)
        return pred_intensity, seg_logits


def weighted_reconstruction_loss(
        pred_intensity: torch.Tensor,
        target_intensity: torch.Tensor,
        bg_recon_weight: float = 0.1,
        bg_threshold: float = 1e-5
) -> torch.Tensor:
    """
    Computes per-voxel L2 reconstruction loss with down-weighted background voxels.

    Args:
        pred_intensity (torch.Tensor): Predicted intensities of shape (N, out_dim).
        target_intensity (torch.Tensor): Ground truth intensities of shape (N, out_dim) in [0, 1].
        bg_recon_weight (float): Multiplicative weight applied to background pixels.
        bg_threshold (float): Intensity threshold below which target pixels are treated as background.

    Returns:
        torch.Tensor: Scalar mean weighted MSE loss of shape ().
    """
    sq_err = (pred_intensity.float() - target_intensity.float()).pow(2)
    weights = torch.where(target_intensity <= bg_threshold, bg_recon_weight, 1.0)
    return torch.mean(weights * sq_err)


def multiclass_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 3.0
) -> torch.Tensor:
    """
    Computes multiclass focal classification loss.

    Args:
        logits (torch.Tensor): Predicted unnormalized class logits of shape (N, C).
        targets (torch.Tensor): Ground truth integer class labels of shape (N,) with dtype torch.long.
        gamma (float): Focal loss focusing parameter.

    Returns:
        torch.Tensor: Scalar mean focal loss of shape ().
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    focal_weight = (1.0 - target_probs).pow(gamma)
    return (-focal_weight * target_log_probs).mean()


def discover_split_subjects(split_dir: Path) -> List[Path]:
    """
    Finds all valid subject directories inside a split directory (`train`, `val`, or `test`).

    Args:
        split_dir (Path): Directory path corresponding to a dataset split.

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


def preload_subjects_as_uint8(
        subject_dirs: List[Path],
        clip_percentile: float = 99.95,
        pin_memory: bool = True
) -> List[CachedSubjectUint8]:
    """
    Loads all subjects into CPU RAM as flattened `torch.uint8` tensors to eliminate disk I/O
    and minimize RAM and PCIe transfer overhead.

    Args:
        subject_dirs (List[Path]): List of subject directories to preload.
        clip_percentile (float): Upper intensity percentile clipped prior to [0, 255] scaling.
            Set to 100.0 for strict min-max normalization.
        pin_memory (bool): If True and CUDA is available, pins uint8 tensors in page-locked CPU RAM.

    Returns:
        List[CachedSubjectUint8]: List of preloaded subjects stored in CPU RAM.
    """
    cached_list: List[CachedSubjectUint8] = []
    total_bytes = 0
    use_pin = pin_memory and torch.cuda.is_available()

    print(f"Preloading {len(subject_dirs)} subjects into CPU RAM as uint8...")
    for subj_dir in tqdm.tqdm(subject_dirs, desc="Caching volumes (uint8)"):
        img_path = subj_dir / "aligned_orig.nii.gz"
        seg_path = subj_dir / "aligned_seg4.nii.gz"

        img_np = np.transpose(nib.load(str(img_path)).get_fdata()).astype(np.float32)
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
            CachedSubjectUint8(
                subject_id=subj_dir.name,
                spatial_shape=spatial_shape,
                img_uint8=img_tensor,
                seg_uint8=seg_tensor
            )
        )

    print(f"Cached {len(cached_list)} full-res subjects in CPU RAM ({total_bytes / (1024 ** 2):.2f} MB total).")
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


def flush_metrics_csv(csv_log_path: Path, metrics_history: List[Dict[str, Any]]) -> None:
    """
    Writes the cumulative per-epoch metrics list to disk as a CSV file.

    Args:
        csv_log_path (Path): Destination CSV file path inside the `phase1/` directory.
        metrics_history (List[Dict[str, Any]]): List of per-loop metric dictionaries.
    """
    if metrics_history:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_history).to_csv(csv_log_path, index=False)


def save_stage1_checkpoint(
        save_path: Path,
        latest_path: Path,
        model: MetaSegINR,
        outer_optimizer: torch.optim.Optimizer,
        cfg: Stage1Config,
        run_timestamp: str,
        train_subject_ids: List[str],
        dataset_loops: int,
        outer_steps: int,
        elapsed_opt_time_sec: float,
        recon_loss: float,
        focal_cls_loss: float,
        reg_loss: float,
        fwd_peak_vram_mb: float,
        bwd_peak_vram_mb: float,
        reserved_vram_mb: float,
        spatial_shape: Tuple[int, int, int],
        metrics_history: List[Dict[str, Any]]
) -> None:
    """
    Saves the Stage 1 `.pth` checkpoint (both numbered and latest) including step counts,
    optimization time, VRAM usage, training subject IDs, and losses.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    total_inner_loss = recon_loss + focal_cls_loss + reg_loss

    payload: Dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "dataset_loops": dataset_loops,
        "outer_steps": outer_steps,
        "elapsed_optimization_time_sec": elapsed_opt_time_sec,
        "recon_loss": recon_loss,
        "focal_cls_loss": focal_cls_loss,
        "reg_loss": reg_loss,
        "total_inner_loss": total_inner_loss,
        "fwd_peak_vram_mb": fwd_peak_vram_mb,
        "bwd_peak_vram_mb": bwd_peak_vram_mb,
        "reserved_vram_mb": reserved_vram_mb,
        "compression_factor": cfg.compression_factor,
        "sample_prop": cfg.sample_prop,
        "spatial_shape": spatial_shape,
        "train_subject_ids": train_subject_ids,
        "stage1_config": asdict(cfg),
        "model_config": {
            "in_dim": model.in_dim,
            "out_dim": model.out_dim,
            "num_classes": model.num_classes,
            "num_hidden_layers": model.num_hidden_layers,
            "hidden_dim": model.hidden_dim,
            "omega_0": model.omega_0,
        },
        "model_state_dict": model.state_dict(),
        "theta_state_dict": model.get_theta_state_dict(),
        "outer_optimizer_state_dict": outer_optimizer.state_dict(),
        "metrics_history": metrics_history,
    }

    torch.save(payload, save_path)
    torch.save(payload, latest_path)


def execute_stochastic_inner_steps(
        model: MetaSegINR,
        inner_optimizer: torch.optim.Optimizer,
        coords_flat: torch.Tensor,
        img_u8_gpu: torch.Tensor,
        seg_u8_gpu: torch.Tensor,
        batch_size: int,
        cfg: Stage1Config,
        amp_enabled: bool
) -> Tuple[float, float, float, float, float]:
    """
    Executes `cfg.inner_steps` stochastic gradient updates on a single subject using a random
    voxel batch of size `batch_size = round(num_voxels * sample_prop)` per step, while profiling
    peak VRAM allocated during forward and backward propagation.
    """
    device = coords_flat.device
    is_cuda = device.type == 'cuda'
    num_subj_voxels = coords_flat.shape[0]

    total_recon = 0.0
    total_cls = 0.0
    total_reg = 0.0
    max_fwd_vram_mb = 0.0
    max_bwd_vram_mb = 0.0

    for _ in range(cfg.inner_steps):
        inner_optimizer.zero_grad(set_to_none=True)

        if batch_size < num_subj_voxels:
            idx = torch.randint(0, num_subj_voxels, (batch_size,), device=device)
            b_coords = coords_flat[idx]
            b_img = img_u8_gpu[idx].to(dtype=torch.float32) * (1.0 / 255.0)
            b_seg = seg_u8_gpu[idx].to(dtype=torch.long)
        else:
            b_coords = coords_flat
            b_img = img_u8_gpu.to(dtype=torch.float32) * (1.0 / 255.0)
            b_seg = seg_u8_gpu.to(dtype=torch.long)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            pred_img, seg_logits = model(b_coords)
            l_recon = weighted_reconstruction_loss(
                pred_img, b_img, cfg.bg_recon_weight, cfg.bg_threshold
            )
            l_cls = multiclass_focal_loss(seg_logits, b_seg, gamma=cfg.focal_gamma)
            l_reg = model.compute_siren_regularization(
                l2_weight=cfg.l2_reg_weight,
                bound_excess_weight=cfg.bound_excess_weight,
                max_bound_mult=cfg.max_weight_bound_mult
            )
            l_inner = l_recon + l_cls + l_reg

        if is_cuda:
            fwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            if fwd_mb > max_fwd_vram_mb:
                max_fwd_vram_mb = fwd_mb
            torch.cuda.reset_peak_memory_stats(device)

        l_inner.backward()
        inner_optimizer.step()

        if is_cuda:
            bwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            if bwd_mb > max_bwd_vram_mb:
                max_bwd_vram_mb = bwd_mb

        total_recon += l_recon.item()
        total_cls += l_cls.item()
        total_reg += l_reg.item()

    inv_steps = 1.0 / float(cfg.inner_steps)
    return (
        total_recon * inv_steps,
        total_cls * inv_steps,
        total_reg * inv_steps,
        max_fwd_vram_mb,
        max_bwd_vram_mb,
    )


def run_stage1_training(
        dataset_dir: Path,
        output_dir: Path,
        cfg: Stage1Config,
        device: torch.device,
        resume_from: Optional[Path] = None
) -> None:
    """
    Executes Stage 1 Meta-Learning inside `<output_dir>/cf_<X>/<timestamp>/phase1/`,
    logging metrics and VRAM usage to CSV every epoch and saving `.pth` checkpoints at
    powers of two (0, 1, 2, 4, 8, 16, ..., 1024).
    """
    if not (0.0 < cfg.sample_prop <= 1.0):
        raise ValueError(f"sample_prop must satisfy 0 < sample_prop <= 1.0, got {cfg.sample_prop}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_subjects = discover_split_subjects(dataset_dir / "train")
    if not train_subjects:
        raise FileNotFoundError(f"No valid training subjects found in {dataset_dir / 'train'}")

    if cfg.max_train_subjects > 0:
        train_subjects = train_subjects[:cfg.max_train_subjects]
        print(f"Capped training subjects to {len(train_subjects)} via max_train_subjects.")

    cached_train = preload_subjects_as_uint8(
        train_subjects,
        clip_percentile=cfg.clip_percentile,
        pin_memory=cfg.pin_cached_memory
    )
    train_subject_ids = [s.subject_id for s in cached_train]

    spatial_shape = cached_train[0].spatial_shape
    num_voxels = int(np.prod(spatial_shape))
    batch_size = max(1, int(round(num_voxels * cfg.sample_prop)))

    hidden_dim, target_theta_params, actual_theta_params = calculate_siren_width_from_compression(
        num_voxels=num_voxels,
        in_dim=cfg.in_dim,
        out_dim=cfg.out_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        compression_factor=cfg.compression_factor
    )

    model = MetaSegINR(
        in_dim=cfg.in_dim,
        out_dim=cfg.out_dim,
        num_classes=cfg.num_classes,
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_dim=hidden_dim,
        omega_0=cfg.omega_0
    ).to(device)

    outer_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.outer_lr,
        weight_decay=cfg.outer_weight_decay
    )

    start_loop = 1
    outer_steps = 0
    cumulative_opt_time = 0.0
    metrics_history: List[Dict[str, Any]] = []

    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        outer_optimizer.load_state_dict(ckpt["outer_optimizer_state_dict"])
        start_loop = int(ckpt["dataset_loops"]) + 1
        outer_steps = int(ckpt["outer_steps"])
        cumulative_opt_time = float(ckpt["elapsed_optimization_time_sec"])
        metrics_history = list(ckpt.get("metrics_history", []))
        print(f"Resumed from {resume_from} at loop {start_loop} "
              f"(Outer steps: {outer_steps}, Opt time: {cumulative_opt_time:.2f}s)")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"cf_{cfg.compression_factor:g}" / run_timestamp
    phase1_dir = run_dir / "phase1"
    ckpt_dir = phase1_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_file_path = phase1_dir / "stage1_config.yml"
    config_dict = asdict(cfg)
    config_dict["run_timestamp"] = run_timestamp
    config_dict["hidden_dim"] = hidden_dim
    config_dict["actual_theta_params"] = actual_theta_params
    config_dict["spatial_shape"] = list(spatial_shape)
    config_dict["train_subject_ids"] = train_subject_ids
    with open(config_file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    latest_path = phase1_dir / "stage1_latest.pth"
    csv_log_path = phase1_dir / "stage1_metrics_log.csv"

    scheduled_ckpts = [
        loop for loop in range(0, cfg.max_dataset_loops + 1)
        if is_power_of_two_checkpoint(loop, cfg.min_checkpoint_loop, cfg.max_dataset_loops)
    ]

    print("==================================================================")
    print(f"Stage 1 Meta-Learning Configuration (Stochastic Batch Inner Steps)")
    print(f"Run Timestamp              : {run_timestamp}")
    print(f"Run Root Directory         : {run_dir}")
    print(f"Phase 1 Directory          : {phase1_dir}")
    print(f"Checkpoints Directory      : {ckpt_dir}")
    print(f"Metrics CSV Path           : {csv_log_path}")
    print(f"Config YAML Path           : {config_file_path}")
    print(f"Volume Spatial Shape       : {spatial_shape} ({num_voxels:,} voxels)")
    print(f"Active Training Subjects   : {len(cached_train)}")
    print(
        f"Stochastic Batch Proportion: {cfg.sample_prop:.4f} -> {batch_size:,} voxels/step ({cfg.inner_steps} steps/subj)")
    print(f"Compression Factor         : {cfg.compression_factor}x -> SIREN Width h = {hidden_dim}")
    print(f"f_theta Parameters         : {actual_theta_params:,} (Target: {target_theta_params:,})")
    print(f"SIREN Regularization       : L2={cfg.l2_reg_weight:.1e}, BoundExcess={cfg.bound_excess_weight:.1e} "
          f"(MaxMult={cfg.max_weight_bound_mult}x, HardClamp={cfg.hard_clamp_siren_weights})")
    print(f"Target Dataset Loops       : {cfg.max_dataset_loops}")
    print(f"Scheduled Checkpoints      : {scheduled_ckpts}")
    print("==================================================================")

    coord_cache = CoordinateGridCache(device=device)
    amp_enabled = cfg.use_amp and device.type == 'cuda'

    model_params: List[nn.Parameter] = list(model.parameters())
    outer_snapshot_buffers: List[torch.Tensor] = [p.data.clone() for p in model_params]

    def _handle_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    completed_loops = start_loop - 1
    last_mean_recon = 0.0
    last_mean_cls = 0.0
    last_mean_reg = 0.0
    loop_fwd_vram_mb = 0.0
    loop_bwd_vram_mb = 0.0
    loop_reserved_vram_mb = 0.0

    try:
        # Pre-training checkpoint block (Loop 0)
        if start_loop == 1 and is_power_of_two_checkpoint(0, cfg.min_checkpoint_loop, cfg.max_dataset_loops):
            ckpt_filename = "stage1_loop_0000.pth"
            ckpt_path = ckpt_dir / ckpt_filename

            metrics_history.append({
                "run_timestamp": run_timestamp,
                "dataset_loops": 0,
                "outer_steps": 0,
                "elapsed_optimization_time_sec": 0.0,
                "loop_time_sec": 0.0,
                "recon_loss": 0.0,
                "focal_cls_loss": 0.0,
                "reg_loss": 0.0,
                "total_inner_loss": 0.0,
                "fwd_peak_vram_mb": 0.0,
                "bwd_peak_vram_mb": 0.0,
                "reserved_vram_mb": 0.0,
                "compression_factor": cfg.compression_factor,
                "hidden_dim": hidden_dim,
                "sample_prop": cfg.sample_prop,
                "checkpoint_file": ckpt_filename,
            })
            flush_metrics_csv(csv_log_path, metrics_history)

            save_stage1_checkpoint(
                save_path=ckpt_path,
                latest_path=latest_path,
                model=model,
                outer_optimizer=outer_optimizer,
                cfg=cfg,
                run_timestamp=run_timestamp,
                train_subject_ids=train_subject_ids,
                dataset_loops=0,
                outer_steps=0,
                elapsed_opt_time_sec=0.0,
                recon_loss=0.0,
                focal_cls_loss=0.0,
                reg_loss=0.0,
                fwd_peak_vram_mb=0.0,
                bwd_peak_vram_mb=0.0,
                reserved_vram_mb=0.0,
                spatial_shape=spatial_shape,
                metrics_history=metrics_history
            )
            print(f"   -> [Power-of-2 Checkpoint] Saved {ckpt_filename} before training.")

        # Main training loop
        for loop_idx in range(start_loop, cfg.max_dataset_loops + 1):
            t_loop_start = time.perf_counter()
            random.shuffle(cached_train)

            loop_recon_loss = 0.0
            loop_cls_loss = 0.0
            loop_reg_loss = 0.0
            loop_fwd_vram_mb = 0.0
            loop_bwd_vram_mb = 0.0

            for subj in tqdm.tqdm(
                    cached_train,
                    desc=f"Stage 1 Loop {loop_idx}/{cfg.max_dataset_loops}",
                    leave=False
            ):
                coords_flat = coord_cache.get_coords(subj.spatial_shape)
                cur_batch_size = min(batch_size, coords_flat.shape[0])

                img_u8_gpu = subj.img_uint8.to(device, non_blocking=True)
                seg_u8_gpu = subj.seg_uint8.to(device, non_blocking=True)

                torch._foreach_copy_(outer_snapshot_buffers, [p.data for p in model_params])

                inner_optimizer = torch.optim.Adam(model_params, lr=cfg.inner_lr)
                model.train()

                mean_recon, mean_cls, mean_reg, subj_fwd_mb, subj_bwd_mb = execute_stochastic_inner_steps(
                    model=model,
                    inner_optimizer=inner_optimizer,
                    coords_flat=coords_flat,
                    img_u8_gpu=img_u8_gpu,
                    seg_u8_gpu=seg_u8_gpu,
                    batch_size=cur_batch_size,
                    cfg=cfg,
                    amp_enabled=amp_enabled
                )

                loop_recon_loss += mean_recon
                loop_cls_loss += mean_cls
                loop_reg_loss += mean_reg
                if subj_fwd_mb > loop_fwd_vram_mb:
                    loop_fwd_vram_mb = subj_fwd_mb
                if subj_bwd_mb > loop_bwd_vram_mb:
                    loop_bwd_vram_mb = subj_bwd_mb

                outer_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    pseudo_grads = torch._foreach_sub(outer_snapshot_buffers, [p.data for p in model_params])
                    torch._foreach_copy_([p.data for p in model_params], outer_snapshot_buffers)
                    for p, g in zip(model_params, pseudo_grads):
                        p.grad = g

                outer_optimizer.step()

                if cfg.hard_clamp_siren_weights:
                    model.clamp_siren_weights(max_bound_mult=cfg.max_weight_bound_mult)

                outer_steps += 1

            if device.type == 'cuda':
                torch.cuda.synchronize()
                loop_reserved_vram_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)

            loop_elapsed = time.perf_counter() - t_loop_start
            cumulative_opt_time += loop_elapsed
            completed_loops = loop_idx

            num_train = len(cached_train)
            last_mean_recon = loop_recon_loss / num_train
            last_mean_cls = loop_cls_loss / num_train
            last_mean_reg = loop_reg_loss / num_train
            last_total_inner = last_mean_recon + last_mean_cls + last_mean_reg

            is_ckpt_loop = is_power_of_two_checkpoint(
                loop_idx, cfg.min_checkpoint_loop, cfg.max_dataset_loops
            )
            ckpt_filename = f"stage1_loop_{loop_idx:04d}.pth" if is_ckpt_loop else ""

            metrics_history.append({
                "run_timestamp": run_timestamp,
                "dataset_loops": completed_loops,
                "outer_steps": outer_steps,
                "elapsed_optimization_time_sec": round(cumulative_opt_time, 4),
                "loop_time_sec": round(loop_elapsed, 4),
                "recon_loss": last_mean_recon,
                "focal_cls_loss": last_mean_cls,
                "reg_loss": last_mean_reg,
                "total_inner_loss": last_total_inner,
                "fwd_peak_vram_mb": round(loop_fwd_vram_mb, 2),
                "bwd_peak_vram_mb": round(loop_bwd_vram_mb, 2),
                "reserved_vram_mb": round(loop_reserved_vram_mb, 2),
                "compression_factor": cfg.compression_factor,
                "hidden_dim": hidden_dim,
                "sample_prop": cfg.sample_prop,
                "checkpoint_file": ckpt_filename,
            })

            flush_metrics_csv(csv_log_path, metrics_history)

            print(
                f"Loop {loop_idx:04d}/{cfg.max_dataset_loops} | "
                f"Outer Steps: {outer_steps:,} | "
                f"Opt Time: {cumulative_opt_time:.1f}s (+{loop_elapsed:.2f}s) | "
                f"Recon: {last_mean_recon:.4f} | "
                f"Focal: {last_mean_cls:.4f} | "
                f"Reg: {last_mean_reg:.2e} | "
                f"VRAM (Fwd/Bwd): {loop_fwd_vram_mb:.0f}/{loop_bwd_vram_mb:.0f} MB"
            )

            if is_ckpt_loop:
                ckpt_path = ckpt_dir / ckpt_filename
                save_stage1_checkpoint(
                    save_path=ckpt_path,
                    latest_path=latest_path,
                    model=model,
                    outer_optimizer=outer_optimizer,
                    cfg=cfg,
                    run_timestamp=run_timestamp,
                    train_subject_ids=train_subject_ids,
                    dataset_loops=completed_loops,
                    outer_steps=outer_steps,
                    elapsed_opt_time_sec=cumulative_opt_time,
                    recon_loss=last_mean_recon,
                    focal_cls_loss=last_mean_cls,
                    reg_loss=last_mean_reg,
                    fwd_peak_vram_mb=loop_fwd_vram_mb,
                    bwd_peak_vram_mb=loop_bwd_vram_mb,
                    reserved_vram_mb=loop_reserved_vram_mb,
                    spatial_shape=spatial_shape,
                    metrics_history=metrics_history
                )
                print(
                    f"    -> [Power-of-2 Checkpoint] Saved {ckpt_path.name} "
                    f"(Cumulative Opt Time: {cumulative_opt_time:.1f}s)"
                )

    except KeyboardInterrupt:
        print(f"\nInterrupted at loop {completed_loops}. Saving emergency checkpoint and CSV...")
        interrupted_filename = f"stage1_interrupted_loop_{completed_loops:04d}_steps_{outer_steps}.pth"
        interrupted_path = ckpt_dir / interrupted_filename

        if metrics_history and metrics_history[-1]["dataset_loops"] == completed_loops:
            metrics_history[-1]["checkpoint_file"] = interrupted_filename

        flush_metrics_csv(csv_log_path, metrics_history)
        save_stage1_checkpoint(
            save_path=interrupted_path,
            latest_path=latest_path,
            model=model,
            outer_optimizer=outer_optimizer,
            cfg=cfg,
            run_timestamp=run_timestamp,
            train_subject_ids=train_subject_ids,
            dataset_loops=completed_loops,
            outer_steps=outer_steps,
            elapsed_opt_time_sec=cumulative_opt_time,
            recon_loss=last_mean_recon,
            focal_cls_loss=last_mean_cls,
            reg_loss=last_mean_reg,
            fwd_peak_vram_mb=loop_fwd_vram_mb,
            bwd_peak_vram_mb=loop_bwd_vram_mb,
            reserved_vram_mb=loop_reserved_vram_mb,
            spatial_shape=spatial_shape,
            metrics_history=metrics_history
        )
        print(f"Saved interrupted checkpoint to: {interrupted_path}")
        print(f"Saved metrics CSV to           : {csv_log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stage 1 Meta-Learning for Full-Resolution 3D MetaSeg. "
                    "CLI flags default to None and fall back to Stage1Config dataclass defaults."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/vol/miltank/projects/L2R/OASIS_L2R",
        help="Path to the root of the dataset containing the train subdirectory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/vol/miltank/projects/L2R/projects/OASIS_METASEG",
        help="Base output path where compression-specific and datetime-stamped folders will be saved."
    )
    parser.add_argument(
        "--compression_factor",
        type=float,
        default=None,
        help="Compression factor relative to total voxels (overrides Stage1Config.compression_factor)."
    )
    parser.add_argument(
        "--sample_prop",
        type=float,
        default=None,
        help="Proportion of voxels (0 < sample_prop <= 1) sampled per inner step (overrides Stage1Config.sample_prop)."
    )
    parser.add_argument(
        "--max_train_subjects",
        type=int,
        default=None,
        help="Limit on training subjects loaded (-1 uses all available subjects, overrides Stage1Config.max_train_subjects)."
    )
    parser.add_argument(
        "--max_dataset_loops",
        type=int,
        default=None,
        help="Total number of training set loops to run (overrides Stage1Config.max_dataset_loops)."
    )
    parser.add_argument(
        "--min_checkpoint_loop",
        type=int,
        default=None,
        help="Minimum power-of-2 loop index at which to start saving checkpoints (overrides Stage1Config.min_checkpoint_loop)."
    )
    parser.add_argument(
        "--l2_reg_weight",
        type=float,
        default=None,
        help="L2 weight regularization coefficient (overrides Stage1Config.l2_reg_weight)."
    )
    parser.add_argument(
        "--bound_excess_weight",
        type=float,
        default=None,
        help="Bound-excess regularization weight (overrides Stage1Config.bound_excess_weight)."
    )
    parser.add_argument(
        "--max_weight_bound_mult",
        type=float,
        default=None,
        help="Allowed multiple of SIREN initialization bound (overrides Stage1Config.max_weight_bound_mult)."
    )
    parser.add_argument(
        "--hard_clamp_siren_weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hard-clamp SIREN weights (overrides Stage1Config.hard_clamp_siren_weights)."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional path to a Stage 1 .pth checkpoint to resume training."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Stage 1 Meta-Learning on {device}...")

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")

    cfg_overrides = {
        k: v for k, v in vars(args).items()
        if k in Stage1Config.__dataclass_fields__ and v is not None
    }
    stage1_cfg = Stage1Config(**cfg_overrides)

    resume_ckpt = Path(args.resume_from) if args.resume_from is not None else None
    run_stage1_training(dataset_path, output_path, stage1_cfg, device, resume_from=resume_ckpt)