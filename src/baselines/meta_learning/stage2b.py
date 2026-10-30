import argparse
import copy
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
class Stage2BConfig:
    """Configuration hyperparameters for Phase 2B (Global Segmentation Head Optimization)."""
    sample_prop: float = 0.25                    # Proportion of voxels (0 < x <= 1) sampled per subject step
    max_head_loops: int = 128                   # Total passes over the N fitted training subjects (powers of 2: 2..128)
    min_checkpoint_loop: int = 0                # Minimum power-of-2 loop at which to save a Stage 2B checkpoint
    head_lr: float = 5e-5                       # Adam learning rate for tuning g_phi (5e-5 for 5 classes in paper)
    head_weight_decay: float = 0.0              # Optional weight decay on segmentation head g_phi
    focal_gamma: float = 3.0                    # Focal loss exponent (3.0 for 3D volumes in paper)
    num_classes: int = 5                        # aligned_seg4.nii.gz classes: 0 (bg), 1, 2, 3, 4
    seed: int = 42

    # Visualization & Full-Volume Evaluation at Power-of-2 Checkpoints
    enable_visualization: bool = False          # If True, saves mid-slice seg PNG and NIfTI label volume in images/
    max_vis_subjects: int = 1                   # Number of subjects to export PNG/NIfTI for (-1 exports all subjects)
    eval_full_dice_at_ckpts: bool = False       # If True, evaluates full-volume Dice across eval_full_max_subjects
    eval_full_max_subjects: int = 5             # Number of training subjects used for full-volume Dice check (-1 = all)
    infer_chunk_size: int = 500_000             # Voxel chunk size for full-volume segmentation inference

    # Hardware & execution settings
    pin_cached_memory: bool = True              # Pin uint8 segmentation tensors in page-locked CPU RAM
    use_amp: bool = True                        # Use bfloat16 autocast
    skip_existing_checkpoints: bool = True      # Skip Stage 2A checkpoints whose Phase 2B outputs are already complete


@dataclass
class CachedSegUint8:
    """Container holding a single subject's full-resolution segmentation mask in CPU RAM as uint8."""
    subject_id: str
    spatial_shape: Tuple[int, int, int]
    affine: np.ndarray        # Original (4, 4) NIfTI affine matrix for accurate volume export
    seg_uint8: torch.Tensor   # Shape: (N_voxels,), dtype: torch.uint8, values in {0..C-1}


def is_power_of_two_loop(loop_idx: int, min_ckpt_loop: int, max_loops: int) -> bool:
    """
    Determines whether `loop_idx` is a power-of-two checkpoint loop (e.g., 2, 4, 8, 16, 32, 64, 128)
    or the final loop of Phase 2B.

    Args:
        loop_idx (int): Current 1-based dataset loop index.
        min_ckpt_loop (int): Minimum loop index before power-of-two saving begins (e.g., 2).
        max_loops (int): Maximum head loops configured for Phase 2B.

    Returns:
        bool: True if a `.pth` checkpoint should be saved at this loop.
    """
    is_pow2 = (loop_idx >= min_ckpt_loop) and ((loop_idx & (loop_idx - 1)) == 0)
    return is_pow2 or (loop_idx == max_loops)


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

    def get_phi_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns a CPU copy of the state dictionary for g_phi (`seg_head`) parameters."""
        return {k: v.detach().cpu().clone() for k, v in self.seg_head.state_dict().items()}

    def load_theta_state_dict(self, theta_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Loads f_theta parameters from a saved theta state dictionary.

        Args:
            theta_state (Dict[str, Dict[str, torch.Tensor]]): State dictionary containing
                'feature_layers' and 'recon_out' weight tensors.
        """
        self.feature_layers.load_state_dict(theta_state["feature_layers"])
        self.recon_out.load_state_dict(theta_state["recon_out"])

    def forward_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes penultimate features f_theta^{L-1}(x).

        Args:
            coords (torch.Tensor): Normalized coordinates of shape (N, in_dim) in [-1, 1].

        Returns:
            torch.Tensor: Penultimate features of shape (N, hidden_dim).
        """
        return self.feature_layers(coords)

    def forward_seg_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts segmentation class logits from penultimate features.

        Args:
            features (torch.Tensor): Penultimate features of shape (N, hidden_dim).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (N, num_classes).
        """
        return self.seg_head(features)


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 3.0
) -> torch.Tensor:
    """
    Computes multiclass focal classification loss as defined in Eq. (2) of MetaSeg.

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


@torch.no_grad()
def compute_macro_dice_scores(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int = 5
) -> Tuple[float, float]:
    """
    Computes macro-averaged Dice similarity coefficients both across all classes (0..C-1)
    and across foreground-only classes (1..C-1).

    Args:
        pred_labels (torch.Tensor): Predicted integer labels of shape (N,).
        true_labels (torch.Tensor): Ground truth integer labels of shape (N,).
        num_classes (int): Total number of segmentation classes C.

    Returns:
        Tuple[float, float]:
            - dice_all: Macro-averaged Dice across all classes (including background 0).
            - dice_fg: Macro-averaged Dice across foreground classes (1..C-1).
    """
    dices: List[float] = []
    eps = 1e-6

    for c in range(num_classes):
        pred_c = (pred_labels == c)
        true_c = (true_labels == c)
        intersection = float((pred_c & true_c).sum().item())
        cardinality = float(pred_c.sum().item() + true_c.sum().item())
        if cardinality == 0.0:
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection + eps) / (cardinality + eps))

    dice_all = float(np.mean(dices))
    dice_fg = float(np.mean(dices[1:])) if num_classes > 1 else dice_all
    return dice_all, dice_fg


@torch.no_grad()
def predict_full_volume_segmentation_and_dice(
    model: MetaSegINR,
    coords_flat: torch.Tensor,
    seg_u8_gpu: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    num_classes: int,
    amp_enabled: bool,
    chunk_size: int = 500_000,
    return_volume: bool = False
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Evaluates the frozen subject backbone `f_{theta_j^*}` and current segmentation head `g_phi`
    across 100% of a volume's coordinates in chunks to compute exact full-volume Dice scores,
    and optionally returns the 3D predicted label volume.

    Args:
        model (MetaSegINR): Active MetaSegINR model on GPU with `theta_j^*` loaded.
        coords_flat (torch.Tensor): Flattened coordinate grid of shape (N_voxels, D) in [-1, 1].
        seg_u8_gpu (torch.Tensor): Ground-truth uint8 segmentation labels on GPU of shape (N_voxels,).
        spatial_shape (Tuple[int, int, int]): 3D spatial shape (Z, Y, X) of the volume.
        num_classes (int): Number of segmentation classes C.
        amp_enabled (bool): Whether bfloat16 autocast is active.
        chunk_size (int): Number of coordinates evaluated per forward chunk.
        return_volume (bool): If True, materializes and returns the 3D uint8 predicted label array.

    Returns:
        Tuple[float, float, Optional[np.ndarray]]:
            - full_dice_all: Exact full-volume macro Dice across all classes (0..C-1).
            - full_dice_fg: Exact full-volume macro Dice across foreground classes (1..C-1).
            - pred_vol_np: 3D uint8 array of shape (Z, Y, X) if `return_volume` is True, else None.
    """
    model.eval()
    num_voxels = coords_flat.shape[0]
    pred_chunks: List[torch.Tensor] = []

    intersections = np.zeros(num_classes, dtype=np.float64)
    cardinalities = np.zeros(num_classes, dtype=np.float64)

    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        b_coords = coords_flat[start:end]
        b_true = seg_u8_gpu[start:end]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            feats = model.forward_features(b_coords)
            logits = model.forward_seg_from_features(feats)
            pred_cls = torch.argmax(logits, dim=-1).to(dtype=torch.uint8)

        for c in range(num_classes):
            pred_c = (pred_cls == c)
            true_c = (b_true == c)
            intersections[c] += float((pred_c & true_c).sum().item())
            cardinalities[c] += float(pred_c.sum().item() + true_c.sum().item())

        if return_volume:
            pred_chunks.append(pred_cls.cpu())

    model.train()
    eps = 1e-6
    dices = [
        1.0 if cardinalities[c] == 0.0 else (2.0 * intersections[c] + eps) / (cardinalities[c] + eps)
        for c in range(num_classes)
    ]
    full_dice_all = float(np.mean(dices))
    full_dice_fg = float(np.mean(dices[1:])) if num_classes > 1 else full_dice_all

    pred_vol_np: Optional[np.ndarray] = None
    if return_volume:
        pred_vol_np = torch.cat(pred_chunks, dim=0).view(*spatial_shape).numpy().astype(np.uint8)

    return full_dice_all, full_dice_fg, pred_vol_np


def save_seg_debug_visualization_and_volume(
    pred_vol_np: np.ndarray,
    gt_seg_uint8_cpu: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    affine: np.ndarray,
    log_dir: Path,
    subject_id: str,
    loop_idx: int,
    num_classes: int,
    full_dice_all: float,
    full_dice_fg: float
) -> Tuple[Path, Path]:
    """
    Saves the predicted 3D segmentation mask as a `.nii.gz` file and the orthogonal middle slices
    (Ground Truth vs. Predicted Segmentation) as a `.png` image inside `log_dir`.

    Args:
        pred_vol_np (np.ndarray): Predicted 3D label array of shape (Z, Y, X) with values in {0..C-1}.
        gt_seg_uint8_cpu (torch.Tensor): Ground truth flattened uint8 tensor of shape (N_voxels,) on CPU.
        spatial_shape (Tuple[int, int, int]): Volume spatial dimensions (Z, Y, X).
        affine (np.ndarray): Original (4, 4) NIfTI affine matrix.
        log_dir (Path): Target directory where `.png` and `.nii.gz` files are saved.
        subject_id (str): Identifier of the subject.
        loop_idx (int): Current Phase 2B head optimization loop index.
        num_classes (int): Number of segmentation classes C.
        full_dice_all (float): Full-volume macro Dice across all classes.
        full_dice_fg (float): Full-volume macro Dice across foreground classes.

    Returns:
        Tuple[Path, Path]:
            - slice_png_path: Path to the saved middle-slice PNG image.
            - nifti_path: Path to the saved NIfTI segmentation volume.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"stage2b_loop_{loop_idx:04d}_{subject_id}"
    slice_png_path = log_dir / f"{prefix}_seg_mid_slice.png"
    nifti_path = log_dir / f"{prefix}_pred_seg4.nii.gz"

    # 1. Save Predicted Segmentation Volume as NIfTI (reverse transpose applied during loading)
    pred_nifti_data = np.transpose(pred_vol_np)
    nib.save(nib.Nifti1Image(pred_nifti_data, affine), str(nifti_path))

    # 2. Save Middle Slice Image (Ground Truth vs. Predicted Segmentation across mid-planes)
    gt_vol_np = gt_seg_uint8_cpu.view(*spatial_shape).numpy().astype(np.uint8)
    z_mid, y_mid, x_mid = spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2

    gt_slices = [gt_vol_np[z_mid, :, :], gt_vol_np[:, y_mid, :], gt_vol_np[:, :, x_mid]]
    pred_slices = [pred_vol_np[z_mid, :, :], pred_vol_np[:, y_mid, :], pred_vol_np[:, :, x_mid]]
    plane_titles = [f"Mid-Plane 0 (idx={z_mid})", f"Mid-Plane 1 (idx={y_mid})", f"Mid-Plane 2 (idx={x_mid})"]

    cmap = plt.get_cmap("tab10", num_classes)
    vmax_cls = max(1, num_classes - 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Phase 2B Segmentation | Subject: {subject_id} | Loop: {loop_idx} | "
        f"Dice (All): {full_dice_all:.4f} | Dice (FG): {full_dice_fg:.4f}",
        fontsize=13
    )

    for col in range(3):
        axes[0, col].imshow(gt_slices[col], cmap=cmap, vmin=0, vmax=vmax_cls, interpolation="nearest", origin="lower")
        axes[0, col].set_title(f"GT Seg - {plane_titles[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(pred_slices[col], cmap=cmap, vmin=0, vmax=vmax_cls, interpolation="nearest", origin="lower")
        axes[1, col].set_title(f"Pred Seg - {plane_titles[col]}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(slice_png_path, dpi=150)
    plt.close(fig)

    return slice_png_path, nifti_path


def preload_segmentations_as_uint8(
    dataset_dir: Path,
    subject_ids: List[str],
    pin_memory: bool = True
) -> Dict[str, CachedSegUint8]:
    """
    Loads all training subjects' segmentation volumes (`aligned_seg4.nii.gz`) into CPU RAM
    as flattened `torch.uint8` tensors along with their original NIfTI affine matrices.

    Args:
        dataset_dir (Path): Root dataset path containing the `train` subdirectory.
        subject_ids (List[str]): Ordered list of training subject IDs from Stage 1 / 2A.
        pin_memory (bool): If True and CUDA is available, pins uint8 tensors in page-locked CPU RAM.

    Returns:
        Dict[str, CachedSegUint8]: Mapping from `subject_id` to its preloaded segmentation mask.
    """
    train_root = dataset_dir / "train"
    cached_dict: Dict[str, CachedSegUint8] = {}
    total_bytes = 0
    use_pin = pin_memory and torch.cuda.is_available()

    print(f"Preloading {len(subject_ids)} segmentation masks into CPU RAM as uint8...")
    for sid in tqdm.tqdm(subject_ids, desc="Caching segmentations (uint8)"):
        seg_path = train_root / sid / "aligned_seg4.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found for subject '{sid}': {seg_path}")

        nii_obj = nib.load(str(seg_path))
        affine = nii_obj.affine.copy()

        seg_np = np.transpose(nii_obj.get_fdata()).astype(np.uint8)
        seg_np = np.squeeze(seg_np)
        spatial_shape: Tuple[int, int, int] = (int(seg_np.shape[0]), int(seg_np.shape[1]), int(seg_np.shape[2]))

        seg_tensor = torch.from_numpy(seg_np).reshape(-1).contiguous()
        if use_pin:
            seg_tensor = seg_tensor.pin_memory()

        total_bytes += seg_tensor.numel() * seg_tensor.element_size()
        cached_dict[sid] = CachedSegUint8(
            subject_id=sid,
            spatial_shape=spatial_shape,
            affine=affine,
            seg_uint8=seg_tensor
        )

    print(f"Cached {len(cached_dict)} full-res masks in CPU RAM ({total_bytes / (1024 ** 2):.2f} MB total).")
    return cached_dict


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
    Writes the cumulative per-loop metrics list to disk as a CSV file.

    Args:
        csv_log_path (Path): Destination CSV file path.
        metrics_history (List[Dict[str, Any]]): List of per-loop metric dictionaries.
    """
    if metrics_history:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_history).to_csv(csv_log_path, index=False)


def process_single_stage2a_checkpoint(
    s2a_ckpt_path: Path,
    s2a_out_dir: Path,
    cached_segs: Dict[str, CachedSegUint8],
    coord_cache: CoordinateGridCache,
    cfg: Stage2BConfig,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Executes Phase 2B (Global Segmentation Head Optimization) for a single Stage 2A checkpoint
    over `cfg.max_head_loops` dataset passes, logging metrics and VRAM at every loop and saving
    lightweight deployable `[theta^*, phi^*]` checkpoints at power-of-two loops.

    Args:
        s2a_ckpt_path (Path): Path to the source `stage2a_step_XXXX.pth` checkpoint.
        s2a_out_dir (Path): Parent Stage 1 folder inside `phase2/` (`<run_dir>/phase2/<s1_stem>`).
        cached_segs (Dict[str, CachedSegUint8]): Preloaded uint8 training segmentation masks.
        coord_cache (CoordinateGridCache): GPU coordinate grid cache.
        cfg (Stage2BConfig): Phase 2B configuration parameters.
        device (torch.device): Target compute device.

    Returns:
        List[Dict[str, Any]]: Summary rows for the power-of-two checkpoints saved during this run.
    """
    s2a_stem = s2a_ckpt_path.stem
    s1_stem = s2a_out_dir.name
    s2b_run_dir = s2a_out_dir / "stage2b" / s2a_stem
    s2b_ckpt_dir = s2b_run_dir / "checkpoints"
    s2b_img_dir = s2b_run_dir / "images"
    csv_log_path = s2b_run_dir / "stage2b_metrics_log.csv"

    checkpoint_loops = [
        l_idx for l_idx in range(1, cfg.max_head_loops + 1)
        if is_power_of_two_loop(l_idx, cfg.min_checkpoint_loop, cfg.max_head_loops)
    ]
    ckpt_loop_set = set(checkpoint_loops)

    if cfg.skip_existing_checkpoints and csv_log_path.exists() and not cfg.enable_visualization:
        all_exist = all((s2b_ckpt_dir / f"stage2b_loop_{l_idx:04d}.pth").exists() for l_idx in checkpoint_loops)
        if all_exist:
            print(f"Skipping {s1_stem}/{s2a_stem}: all Phase 2B checkpoints already exist in {s2b_ckpt_dir}")
            df_existing = pd.read_csv(csv_log_path)
            ckpt_rows = df_existing[df_existing["checkpoint_file"].notna() & (df_existing["checkpoint_file"] != "")]
            return ckpt_rows.to_dict(orient="records")

    s2b_ckpt_dir.mkdir(parents=True, exist_ok=True)
    s2a_data = torch.load(s2a_ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = s2a_data["model_config"]
    model = MetaSegINR(
        in_dim=int(model_cfg["in_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        num_classes=int(model_cfg["num_classes"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        omega_0=float(model_cfg["omega_0"])
    ).to(device)

    # Initialize model from Stage 1 state so g_phi starts at phi^{T_o}
    model.load_state_dict(s2a_data["stage1_model_state_dict"])
    stage1_theta_state = s2a_data["stage1_theta_state_dict"]
    subject_thetas: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = s2a_data["subject_thetas"]
    active_subject_ids = [sid for sid in s2a_data["train_subject_ids"] if sid in subject_thetas and sid in cached_segs]

    # Freeze backbone f_theta completely; optimize only segmentation head g_phi
    for p in model.get_theta_parameters():
        p.requires_grad_(False)
    phi_params = model.get_phi_parameters()
    for p in phi_params:
        p.requires_grad_(True)

    head_optimizer = torch.optim.AdamW(
        phi_params,
        lr=cfg.head_lr,
        weight_decay=cfg.head_weight_decay
    )

    sample_subj = cached_segs[active_subject_ids[0]]
    spatial_shape = sample_subj.spatial_shape
    num_voxels = int(np.prod(spatial_shape))
    batch_size = max(1, int(round(num_voxels * cfg.sample_prop)))
    amp_enabled = cfg.use_amp and device.type == 'cuda'
    is_cuda = device.type == 'cuda'

    s1_loops = int(s2a_data.get("stage1_dataset_loops", 0))
    s1_outer_steps = int(s2a_data.get("stage1_outer_steps", 0))
    s1_opt_time = float(s2a_data.get("stage1_elapsed_opt_time_sec", 0.0))
    s2a_fit_steps = int(s2a_data.get("fit_steps", 0))
    s2a_opt_time = float(s2a_data.get("elapsed_stage2a_opt_time_sec", 0.0))
    s2a_full_psnr = float(s2a_data.get("mean_full_vol_psnr_db", 0.0))
    compression_factor = float(s2a_data.get("compression_factor", 10.0))

    print(
        f"\n--- Running Phase 2B for [{s1_stem} | {s2a_stem} (T_f={s2a_fit_steps}, PSNR={s2a_full_psnr:.2f} dB)] ---"
    )

    metrics_history: List[Dict[str, Any]] = []
    summary_ckpt_rows: List[Dict[str, Any]] = []
    cumulative_s2b_time = 0.0
    total_head_steps = 0

    for loop_idx in range(1, cfg.max_head_loops + 1):
        t_loop_start = time.perf_counter()
        subj_order = list(active_subject_ids)
        random.shuffle(subj_order)

        loop_focal_loss = 0.0
        loop_batch_dice_all = 0.0
        loop_batch_dice_fg = 0.0
        loop_fwd_vram_mb = 0.0
        loop_bwd_vram_mb = 0.0

        model.train()

        for sid in tqdm.tqdm(
            subj_order,
            desc=f"Phase 2B [{s1_stem}/{s2a_stem}] Loop {loop_idx}/{cfg.max_head_loops}",
            leave=False
        ):
            subj_seg = cached_segs[sid]
            coords_flat = coord_cache.get_coords(subj_seg.spatial_shape)
            num_subj_voxels = coords_flat.shape[0]
            cur_batch_size = min(batch_size, num_subj_voxels)

            # Load frozen scan-specific backbone theta_j^* into feature_layers
            model.load_theta_state_dict(subject_thetas[sid])
            seg_u8_gpu = subj_seg.seg_uint8.to(device, non_blocking=True)

            head_optimizer.zero_grad(set_to_none=True)

            if cur_batch_size < num_subj_voxels:
                idx = torch.randint(0, num_subj_voxels, (cur_batch_size,), device=device)
                b_coords = coords_flat[idx]
                b_seg = seg_u8_gpu[idx].to(dtype=torch.long)
            else:
                b_coords = coords_flat
                b_seg = seg_u8_gpu.to(dtype=torch.long)

            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                # Backbone theta_j^* is frozen: extract penultimate features without autograd graph
                with torch.no_grad():
                    penultimate_feats = model.forward_features(b_coords)
                seg_logits = model.forward_seg_from_features(penultimate_feats)
                l_cls = multiclass_focal_loss(seg_logits, b_seg, gamma=cfg.focal_gamma)

            if is_cuda:
                fwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if fwd_mb > loop_fwd_vram_mb:
                    loop_fwd_vram_mb = fwd_mb
                torch.cuda.reset_peak_memory_stats(device)

            l_cls.backward()
            head_optimizer.step()

            if is_cuda:
                bwd_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if bwd_mb > loop_bwd_vram_mb:
                    loop_bwd_vram_mb = bwd_mb

            with torch.no_grad():
                pred_cls = torch.argmax(seg_logits.detach(), dim=-1)
                b_dice_all, b_dice_fg = compute_macro_dice_scores(pred_cls, b_seg, cfg.num_classes)

            loop_focal_loss += float(l_cls.item())
            loop_batch_dice_all += b_dice_all
            loop_batch_dice_fg += b_dice_fg
            total_head_steps += 1

        if is_cuda:
            torch.cuda.synchronize()
            reserved_vram_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
        else:
            reserved_vram_mb = 0.0

        loop_elapsed = time.perf_counter() - t_loop_start
        cumulative_s2b_time += loop_elapsed

        num_subjs = len(active_subject_ids)
        mean_focal = loop_focal_loss / num_subjs
        mean_b_dice_all = loop_batch_dice_all / num_subjs
        mean_b_dice_fg = loop_batch_dice_fg / num_subjs

        is_ckpt_loop = loop_idx in ckpt_loop_set
        ckpt_fname = f"stage2b_loop_{loop_idx:04d}.pth" if is_ckpt_loop else ""

        mean_full_dice_all_val = np.nan
        mean_full_dice_fg_val = np.nan

        if is_ckpt_loop and (cfg.eval_full_dice_at_ckpts or cfg.enable_visualization):
            eval_sids = active_subject_ids
            if cfg.eval_full_max_subjects > 0 and not cfg.enable_visualization:
                eval_sids = active_subject_ids[:cfg.eval_full_max_subjects]
            elif cfg.enable_visualization and cfg.max_vis_subjects > 0 and not cfg.eval_full_dice_at_ckpts:
                eval_sids = active_subject_ids[:cfg.max_vis_subjects]

            full_dices_all_list: List[float] = []
            full_dices_fg_list: List[float] = []

            for vis_idx, sid in enumerate(eval_sids):
                subj_seg = cached_segs[sid]
                coords_flat = coord_cache.get_coords(subj_seg.spatial_shape)
                seg_u8_gpu = subj_seg.seg_uint8.to(device, non_blocking=True)
                model.load_theta_state_dict(subject_thetas[sid])

                should_vis_subj = cfg.enable_visualization and (
                    cfg.max_vis_subjects < 0 or vis_idx < cfg.max_vis_subjects
                )

                f_dice_all, f_dice_fg, pred_vol_np = predict_full_volume_segmentation_and_dice(
                    model=model,
                    coords_flat=coords_flat,
                    seg_u8_gpu=seg_u8_gpu,
                    spatial_shape=subj_seg.spatial_shape,
                    num_classes=cfg.num_classes,
                    amp_enabled=amp_enabled,
                    chunk_size=cfg.infer_chunk_size,
                    return_volume=should_vis_subj
                )
                full_dices_all_list.append(f_dice_all)
                full_dices_fg_list.append(f_dice_fg)

                if should_vis_subj and pred_vol_np is not None:
                    save_seg_debug_visualization_and_volume(
                        pred_vol_np=pred_vol_np,
                        gt_seg_uint8_cpu=subj_seg.seg_uint8,
                        spatial_shape=subj_seg.spatial_shape,
                        affine=subj_seg.affine,
                        log_dir=s2b_img_dir,
                        subject_id=sid,
                        loop_idx=loop_idx,
                        num_classes=cfg.num_classes,
                        full_dice_all=f_dice_all,
                        full_dice_fg=f_dice_fg
                    )

            mean_full_dice_all_val = round(float(np.mean(full_dices_all_list)), 4)
            mean_full_dice_fg_val = round(float(np.mean(full_dices_fg_list)), 4)

        row_dict: Dict[str, Any] = {
            "stage1_checkpoint": s1_stem,
            "stage1_dataset_loops": s1_loops,
            "stage1_outer_steps": s1_outer_steps,
            "stage1_opt_time_sec": round(s1_opt_time, 4),
            "stage2a_checkpoint": s2a_stem,
            "stage2a_fit_steps": s2a_fit_steps,
            "stage2a_opt_time_sec": round(s2a_opt_time, 4),
            "stage2a_full_psnr_db": round(s2a_full_psnr, 4),
            "head_loop": loop_idx,
            "head_gradient_steps": total_head_steps,
            "elapsed_stage2b_opt_time_sec": round(cumulative_s2b_time, 4),
            "loop_opt_time_sec": round(loop_elapsed, 4),
            "total_pipeline_opt_time_sec": round(s1_opt_time + s2a_opt_time + cumulative_s2b_time, 4),
            "mean_focal_cls_loss": round(mean_focal, 6),
            "mean_batch_dice_all": round(mean_b_dice_all, 4),
            "mean_batch_dice_fg": round(mean_b_dice_fg, 4),
            "mean_full_vol_dice_all": mean_full_dice_all_val,
            "mean_full_vol_dice_fg": mean_full_dice_fg_val,
            "fwd_peak_vram_mb": round(loop_fwd_vram_mb, 2),
            "bwd_peak_vram_mb": round(loop_bwd_vram_mb, 2),
            "reserved_vram_mb": round(reserved_vram_mb, 2),
            "compression_factor": compression_factor,
            "sample_prop": cfg.sample_prop,
            "checkpoint_file": ckpt_fname,
        }
        metrics_history.append(row_dict)

        # Flush CSV after every single loop
        flush_metrics_csv(csv_log_path, metrics_history)

        if is_ckpt_loop:
            # Restore Stage 1 shared initialization theta^* so saved model_state_dict has [theta^*, phi^*]
            model.load_theta_state_dict(stage1_theta_state)
            ckpt_file = s2b_ckpt_dir / ckpt_fname

            payload: Dict[str, Any] = {
                "stage1_checkpoint_name": s1_stem,
                "stage1_dataset_loops": s1_loops,
                "stage1_outer_steps": s1_outer_steps,
                "stage1_elapsed_opt_time_sec": s1_opt_time,
                "stage2a_checkpoint_name": s2a_stem,
                "stage2a_fit_steps": s2a_fit_steps,
                "stage2a_elapsed_opt_time_sec": s2a_opt_time,
                "stage2a_mean_full_vol_psnr_db": s2a_full_psnr,
                "stage2b_head_loops": loop_idx,
                "stage2b_head_gradient_steps": total_head_steps,
                "elapsed_stage2b_opt_time_sec": cumulative_s2b_time,
                "total_pipeline_opt_time_sec": s1_opt_time + s2a_opt_time + cumulative_s2b_time,
                "mean_focal_cls_loss": mean_focal,
                "mean_batch_dice_all": mean_b_dice_all,
                "mean_batch_dice_fg": mean_b_dice_fg,
                "mean_full_vol_dice_all": mean_full_dice_all_val,
                "mean_full_vol_dice_fg": mean_full_dice_fg_val,
                "fwd_peak_vram_mb": loop_fwd_vram_mb,
                "bwd_peak_vram_mb": loop_bwd_vram_mb,
                "reserved_vram_mb": reserved_vram_mb,
                "compression_factor": compression_factor,
                "sample_prop": cfg.sample_prop,
                "spatial_shape": spatial_shape,
                "train_subject_ids": active_subject_ids,
                "model_config": model_cfg,
                "stage1_config": s2a_data.get("stage1_config", {}),
                "stage2a_config": s2a_data.get("stage2a_config", {}),
                "stage2b_config": asdict(cfg),
                # Optimal deployable [theta^*, phi^*] weights ready for Stage 3 test-time inference
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "theta_star_state_dict": stage1_theta_state,
                "phi_star_state_dict": model.get_phi_state_dict(),
            }
            torch.save(payload, ckpt_file)
            summary_ckpt_rows.append(row_dict)

            print(
                f"   -> [Loop {loop_idx:03d}/{cfg.max_head_loops}] Saved {ckpt_file.name} | "
                f"Focal: {mean_focal:.4f} | Batch Dice (All/FG): {mean_b_dice_all:.4f}/{mean_b_dice_fg:.4f} | "
                f"Opt Time: {cumulative_s2b_time:.1f}s | VRAM (Fwd/Bwd): {loop_fwd_vram_mb:.0f}/{loop_bwd_vram_mb:.0f} MB"
            )

    return summary_ckpt_rows


def run_stage2b_pipeline(
    dataset_dir: Path,
    run_dir: Path,
    cfg: Stage2BConfig,
    device: torch.device,
    specific_stage1_ckpt: Optional[str] = None,
    specific_stage2a_ckpt: Optional[str] = None
) -> None:
    """
    Discovers all Stage 1 subdirectories and Stage 2A checkpoints inside `<run_dir>/phase2/`,
    preloads the training segmentation masks as `uint8` in CPU RAM, and executes Phase 2B
    across all checkpoints.

    Args:
        dataset_dir (Path): Root dataset path containing the `train` subdirectory.
        run_dir (Path): Timestamped run directory containing `phase2/`.
        cfg (Stage2BConfig): Phase 2B hyperparameter configuration.
        device (torch.device): Target compute device.
        specific_stage1_ckpt (Optional[str]): Optional Stage 1 stem/filename filter (e.g., 'stage1_loop_0064').
        specific_stage2a_ckpt (Optional[str]): Optional Stage 2A filename filter (e.g., 'stage2a_step_0064.pth').
    """
    if not (0.0 < cfg.sample_prop <= 1.0):
        raise ValueError(f"sample_prop must satisfy 0 < sample_prop <= 1.0, got {cfg.sample_prop}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    phase2_root_dir = run_dir / "phase2"
    if not phase2_root_dir.exists():
        raise FileNotFoundError(f"Phase 2 directory not found: {phase2_root_dir}")

    # Discover Stage 1 subdirectories under <run_dir>/phase2/
    if specific_stage1_ckpt is not None:
        s1_stem_clean = Path(specific_stage1_ckpt).stem
        candidate_dir = phase2_root_dir / s1_stem_clean
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"Specified Stage 1 directory not found in phase2: {candidate_dir}")
        s1_dirs = [candidate_dir]
    else:
        s1_dirs = sorted([d for d in phase2_root_dir.iterdir() if d.is_dir() and d.name.startswith("stage1_")])

    if not s1_dirs:
        raise FileNotFoundError(f"No 'stage1_*' subdirectories found inside {phase2_root_dir}")

    # Collect all (s1_dir, s2a_ckpt_path) pairs
    s2a_tasks: List[Tuple[Path, Path]] = []
    for s1_dir in s1_dirs:
        s2a_ckpt_dir = s1_dir / "checkpoints"
        if not s2a_ckpt_dir.is_dir():
            continue
        if specific_stage2a_ckpt is not None:
            s2a_name = (
                specific_stage2a_ckpt
                if specific_stage2a_ckpt.endswith(".pth")
                else f"{specific_stage2a_ckpt}.pth"
            )
            cand_s2a = s2a_ckpt_dir / s2a_name
            if cand_s2a.exists():
                s2a_tasks.append((s1_dir, cand_s2a))
        else:
            for p in sorted(s2a_ckpt_dir.glob("stage2a_step_*.pth")):
                s2a_tasks.append((s1_dir, p))

    if not s2a_tasks:
        raise FileNotFoundError(f"No Stage 2A checkpoint files ('stage2a_step_*.pth') found under {phase2_root_dir}")

    # Peek at the first Stage 2A checkpoint to get the exact training subject IDs
    first_s2a_data = torch.load(s2a_tasks[0][1], map_location="cpu", weights_only=False)
    train_subject_ids: List[str] = list(first_s2a_data["train_subject_ids"])

    cached_segs = preload_segmentations_as_uint8(
        dataset_dir=dataset_dir,
        subject_ids=train_subject_ids,
        pin_memory=cfg.pin_cached_memory
    )

    config_file_path = phase2_root_dir / "stage2b_config.yml"
    with open(config_file_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    checkpoint_loops = [
        l_idx for l_idx in range(1, cfg.max_head_loops + 1)
        if is_power_of_two_loop(l_idx, cfg.min_checkpoint_loop, cfg.max_head_loops)
    ]

    print("==================================================================")
    print(f"Phase 2B Global Segmentation Head Optimization Configuration")
    print(f"Run Root Directory         : {run_dir}")
    print(f"Phase 2 Root Directory     : {phase2_root_dir}")
    print(f"Stage 2A Checkpoints Queued: {len(s2a_tasks)}")
    print(f"Active Training Subjects   : {len(cached_segs)}")
    print(f"Stochastic Batch Proportion: {cfg.sample_prop:.4f}")
    print(f"Max Head Loops             : {cfg.max_head_loops} (LR={cfg.head_lr:.1e}, Gamma={cfg.focal_gamma})")
    print(f"Power-of-2 Checkpoints     : {checkpoint_loops}")
    print(f"Enable Visualization       : {cfg.enable_visualization} (Max Vis Subjects: {cfg.max_vis_subjects})")
    print("==================================================================")

    coord_cache = CoordinateGridCache(device=device)
    master_summary_rows: List[Dict[str, Any]] = []
    master_csv_path = phase2_root_dir / "stage2b_all_runs_summary.csv"

    def _handle_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        for s1_dir, s2a_path in s2a_tasks:
            ckpt_rows = process_single_stage2a_checkpoint(
                s2a_ckpt_path=s2a_path,
                s2a_out_dir=s1_dir,
                cached_segs=cached_segs,
                coord_cache=coord_cache,
                cfg=cfg,
                device=device
            )
            master_summary_rows.extend(ckpt_rows)
            if master_summary_rows:
                pd.DataFrame(master_summary_rows).to_csv(master_csv_path, index=False)

    except KeyboardInterrupt:
        print("\nPhase 2B interrupted by user/signal. Flushing master summary CSV...")
        if master_summary_rows:
            pd.DataFrame(master_summary_rows).to_csv(master_csv_path, index=False)

    if master_summary_rows:
        print(f"\nSaved master Phase 2B summary CSV ({len(master_summary_rows)} checkpoints) to: {master_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 2B Global Segmentation Head Optimization for Full-Resolution 3D MetaSeg. "
                    "CLI flags default to None and fall back to Stage2BConfig dataclass defaults."
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
        help="Optional specific Stage 1 checkpoint stem or filename (e.g., stage1_loop_0064) to process alone."
    )
    parser.add_argument(
        "--specific_stage2a_ckpt",
        type=str,
        default=None,
        help="Optional specific Stage 2A checkpoint filename (e.g., stage2a_step_0064.pth) to process alone."
    )
    parser.add_argument(
        "--enable_visualization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save middle-slice segmentation PNGs and predicted NIfTI label volumes in images/."
    )
    parser.add_argument(
        "--max_vis_subjects",
        type=int,
        default=None,
        help="Number of subjects to save PNG/NIfTI visualizations for (-1 saves all subjects, default 1)."
    )
    parser.add_argument(
        "--eval_full_dice_at_ckpts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute exact full-volume Dice at power-of-2 checkpoints."
    )
    parser.add_argument(
        "--eval_full_max_subjects",
        type=int,
        default=None,
        help="Number of subjects evaluated when eval_full_dice_at_ckpts is True (-1 evaluates all)."
    )
    parser.add_argument(
        "--sample_prop",
        type=float,
        default=None,
        help="Proportion of voxels (0 < sample_prop <= 1) sampled per subject step."
    )
    parser.add_argument(
        "--max_head_loops",
        type=int,
        default=None,
        help="Maximum passes over the fitted training set (overrides Stage2BConfig.max_head_loops, default 128)."
    )
    parser.add_argument(
        "--min_checkpoint_loop",
        type=int,
        default=None,
        help="Minimum power-of-2 loop to save checkpoints (overrides Stage2BConfig.min_checkpoint_loop, default 2)."
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=None,
        help="AdamW learning rate for segmentation head (overrides Stage2BConfig.head_lr, default 5e-5)."
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Focal loss focusing parameter gamma (overrides Stage2BConfig.focal_gamma, default 3.0)."
    )
    parser.add_argument(
        "--skip_existing_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Stage 2A checkpoints whose Phase 2B outputs are already complete."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 2B on {device}...")

    dataset_path = Path(args.dataset_path)
    run_dir_path = Path(args.run_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")
    if not run_dir_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")

    cfg_overrides = {
        k: v for k, v in vars(args).items()
        if k in Stage2BConfig.__dataclass_fields__ and v is not None
    }
    stage2b_cfg = Stage2BConfig(**cfg_overrides)

    run_stage2b_pipeline(
        dataset_dir=dataset_path,
        run_dir=run_dir_path,
        cfg=stage2b_cfg,
        device=device,
        specific_stage1_ckpt=args.specific_stage1_ckpt,
        specific_stage2a_ckpt=args.specific_stage2a_ckpt
    )