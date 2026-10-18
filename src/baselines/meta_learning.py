import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

torch.set_float32_matmul_precision('high')


@dataclass
class MetaSegConfig:
    """Configuration hyperparameters for 3D MetaSeg training and inference."""
    # Data preprocessing
    use_paper_preprocessing: bool = True  # Crop to (160, 160, 200) and 2x downsample to (80, 80, 100)
    crop_shape: Tuple[int, int, int] = (160, 160, 200)
    downsample_factor: int = 2
    num_classes: int = 5  # aligned_seg4.nii.gz has 5 classes: 0 (bg), 1, 2, 3, 4

    # Split ratios (Paper uses 214 train / 100 val / 100 test for 414 scans; adapts if fewer scans exist)
    train_ratio: float = 0.52
    val_ratio: float = 0.24
    seed: int = 42

    # Architecture (3D configuration from paper: L=5, h=256, w0=30 -> ~331K params)
    in_dim: int = 3
    out_dim: int = 1
    num_hidden_layers: int = 5
    hidden_dim: int = 256
    omega_0: float = 30.0

    # Meta-learning (Phase 1)
    outer_epochs: int = 10          # T_o = 10 epochs over training set
    inner_steps: int = 2            # T_i = 2 gradient steps per subject
    inner_lr: float = 1e-4          # Adam LR for inner loop
    outer_lr: float = 1e-4          # Adam LR for outer loop
    val_interval: int = 50          # Validate every 50 outer steps

    # Loss hyperparameters for 3D
    focal_gamma: float = 3.0        # gamma = 3.0 for 3D volumes
    bg_recon_weight: float = 0.1    # 0.1 scaling for background pixels in 3D
    bg_threshold: float = 1e-5      # Intensity threshold to identify background without label leakage

    # Segmentation Head Optimization (Phase 2) & Inference (Phase 3)
    fit_steps: int = 100            # T_f = 100 steps for pixel fitting
    fit_lr: float = 1e-4            # Adam LR during test-time / Phase 2 pixel fitting
    head_epochs: int = 10           # Epochs to globally optimize g_phi in Phase 2
    head_lr: float = 5e-5           # Reduced LR (5e-5) for 5-class segmentation head tuning

    # Memory management: number of coordinates sampled per step (None = full volume if downsampled)
    coord_batch_size: Optional[int] = 160_000


class SineLayer(nn.Module):
    """
    Linear layer followed by a periodic sine activation with SIREN initialization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ) -> None:
        """
        Initializes the SineLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, includes a learnable bias.
            is_first (bool): If True, applies first-layer SIREN uniform initialization.
            omega_0 (float): Frequency scaling factor for the sine activation.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights according to Sitzmann et al. (SIREN)."""
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying scaled sine activation.

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
        hidden_dim: int = 256,
        omega_0: float = 30.0
    ) -> None:
        """
        Initializes the MetaSegINR architecture.

        Args:
            in_dim (int): Spatial coordinate dimensionality (e.g., 3 for 3D MRI).
            out_dim (int): Output intensity channels (e.g., 1 for T1 MRI).
            num_classes (int): Number of segmentation classes C.
            num_hidden_layers (int): Number of SIREN hidden layers L.
            hidden_dim (int): Width h of the hidden layers.
            omega_0 (float): SIREN frequency parameter.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Reconstruction backbone f_theta
        layers: List[nn.Module] = [
            SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.feature_layers = nn.Sequential(*layers)

        self.recon_out = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.recon_out.weight.uniform_(-bound, bound)
            self.recon_out.bias.zero_()

        # Segmentation head g_phi: 1 FC layer with LeakyReLU + Linear layer with C outputs
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

    def get_theta_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns a CPU state dictionary copy for f_theta parameters only."""
        return {
            "feature_layers": {k: v.detach().cpu().clone() for k, v in self.feature_layers.state_dict().items()},
            "recon_out": {k: v.detach().cpu().clone() for k, v in self.recon_out.state_dict().items()},
        }

    def load_theta_state_dict(self, theta_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Loads f_theta parameters from a saved theta state dictionary.

        Args:
            theta_state (Dict[str, Dict[str, torch.Tensor]]): Dictionary containing 'feature_layers'
                and 'recon_out' state dicts.
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

    def forward_recon(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Predicts only image intensities I(x) = f_theta(x) for fast test-time fitting.

        Args:
            coords (torch.Tensor): Normalized coordinates of shape (N, in_dim) in [-1, 1].

        Returns:
            torch.Tensor: Predicted intensities of shape (N, out_dim).
        """
        features = self.feature_layers(coords)
        return self.recon_out(features)

    def forward_seg_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts segmentation class logits from penultimate features.

        Args:
            features (torch.Tensor): Penultimate features of shape (N, hidden_dim).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (N, num_classes).
        """
        return self.seg_head(features)

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
        pred_intensity (torch.Tensor): Predicted intensities of shape (N, D).
        target_intensity (torch.Tensor): Ground truth intensities of shape (N, D) in [0, 1].
        bg_recon_weight (float): Multiplicative weight applied to background pixels.
        bg_threshold (float): Threshold below which target intensities are treated as background.

    Returns:
        torch.Tensor: Scalar mean weighted MSE loss of shape ().
    """
    sq_err = (pred_intensity - target_intensity).pow(2)
    weights = torch.where(target_intensity <= bg_threshold, bg_recon_weight, 1.0)
    return torch.mean(weights * sq_err)


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 3.0
) -> torch.Tensor:
    """
    Computes multiclass focal classification loss as defined in Eq. (2) of MetaSeg.

    Args:
        logits (torch.Tensor): Predicted raw class logits of shape (N, C).
        targets (torch.Tensor): Ground truth integer class indices of shape (N,).
        gamma (float): Focal loss focusing hyperparameter.

    Returns:
        torch.Tensor: Scalar mean focal loss of shape ().
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    focal_weight = (1.0 - target_probs).pow(gamma)
    loss = -focal_weight * target_log_probs
    return loss.mean()


def compute_dice_score(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int = 5,
    include_background: bool = True
) -> float:
    """
    Computes the macro-averaged Dice similarity coefficient across classes.

    Args:
        pred_labels (torch.Tensor): Predicted integer labels of shape (N,).
        true_labels (torch.Tensor): Ground truth integer labels of shape (N,).
        num_classes (int): Total number of segmentation classes C.
        include_background (bool): Whether to include class 0 in the average (paper averages all 5 classes).

    Returns:
        float: Mean Dice score across evaluated classes.
    """
    start_cls = 0 if include_background else 1
    dice_scores: List[float] = []
    eps = 1e-6

    for c in range(start_cls, num_classes):
        pred_c = (pred_labels == c)
        true_c = (true_labels == c)
        intersection = (pred_c & true_c).sum().item()
        cardinality = pred_c.sum().item() + true_c.sum().item()
        if cardinality == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection + eps) / (cardinality + eps))

    return float(np.mean(dice_scores))


def center_crop_or_pad_3d(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Center-crops or zero-pads a 3D volume to the specified spatial dimensions.

    Args:
        volume (torch.Tensor): Input 3D tensor of shape (Z, Y, X).
        target_shape (Tuple[int, int, int]): Target spatial dimensions (tZ, tY, tX).

    Returns:
        torch.Tensor: Cropped or padded 3D tensor of shape (tZ, tY, tX).
    """
    out = volume
    for dim in range(3):
        curr_size = out.shape[dim]
        targ_size = target_shape[dim]
        if curr_size > targ_size:
            start = (curr_size - targ_size) // 2
            out = out.narrow(dim, start, targ_size)
        elif curr_size < targ_size:
            pad_total = targ_size - curr_size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_spec = [0, 0, 0, 0, 0, 0]
            # F.pad expects reverse dimension order: (X_left, X_right, Y_left, Y_right, Z_left, Z_right)
            rev_idx = 2 * (2 - dim)
            pad_spec[rev_idx] = pad_before
            pad_spec[rev_idx + 1] = pad_after
            out = F.pad(out, pad_spec, mode='constant', value=0)
    return out


def load_oasis_subject(
    subject_dir: Path,
    cfg: MetaSegConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads and preprocesses `aligned_orig.nii.gz` and `aligned_seg4.nii.gz` for a subject.

    Args:
        subject_dir (Path): Path to the OASIS subject directory.
        cfg (MetaSegConfig): Configuration object controlling cropping and downsampling.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - img_tensor: Normalized float32 intensity volume of shape (1, Z, Y, X) in [0, 1].
            - seg_tensor: Long integer segmentation mask of shape (Z, Y, X) with labels in {0..C-1}.
    """
    img_path = subject_dir / "aligned_orig.nii.gz"
    seg_path = subject_dir / "aligned_seg4.nii.gz"

    img_np = np.transpose(nib.load(str(img_path)).get_fdata()).astype(np.float32)
    seg_np = np.transpose(nib.load(str(seg_path)).get_fdata()).astype(np.int64)

    img_tensor = torch.from_numpy(img_np).squeeze()
    seg_tensor = torch.from_numpy(seg_np).squeeze()

    if cfg.use_paper_preprocessing:
        img_tensor = center_crop_or_pad_3d(img_tensor, cfg.crop_shape)
        seg_tensor = center_crop_or_pad_3d(seg_tensor, cfg.crop_shape)

        if cfg.downsample_factor > 1:
            scale = 1.0 / cfg.downsample_factor
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0).unsqueeze(0),
                scale_factor=scale,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            seg_tensor = F.interpolate(
                seg_tensor.unsqueeze(0).unsqueeze(0).float(),
                scale_factor=scale,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()

    # Normalize intensities to [0, 1] as required by MetaSeg
    min_val = img_tensor.min()
    max_val = img_tensor.max()
    if max_val > min_val:
        img_tensor = (img_tensor - min_val) / (max_val - min_val)

    return img_tensor.unsqueeze(0), seg_tensor


def build_normalized_coordinate_grid(
    spatial_shape: Tuple[int, ...],
    device: torch.device
) -> torch.Tensor:
    """
    Generates a flattened 3D coordinate grid normalized to [-1, 1]^D.

    Args:
        spatial_shape (Tuple[int, ...]): Spatial dimensions (Z, Y, X) of the volume.
        device (torch.device): Target compute device.

    Returns:
        torch.Tensor: Flattened coordinate grid of shape (N_voxels, D) in [-1, 1].
    """
    axes = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    return grid.view(-1, len(spatial_shape))


def fit_INR_to_scan_pixels(
    model: MetaSegINR,
    theta_init: Dict[str, Dict[str, torch.Tensor]],
    coords_flat: torch.Tensor,
    img_flat: torch.Tensor,
    cfg: MetaSegConfig
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], float]:
    """
    Fits the reconstruction backbone f_theta initialized at theta^* to a scan's pixels
    for T_f iterations using only the reconstruction loss (used in Phase 2 and Phase 3).

    Args:
        model (MetaSegINR): The MetaSeg model instance.
        theta_init (Dict[str, Dict[str, torch.Tensor]]): Meta-learned initial weights theta^*.
        coords_flat (torch.Tensor): Flattened coordinate grid of shape (N_voxels, D).
        img_flat (torch.Tensor): Flattened target intensities of shape (N_voxels, out_dim).
        cfg (MetaSegConfig): Hyperparameter configuration.

    Returns:
        Tuple[Dict[str, Dict[str, torch.Tensor]], float]:
            - converged_theta: Fitted state dictionary for f_theta.
            - psnr: Final full-volume reconstruction PSNR in dB.
    """
    model.load_theta_state_dict(theta_init)
    model.train()

    for p in model.get_theta_parameters():
        p.requires_grad_(True)
    for p in model.get_phi_parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.get_theta_parameters(), lr=cfg.fit_lr)
    num_voxels = coords_flat.shape[0]
    batch_size = cfg.coord_batch_size

    for _ in range(cfg.fit_steps):
        optimizer.zero_grad(set_to_none=True)
        if batch_size is not None and batch_size < num_voxels:
            idx = torch.randint(0, num_voxels, (batch_size,), device=coords_flat.device)
            b_coords = coords_flat[idx]
            b_target = img_flat[idx]
        else:
            b_coords = coords_flat
            b_target = img_flat

        pred_img = model.forward_recon(b_coords)
        loss = weighted_reconstruction_loss(
            pred_img, b_target, cfg.bg_recon_weight, cfg.bg_threshold
        )
        loss.backward()
        optimizer.step()

    # Compute full-volume PSNR
    model.eval()
    mse_sum = 0.0
    chunk_size = 250_000
    with torch.no_grad():
        for i in range(0, num_voxels, chunk_size):
            end = min(i + chunk_size, num_voxels)
            pred = model.forward_recon(coords_flat[i:end])
            mse_sum += F.mse_loss(pred, img_flat[i:end], reduction='sum').item()

    full_mse = mse_sum / num_voxels
    psnr = 10.0 * math.log10(1.0 / full_mse) if full_mse > 0 else 100.0
    return model.get_theta_state_dict(), psnr


def predict_volume_segmentation(
    model: MetaSegINR,
    coords_flat: torch.Tensor,
    chunk_size: int = 250_000
) -> torch.Tensor:
    """
    Predicts per-voxel class labels in chunks using the fitted f_theta and g_phi.

    Args:
        model (MetaSegINR): Fitted MetaSegINR model.
        coords_flat (torch.Tensor): Flattened coordinate grid of shape (N_voxels, D).
        chunk_size (int): Maximum number of coordinates evaluated per forward pass.

    Returns:
        torch.Tensor: Predicted integer class labels of shape (N_voxels,).
    """
    model.eval()
    preds: List[torch.Tensor] = []
    num_voxels = coords_flat.shape[0]

    with torch.no_grad():
        for i in range(0, num_voxels, chunk_size):
            end = min(i + chunk_size, num_voxels)
            features = model.forward_features(coords_flat[i:end])
            logits = model.forward_seg_from_features(features)
            pred_cls = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            preds.append(pred_cls)

    return torch.cat(preds, dim=0)


def run_metaseg_pipeline(
    oasis_dir: Path,
    results_dir: Path,
    cfg: MetaSegConfig,
    device: torch.device
) -> None:
    """
    Executes the complete 3-phase MetaSeg pipeline (Meta-Learning, Head Optimization,
    and Test-Time Inference) on the OASIS dataset.

    Args:
        oasis_dir (Path): Root directory containing OASIS subject folders.
        results_dir (Path): Directory to save checkpoints and test predictions.
        cfg (MetaSegConfig): Configuration parameters.
        device (torch.device): Compute device.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Discover valid subjects with both aligned_orig.nii.gz and aligned_seg4.nii.gz
    valid_subjects: List[Path] = []
    for p in sorted(oasis_dir.iterdir()):
        if p.is_dir() and (p / "aligned_orig.nii.gz").exists() and (p / "aligned_seg4.nii.gz").exists():
            valid_subjects.append(p)

    if not valid_subjects:
        raise FileNotFoundError(
            f"No valid OASIS subjects containing both 'aligned_orig.nii.gz' and "
            f"'aligned_seg4.nii.gz' found in {oasis_dir}"
        )

    random.shuffle(valid_subjects)
    n_total = len(valid_subjects)

    if n_total >= 414:
        train_subjects = valid_subjects[:214]
        val_subjects = valid_subjects[214:314]
        test_subjects = valid_subjects[314:414]
    else:
        n_train = max(1, int(n_total * cfg.train_ratio))
        n_val = max(1, int(n_total * cfg.val_ratio))
        train_subjects = valid_subjects[:n_train]
        val_subjects = valid_subjects[n_train:n_train + n_val]
        test_subjects = valid_subjects[n_train + n_val:]
        if not test_subjects:
            test_subjects = val_subjects

    print(f"Found {n_total} valid OASIS subjects -> "
          f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    model = MetaSegINR(
        in_dim=cfg.in_dim,
        out_dim=cfg.out_dim,
        num_classes=cfg.num_classes,
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_dim=cfg.hidden_dim,
        omega_0=cfg.omega_0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized MetaSegINR with {total_params:,} total parameters.")

    # =========================================================================
    # PHASE 1: Meta-Learning Initialization (theta^*, phi^{T_o})
    # =========================================================================
    print("\n--- Phase 1: Meta-Learning Initialization ---")
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
    global_step = 0

    for epoch in range(cfg.outer_epochs):
        random.shuffle(train_subjects)
        for subj_dir in tqdm.tqdm(train_subjects, desc=f"Meta Epoch {epoch + 1}/{cfg.outer_epochs}"):
            img_tensor, seg_tensor = load_oasis_subject(subj_dir, cfg)
            spatial_shape = img_tensor.shape[1:]
            coords_flat = build_normalized_coordinate_grid(spatial_shape, device)
            img_flat = img_tensor.permute(1, 2, 3, 0).reshape(-1, cfg.out_dim).to(device)
            seg_flat = seg_tensor.reshape(-1).to(device)
            num_voxels = coords_flat.shape[0]

            # Snapshot current outer parameters [theta^t, phi^t]
            outer_state = copy.deepcopy(model.state_dict())
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.inner_lr)

            model.train()
            for p in model.parameters():
                p.requires_grad_(True)

            # Inner routine: T_i steps on subject j
            for _ in range(cfg.inner_steps):
                inner_optimizer.zero_grad(set_to_none=True)
                if cfg.coord_batch_size is not None and cfg.coord_batch_size < num_voxels:
                    idx = torch.randint(0, num_voxels, (cfg.coord_batch_size,), device=device)
                    b_coords = coords_flat[idx]
                    b_img = img_flat[idx]
                    b_seg = seg_flat[idx]
                else:
                    b_coords = coords_flat
                    b_img = img_flat
                    b_seg = seg_flat

                pred_img, seg_logits = model(b_coords)
                l_recon = weighted_reconstruction_loss(
                    pred_img, b_img, cfg.bg_recon_weight, cfg.bg_threshold
                )
                l_cls = multiclass_focal_loss(seg_logits, b_seg, gamma=cfg.focal_gamma)
                l_inner = l_recon + l_cls
                l_inner.backward()
                inner_optimizer.step()

            # Outer update (Eq. 3): Use parameter difference as gradient signal
            converged_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(outer_state)
            outer_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.grad = param.data - converged_state[name].data

            outer_optimizer.step()
            global_step += 1

            # Periodic validation check
            if global_step % cfg.val_interval == 0 and len(val_subjects) > 0:
                val_subj = random.choice(val_subjects)
                v_img, v_seg = load_oasis_subject(val_subj, cfg)
                v_coords = build_normalized_coordinate_grid(v_img.shape[1:], device)
                v_img_flat = v_img.permute(1, 2, 3, 0).reshape(-1, cfg.out_dim).to(device)
                v_seg_flat = v_seg.reshape(-1).to(device)

                curr_theta = model.get_theta_state_dict()
                _, v_psnr = fit_INR_to_scan_pixels(model, curr_theta, v_coords, v_img_flat, cfg)
                v_pred_seg = predict_volume_segmentation(model, v_coords)
                v_dice = compute_dice_score(v_pred_seg, v_seg_flat, cfg.num_classes)
                model.load_theta_state_dict(curr_theta)
                print(f" [Step {global_step}] Val Subject ({val_subj.name}) -> "
                      f"PSNR: {v_psnr:.2f} dB | Dice: {v_dice:.4f}")

    # Freeze theta^* = theta^{T_o}
    theta_star = model.get_theta_state_dict()
    torch.save(model.state_dict(), results_dir / "metaseg_phase1_init.pth")

    # =========================================================================
    # PHASE 2: Fit f_{theta^*} on Training Scans & Globally Optimize g_phi
    # =========================================================================
    print("\n--- Phase 2: Fitting f_{theta^*} per Training Scan & Optimizing g_phi ---")
    fitted_train_thetas: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    for subj_dir in tqdm.tqdm(train_subjects, desc="Fitting theta^* to training scans"):
        img_tensor, _ = load_oasis_subject(subj_dir, cfg)
        coords_flat = build_normalized_coordinate_grid(img_tensor.shape[1:], device)
        img_flat = img_tensor.permute(1, 2, 3, 0).reshape(-1, cfg.out_dim).to(device)

        subj_theta_star, _ = fit_INR_to_scan_pixels(model, theta_star, coords_flat, img_flat, cfg)
        fitted_train_thetas[subj_dir.name] = subj_theta_star

    # Globally optimize g_phi over all training scans (Eq. 4)
    for p in model.get_theta_parameters():
        p.requires_grad_(False)
    for p in model.get_phi_parameters():
        p.requires_grad_(True)

    head_optimizer = torch.optim.Adam(model.get_phi_parameters(), lr=cfg.head_lr)

    for epoch in range(cfg.head_epochs):
        random.shuffle(train_subjects)
        epoch_loss = 0.0
        for subj_dir in tqdm.tqdm(train_subjects, desc=f"Head Opt Epoch {epoch + 1}/{cfg.head_epochs}"):
            img_tensor, seg_tensor = load_oasis_subject(subj_dir, cfg)
            coords_flat = build_normalized_coordinate_grid(img_tensor.shape[1:], device)
            seg_flat = seg_tensor.reshape(-1).to(device)
            num_voxels = coords_flat.shape[0]

            model.load_theta_state_dict(fitted_train_thetas[subj_dir.name])
            model.train()
            head_optimizer.zero_grad(set_to_none=True)

            if cfg.coord_batch_size is not None and cfg.coord_batch_size < num_voxels:
                idx = torch.randint(0, num_voxels, (cfg.coord_batch_size,), device=device)
                b_coords = coords_flat[idx]
                b_seg = seg_flat[idx]
            else:
                b_coords = coords_flat
                b_seg = seg_flat

            with torch.no_grad():
                penultimate_feats = model.forward_features(b_coords)

            seg_logits = model.forward_seg_from_features(penultimate_feats)
            loss_seg = multiclass_focal_loss(seg_logits, b_seg, gamma=cfg.focal_gamma)
            loss_seg.backward()
            head_optimizer.step()
            epoch_loss += loss_seg.item()

        print(f"Head Epoch {epoch + 1}/{cfg.head_epochs} | Mean Focal Loss: {epoch_loss / len(train_subjects):.4f}")

    # Restore theta^* alongside converged phi^* and save final meta-checkpoint
    model.load_theta_state_dict(theta_star)
    final_ckpt_path = results_dir / "metaseg_optimal_theta_phi.pth"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved optimal [theta^*, phi^*] checkpoint to {final_ckpt_path}")

    # =========================================================================
    # PHASE 3: Test-Time Inference ("Fit Pixels, Get Labels")
    # =========================================================================
    print("\n--- Phase 3: Test-Time Inference ---")
    test_dices: List[float] = []
    test_psnrs: List[float] = []

    for subj_dir in tqdm.tqdm(test_subjects, desc="Evaluating Test Scans"):
        t0 = time.time()
        img_tensor, seg_tensor = load_oasis_subject(subj_dir, cfg)
        spatial_shape = img_tensor.shape[1:]
        coords_flat = build_normalized_coordinate_grid(spatial_shape, device)
        img_flat = img_tensor.permute(1, 2, 3, 0).reshape(-1, cfg.out_dim).to(device)
        seg_flat = seg_tensor.reshape(-1).to(device)

        # Fit pixels for T_f = 100 steps starting from theta^*
        _, test_psnr = fit_INR_to_scan_pixels(model, theta_star, coords_flat, img_flat, cfg)

        # Decode segmentation in one feed-forward pass through g_{phi^*}
        pred_seg_flat = predict_volume_segmentation(model, coords_flat)
        test_dice = compute_dice_score(pred_seg_flat, seg_flat, cfg.num_classes)

        test_psnrs.append(test_psnr)
        test_dices.append(test_dice)

        # Save predicted segmentation volume
        subj_out_dir = results_dir / subj_dir.name
        subj_out_dir.mkdir(parents=True, exist_ok=True)
        pred_vol_np = pred_seg_flat.view(*spatial_shape).cpu().numpy().astype(np.uint8)
        pred_vol_np = np.transpose(pred_vol_np)
        nib.save(nib.Nifti1Image(pred_vol_np, np.eye(4)), str(subj_out_dir / "metaseg_pred_seg4.nii.gz"))

        print(f"Subject {subj_dir.name} ({time.time() - t0:.2f}s) -> "
              f"PSNR: {test_psnr:.2f} dB | Dice: {test_dice:.4f}")

    print("\n==========================================")
    print(f"Test Evaluation Summary ({len(test_subjects)} scans):")
    print(f"Mean PSNR: {np.mean(test_psnrs):.2f} +/- {np.std(test_psnrs):.2f} dB")
    print(f"Mean Dice: {np.mean(test_dices):.4f} +/- {np.std(test_dices):.4f}")
    print("==========================================")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running MetaSeg baseline on {device}...")

    oasis_path = Path("/home/nil/data/OASIS/")
    results_path = Path("./results_metaseg/")

    if not oasis_path.exists():
        raise FileNotFoundError(f"OASIS dataset path does not exist: {oasis_path}")

    config = MetaSegConfig()
    run_metaseg_pipeline(oasis_path, results_path, config, device)