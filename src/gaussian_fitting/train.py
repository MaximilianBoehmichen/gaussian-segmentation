import time
import math
import warnings
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import tqdm
from torch_geometric.data import Data

from gaussian_fitting import knn
from gaussian_fitting.gaussian_representation import GaussianIntensityField
from gaussian_fitting.visualization import save_visualization, save_reconstructed_volume, save_interpolated_volume

# 1. Enable TF32 for massive Ampere+ Tensor Core speedups
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")
torch.set_float32_matmul_precision('high')


@dataclass
class TrainConfig:
    compression_factor: float = 10.0
    max_epochs: int = 101
    logging_rate: int = 20
    sample_prop: float = 0.1
    # Component-specific learning rates
    lr_mus: float = 1e-1
    lr_intensities: float = 1e-1
    lr_L_diag: float = 1e-1
    lr_L_offdiag: float = 1e-1
    # Scheduler configuration
    scheduler_type: str = 'linear'  # Options: 'cosine', 'exponential', 'linear'
    scheduler_eta_min: float = 1e-1
    scheduler_gamma: float = 0.9857  # Used for 'exponential' and 'step'
    knn_type: str = 'structured_voronoi'  # 'knn', 'voronoi', 'structured_voronoi'
    knn_update_rate: int = 50
    K: int = 24
    k_graph: int = 15  # NEW: Number of neighbors for the Gaussian-to-Gaussian PyG graph
    K_candidates: int = 300  # For structured component
    deform_control_points_per_dim: Tuple[int, int, int] = (20, 20, 20)  # For structured component
    voronoi_nlist: int = 1024  # Ideally should be ~sqrt(centroids). For voronoi component
    voronoi_nprobe: int = 32  # Should be 1% - 10% of nlist. For voronoi component

    # Geometric bounds
    min_spatial_scale: float = 1.0
    max_spatial_scale: float = 20.0


def extract_pyg_data(gs: GaussianIntensityField, k_graph: int = 15, psnr: Optional[float] = None) -> Data:
    with torch.no_grad():
        mus = gs.mus.detach().cpu()

        max_prec_diag = 1.0 / gs.min_spatial_scale
        min_prec_diag = 1.0 / gs.max_spatial_scale

        diag_elements = min_prec_diag + (max_prec_diag - min_prec_diag) * torch.sigmoid(gs.L_diag.detach().cpu())
        bounded_offdiag = torch.tanh(gs.L_offdiag.detach().cpu()) * max_prec_diag
        intensities = gs.intensities.detach().cpu()

    x = torch.cat([mus, diag_elements, bounded_offdiag, intensities], dim=-1)
    num_nodes = mus.shape[0]

    index = faiss.IndexFlatL2(mus.shape[1])
    index.add(mus.numpy().astype(np.float32))

    _, knn_indices = index.search(mus.numpy().astype(np.float32), k_graph + 1)
    knn_indices = knn_indices[:, 1:]  # (N, k_graph)
    src = np.repeat(np.arange(num_nodes), k_graph)
    dst = knn_indices.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    data = Data(x=x, pos=mus, edge_index=edge_index)
    if psnr is not None:
        data.psnr = torch.tensor([psnr], dtype=torch.float32)

    return data


def train(run_dir: Path, gs: GaussianIntensityField, img_tensor: torch.Tensor,
          cfg: TrainConfig, device: torch.device, enable_logging: bool = True) -> float:
    """
    Main training loop optimizing the Gaussian Intensity Field.

    Args:
        run_dir (Path): Output directory for logs and models.
        gs (GaussianIntensityField): The model to train.
        img_tensor (torch.Tensor): Ground truth volume. Expected shape: (C, Z, Y, X).
        cfg (TrainConfig): Training configuration hyperparameters.
        device (torch.device): Compute device.
        enable_logging (bool): If True, generates plots and NIfTI volumes.

    Returns:
        float: The final full-volume reconstruction PSNR (in dB).
    """
    # Start timer at the very beginning to include initialization
    t0 = time.time()

    # Initialize from image happens inside training routine now
    gs.initialize_parameters_from_image(img_tensor)

    spatial_shape = img_tensor.shape[1:]

    grid_axes = [torch.arange(s, device=device) for s in spatial_shape]
    coords_voxel = torch.stack(torch.meshgrid(*grid_axes, indexing='ij'), dim=-1).float()
    coords_voxel_flat = coords_voxel.view(-1, gs.D)

    img_flat = img_tensor.permute(*range(1, img_tensor.ndim), 0).reshape(-1, gs.out_dim)

    num_voxels = coords_voxel_flat.shape[0]
    K = cfg.K
    batch_size = int(num_voxels * cfg.sample_prop)
    print("Batch size: ", batch_size)

    if cfg.knn_type == "knn":
        knn_index = knn.FaissKNN(K, dim=3, device=device)
    elif cfg.knn_type == "voronoi":
        knn_index = knn.VoronoiFAISSKNN(K, dim=3, device=device,
                                        nlist=cfg.voronoi_nlist,
                                        nprobe=cfg.voronoi_nprobe)
    elif cfg.knn_type == "structured_voronoi":
        knn_index = knn.StructuredVoronoiFAISSKNN(K, dim=3,
                                                  range_min=gs.coord_min, range_max=gs.coord_max,
                                                  spacing=(gs.coord_max - gs.coord_min) / torch.tensor(
                                                      cfg.deform_control_points_per_dim),
                                                  nlist=cfg.voronoi_nlist,
                                                  nprobe=cfg.voronoi_nprobe,
                                                  k_candidates=cfg.K_candidates, device=device)
    else:
        raise ValueError(f"Unknown knn type: {cfg.knn_type}")

    knn_index.build(gs.mus)
    knn_cache_flat = knn_index.query_via_chunks(coords_voxel_flat)

    gs = torch.compile(gs, mode="default")

    param_groups = [
        {'params': [gs.mus], 'lr': cfg.lr_mus, 'name': 'mus'},
        {'params': [gs.raw_intensities], 'lr': cfg.lr_intensities, 'name': 'intensities'},
        {'params': [gs.L_diag], 'lr': cfg.lr_L_diag, 'name': 'L_diag'},
        {'params': [gs.L_offdiag], 'lr': cfg.lr_L_offdiag, 'name': 'L_offdiag'}
    ]
    optimizer = torch.optim.Adam(param_groups)

    if cfg.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs, eta_min=cfg.scheduler_eta_min
        )
    elif cfg.scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.scheduler_gamma
        )
    elif cfg.scheduler_type == 'linear':
        max_initial_lr = max([group['lr'] for group in optimizer.param_groups])
        end_factor = cfg.scheduler_eta_min / max_initial_lr
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=cfg.max_epochs
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.scheduler_type}")

    scaler = torch.amp.GradScaler('cuda')
    loss_fn = torch.nn.MSELoss()

    loss = torch.full((1,), 0.0)

    epochs_list = []
    loss_list = []
    psnr_epochs = []
    psnr_list = []
    max_img_val = img_tensor.max().item()

    for epoch in range(cfg.max_epochs):
        if epoch > 0 and epoch % cfg.knn_update_rate == 0:
            index_timer = time.time()
            knn_index.build(gs.mus.data.detach())
            knn_cache_flat = knn_index.query_via_chunks(coords_voxel_flat)
            print(
                f'Epoch {epoch}: Rebuilt index for {coords_voxel_flat.shape[0]} centers in {time.time() - index_timer:.2f}s')

        rand_indices = torch.randperm(num_voxels, device=device)[:batch_size]
        batch_coords = coords_voxel_flat[rand_indices]
        batch_target_vals = img_flat[rand_indices]
        batch_knn = knn_cache_flat[rand_indices]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            pred_intensities = gs(batch_coords, batch_knn)
            loss = loss_fn(pred_intensities, batch_target_vals)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Time: {time.time() - t0:.2f}s | "
                  f"Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Intermediate Logging Block
        if enable_logging:
            epochs_list.append(epoch)
            loss_list.append(loss.item())

            if epoch > 0 and epoch % cfg.logging_rate == 0:
                save_visualization(gs, img_tensor, knn_index, epoch, run_dir, device)

                # Chunked intermediate PSNR calculation
                mse_sum = 0.0
                chunk_size = 500_000
                gs.eval()
                with torch.no_grad():
                    for i in range(0, num_voxels, chunk_size):
                        end = min(i + chunk_size, num_voxels)
                        b_coords = coords_voxel_flat[i:end]
                        b_knn = knn_cache_flat[i:end]
                        b_targets = img_flat[i:end]

                        with torch.amp.autocast('cuda'):
                            pred = gs(b_coords, b_knn)
                            mse_sum += F.mse_loss(pred, b_targets, reduction='sum').item()
                gs.train()

                full_mse = mse_sum / num_voxels
                intermediate_psnr = 10 * math.log10((max_img_val ** 2) / full_mse) if full_mse > 0 else 100.0

                psnr_epochs.append(epoch)
                psnr_list.append(intermediate_psnr)
                print(f"--> Intermediate Full Volume PSNR: {intermediate_psnr:.2f} dB")

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Batch MSE Loss', color='tab:red')
                ax1.plot(epochs_list, loss_list, color='tab:red', alpha=0.8, label='Train Loss')
                ax1.tick_params(axis='y', labelcolor='tab:red')
                ax1.set_yscale('log')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Full Volume PSNR (dB)', color='tab:blue')
                ax2.plot(psnr_epochs, psnr_list, color='tab:blue', marker='o', linewidth=2, label='PSNR')
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                plt.title('Training Metrics over Epochs')
                fig.tight_layout()
                fig.savefig(run_dir / 'metrics_plot.png')
                plt.close(fig)

    print(f"Training complete in {time.time() - t0:.2f}s.")

    # Unconditional Final PSNR Calculation (Guaranteed to execute on optimized weights)
    mse_sum = 0.0
    chunk_size = 500_000
    gs.eval()
    with torch.no_grad():
        for i in range(0, num_voxels, chunk_size):
            end = min(i + chunk_size, num_voxels)
            b_coords = coords_voxel_flat[i:end]
            b_knn = knn_cache_flat[i:end]
            b_targets = img_flat[i:end]

            with torch.amp.autocast('cuda'):
                pred = gs(b_coords, b_knn)
                mse_sum += F.mse_loss(pred, b_targets, reduction='sum').item()
    gs.train()

    full_mse = mse_sum / num_voxels
    final_psnr = 10 * math.log10((max_img_val ** 2) / full_mse) if full_mse > 0 else 100.0

    print(f"--> Final Full Volume PSNR: {final_psnr:.2f} dB")

    if enable_logging:
        save_reconstructed_volume(gs, knn_index, img_tensor.shape, run_dir / "reconstructed_vol.nii.gz", device)
        save_interpolated_volume(gs, knn_index, img_tensor.shape, run_dir / "reconstructed_vol.nii.gz", device)

    return final_psnr


def load_nifti_as_tensor(file_path: Union[str, Path], squeeze_dims: bool = True) -> torch.Tensor:
    img_array = nib.load(str(file_path)).get_fdata()
    img_array = np.transpose(img_array)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)

    if squeeze_dims:
        img_tensor = img_tensor.squeeze()

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    print(f"Loaded volume shape: {img_tensor.shape}")
    return img_tensor


def process_oasis_dataset(dataset_dir: Path, results_dir: Path, device: torch.device,
                          enable_logging: bool = True) -> None:
    splits = ['train', 'val', 'test']

    for split in splits:
        split_in_dir = dataset_dir / split
        split_out_dir = results_dir / split

        if not split_in_dir.exists() or not split_in_dir.is_dir():
            print(f"\n--- Split directory not found, skipping: {split_in_dir} ---")
            continue

        print(f"\n========== Processing Split: {split} ==========")
        split_out_dir.mkdir(parents=True, exist_ok=True)

        for subject_path in tqdm.tqdm(list(split_in_dir.iterdir()), desc=f"Processing {split} subjects"):
            if not subject_path.is_dir():
                continue

            subject_id = subject_path.name
            nii_file = subject_path / "aligned_orig.nii.gz"

            if not nii_file.exists():
                print(f"\nSkipping {subject_id}: 'aligned_orig.nii.gz' not found.")
                continue

            subject_out_dir = split_out_dir / subject_id
            subject_out_dir.mkdir(parents=True, exist_ok=True)

            img_tensor = load_nifti_as_tensor(nii_file)
            img_tensor = img_tensor.to(device)

            cfg = TrainConfig()

            spatial_shape = img_tensor.shape[1:]
            num_channels = img_tensor.shape[0]
            num_voxels = int(np.prod(spatial_shape))

            params_per_gaussian = GaussianIntensityField.params_per_gaussian(len(spatial_shape), num_channels)
            original_params = num_voxels * num_channels
            target_total_params = original_params / cfg.compression_factor
            num_gaussians = max(1, int(target_total_params / params_per_gaussian))

            gs = GaussianIntensityField(
                num_gaussians=num_gaussians,
                in_dim=len(spatial_shape),
                out_dim=num_channels,
                coord_bounds=(0, spatial_shape),
                min_spatial_scale=cfg.min_spatial_scale,
                max_spatial_scale=cfg.max_spatial_scale
            ).to(device)

            print(f"\nSubject: {subject_id}")
            print(f"Image shape: {img_tensor.shape}, num voxels: {num_voxels}")
            print(f"Original Params: {original_params} -> Gaussians Params: {num_gaussians * params_per_gaussian}")
            print(f"Calculated Num gaussians: {gs.num_gaussians} for compression factor {cfg.compression_factor}")

            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = subject_out_dir / run_id

            if enable_logging:
                run_dir.mkdir(exist_ok=True, parents=True)

            final_psnr = train(run_dir, gs, img_tensor, cfg, device, enable_logging=enable_logging)

            print(f"Saving representations for {subject_id}. Final PSNR: {final_psnr:.2f} dB")

            # Save standard representation state dict
            param_save_path = subject_out_dir / "learned_params.pth"
            torch.save(gs.state_dict(), param_save_path)
            print(f"Saved learned parameters to: {param_save_path}")

            # Extract and save PyG Graph Data
            pyg_data = extract_pyg_data(gs, k_graph=cfg.k_graph, psnr=final_psnr)
            pyg_save_path = subject_out_dir / "graph_data.pt"
            torch.save(pyg_data, pyg_save_path)
            print(f"Saved PyG graph data to: {pyg_save_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit Gaussian Intensity Fields to OASIS dataset splits.")
    parser.add_argument("--dataset_path", type=str, default="/vol/miltank/projects/L2R/OASIS_L2R",
                        help="Path to the root of the dataset containing train, val, and test subdirectories.")
    parser.add_argument("--output_path", type=str, default="/vol/miltank/projects/L2R/projects/OASIS_GAUSS_COMPRESS",
                        help="Path to save the output representations, logs, and graphs.")
    parser.add_argument("--enable_logging", action="store_true",
                        help="Enable intermediate logging, plots, and volume visualization saving to speed up training.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")

    enable_logging = args.enable_logging

    process_oasis_dataset(dataset_path, output_path, device, enable_logging=enable_logging)
