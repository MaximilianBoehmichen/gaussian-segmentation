import time
import math
import warnings
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
from torch_geometric.data import Data

import knn
from visualization import save_visualization, save_reconstructed_volume, save_interpolated_volume

# 1. Enable TF32 for massive Ampere+ Tensor Core speedups
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")
torch.set_float32_matmul_precision('high')


@dataclass
class TrainConfig:
    compression_factor: float = 10.0
    max_epochs: int = 201
    logging_rate: int = 20
    sample_prop: float = 0.1
    # Component-specific learning rates
    lr_mus: float = 1e-1
    lr_intensities: float = 1e-1
    lr_L_diag: float = 1e-1
    lr_L_offdiag: float = 1e-1
    # Scheduler configuration
    scheduler_type: str = 'linear'  # Options: 'cosine', 'exponential', 'linear'
    scheduler_eta_min: float = 1e-6
    scheduler_gamma: float = 0.9857  # Used for 'exponential' and 'step'
    knn_type: str = 'structured_voronoi'  # 'knn', 'voronoi', 'structured_voronoi'
    knn_update_rate: int = 50
    K: int = 24
    k_graph: int = 15  # Number of neighbors for the Gaussian-to-Gaussian PyG graph
    K_candidates: int = 300  # For structured component
    deform_control_points_per_dim: Tuple[int, int, int] = (20, 20, 20)  # For structured component
    voronoi_nlist: int = 1024  # Ideally should be ~sqrt(centroids). For voronoi component
    voronoi_nprobe: int = 32  # Should be 1% - 10% of nlist. For voronoi component

    # Geometric bounds
    min_spatial_scale: float = 1.0
    max_spatial_scale: float = 20.0


@torch.compile(mode="default")
def fused_mahalanobis_distance_nd(precision_matrices_nk: torch.Tensor, x_minus_mu_nk: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Mahalanobis distance v^T Lambda v for each (point, Gaussian) pair.
    Compiled to fuse operations in memory, avoiding explicit intermediate allocations.
    Automatically generalizes to any dimensionality D.

    Args:
        precision_matrices_nk (torch.Tensor): Precision matrices Lambda. Expected shape: (N, K, D, D).
        x_minus_mu_nk (torch.Tensor): Centred coordinates (x - mu). Expected shape: (N, K, D).

    Returns:
        torch.Tensor: Squared distances. Expected shape: (N, K).
    """
    M_v = torch.einsum('nkij,nkj->nki', precision_matrices_nk, x_minus_mu_nk)
    return torch.sum(x_minus_mu_nk * M_v, dim=2)


class GaussianIntensityField(nn.Module):
    @classmethod
    def params_per_gaussian(cls, dimensions: int, channels: int) -> int:
        mu_p = dimensions
        L_diag_p = dimensions
        L_offdiag_p = dimensions * (dimensions - 1) // 2
        intensity_p = channels
        return mu_p + L_diag_p + L_offdiag_p + intensity_p

    def __init__(self,
                 num_gaussians: int,
                 in_dim: int,
                 out_dim: int,
                 image_dims: Tuple[int, ...],
                 coord_bounds: Tuple[
                     Union[int, float, tuple, list, torch.Tensor], Union[int, float, tuple, list, torch.Tensor]],
                 min_spatial_scale: float = 0.5,
                 max_spatial_scale: float = 40.0):
        super().__init__()

        if isinstance(coord_bounds[0], (int, float)):
            self.coord_min = torch.tensor([coord_bounds[0]] * in_dim)
        elif isinstance(coord_bounds[0], (list, tuple)):
            self.coord_min = torch.tensor(coord_bounds[0])
        elif isinstance(coord_bounds[0], torch.Tensor):
            self.coord_min = coord_bounds[0].clone()
        else:
            raise TypeError("coord_bounds[0] must be an int, float, list, tuple, or torch.Tensor.")

        if isinstance(coord_bounds[1], (int, float)):
            self.coord_max = torch.tensor([coord_bounds[1]] * in_dim)
        elif isinstance(coord_bounds[1], (list, tuple)):
            self.coord_max = torch.tensor(coord_bounds[1])
        elif isinstance(coord_bounds[1], torch.Tensor):
            self.coord_max = coord_bounds[1].clone()
        else:
            raise TypeError("coord_bounds[1] must be an int, float, list, tuple, or torch.Tensor.")

        self.coord_min = self.coord_min.float()
        self.coord_max = self.coord_max.float()
        self.register_buffer('image_dims', torch.tensor(image_dims, dtype=torch.float32))

        self.D = in_dim
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians
        self.min_spatial_scale = min_spatial_scale
        self.max_spatial_scale = max_spatial_scale

        self.mus = nn.Parameter(
            torch.rand((num_gaussians, self.D)) * (self.coord_max - self.coord_min) + self.coord_min)

        self.L_diag = nn.Parameter(torch.zeros((num_gaussians, self.D)))
        num_off_diagonals = (self.D * (self.D - 1)) // 2
        self.L_offdiag = nn.Parameter(torch.zeros((num_gaussians, num_off_diagonals)))

        self.raw_intensities = nn.Parameter(torch.zeros((num_gaussians, self.out_dim)))

    @property
    def intensities(self) -> torch.Tensor:
        return F.softplus(self.raw_intensities)

    def get_L_matrix(self) -> torch.Tensor:
        max_prec_diag = 1.0 / self.min_spatial_scale
        min_prec_diag = 1.0 / self.max_spatial_scale

        diag_elements = min_prec_diag + (max_prec_diag - min_prec_diag) * torch.sigmoid(self.L_diag)
        diag_L = torch.diag_embed(diag_elements)

        bounded_offdiag = torch.tanh(self.L_offdiag) * max_prec_diag
        off_diag_L = torch.zeros_like(diag_L)
        tril_indices = torch.tril_indices(row=self.D, col=self.D, offset=-1)

        off_diag_L[:, tril_indices[0], tril_indices[1]] = bounded_offdiag

        return diag_L + off_diag_L

    def get_precision_matrices(self) -> torch.Tensor:
        L = self.get_L_matrix()
        return torch.bmm(L, L.transpose(1, 2))

    def forward(self, coords: torch.Tensor, top_k_idcs: torch.Tensor) -> torch.Tensor:
        precisions_all = self.get_precision_matrices()

        mus_local = self.mus[top_k_idcs]
        precisions_local = precisions_all[top_k_idcs]

        intensities_local = self.intensities[top_k_idcs]
        x_minus_mu = coords.unsqueeze(1) - mus_local

        inner_term = fused_mahalanobis_distance_nd(precisions_local, x_minus_mu)

        weights = torch.exp(-0.5 * inner_term)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)

        out = torch.sum(weights.unsqueeze(-1) * intensities_local, dim=1)
        return out

    def initialize_parameters_from_image(self, images: torch.Tensor) -> None:
        print("Running structure-aware and space-filling initialization...")
        start = time.time()
        device = self.mus.device
        spatial_shape = images.shape[1:]
        D_spat = len(spatial_shape)

        if self.D != D_spat:
            raise ValueError(f"Model spatial dimension ({self.D}) does not match image spatial dimensions ({D_spat}).")

        grads = torch.gradient(images, dim=list(range(1, images.ndim)))
        grad_tensor = torch.stack(grads, dim=-1)
        grad_mag = torch.linalg.norm(grad_tensor, dim=-1).mean(dim=0)
        img_mag = images.mean(dim=0)

        grad_prob = grad_mag / (grad_mag.sum() + 1e-8)
        img_prob = img_mag / (img_mag.sum() + 1e-8)

        alpha_grad = 0.7
        prob = alpha_grad * grad_prob + (1.0 - alpha_grad) * img_prob
        prob_flat = prob.flatten()
        prob_flat += 1e-6 / prob_flat.numel()
        prob_flat /= prob_flat.sum()

        sampled_indices = torch.multinomial(prob_flat, num_samples=self.num_gaussians, replacement=False)

        unraveled = torch.unravel_index(sampled_indices, spatial_shape)
        sampled_coords = torch.stack(unraveled, dim=-1).float()
        sampled_coords += self.coord_min.to(device)

        sampled_intensities = images.view(images.shape[0], -1)[:, sampled_indices].T

        grad_tensor_mean = grad_tensor.mean(dim=0)
        sampled_grads = grad_tensor_mean.view(-1, D_spat)[sampled_indices]

        sampled_grad_mag = torch.linalg.norm(sampled_grads, dim=-1, keepdim=True)
        directions = sampled_grads / (sampled_grad_mag + 1e-8)

        max_grad = grad_mag.max() + 1e-8
        norm_mag = (sampled_grad_mag / max_grad).clamp(0.0, 1.0)

        mus_np = sampled_coords.cpu().numpy().astype(np.float32)
        cpu_index = faiss.IndexFlatL2(D_spat)

        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except AttributeError:
            index = cpu_index

        index.add(mus_np)
        D_knn, _ = index.search(mus_np, 4)
        mean_sq_dist = torch.tensor(D_knn[:, 1:]).mean(dim=1, dtype=torch.float32).to(device)
        base_scale = torch.sqrt(mean_sq_dist).clamp(min=self.min_spatial_scale, max=self.max_spatial_scale)
        base_scale = base_scale.unsqueeze(-1)

        scale_along_grad = base_scale * (1.0 - 0.8 * norm_mag)
        scale_ortho = base_scale * (1.0 + 0.5 * norm_mag)

        scale_along_grad = torch.clamp(scale_along_grad, min=self.min_spatial_scale, max=self.max_spatial_scale)
        scale_ortho = torch.clamp(scale_ortho, min=self.min_spatial_scale, max=self.max_spatial_scale)

        prec_along = 1.0 / (scale_along_grad ** 2)
        prec_ortho = 1.0 / (scale_ortho ** 2)

        I = torch.eye(D_spat, device=device).unsqueeze(0)
        d_outer = torch.bmm(directions.unsqueeze(2), directions.unsqueeze(1))

        P_target = prec_along.unsqueeze(-1) * d_outer + prec_ortho.unsqueeze(-1) * (I - d_outer)
        P_target = P_target + I * 1e-6
        L_target = torch.linalg.cholesky(P_target)

        diag_elements_target = torch.diagonal(L_target, dim1=-2, dim2=-1)
        tril_indices = torch.tril_indices(row=D_spat, col=D_spat, offset=-1)

        max_prec_diag = 1.0 / self.min_spatial_scale
        min_prec_diag = 1.0 / self.max_spatial_scale
        eps = 1e-6

        with torch.no_grad():
            self.mus.data = sampled_coords
            self.raw_intensities.data = torch.log(torch.expm1(sampled_intensities.clamp(min=1e-6)))

            diag_clamped = torch.clamp(diag_elements_target, min_prec_diag + eps, max_prec_diag - eps)
            sigmoid_val = (diag_clamped - min_prec_diag) / (max_prec_diag - min_prec_diag)
            self.L_diag.data = torch.logit(sigmoid_val)

            offdiag_target = L_target[:, tril_indices[0], tril_indices[1]]
            offdiag_normalized = offdiag_target / max_prec_diag
            offdiag_clamped = torch.clamp(offdiag_normalized, -1.0 + eps, 1.0 - eps)
            self.L_offdiag.data = torch.atanh(offdiag_clamped)

        print(f"Initialized {self.num_gaussians} Gaussians successfully in {time.time() - start:.2f}s.")


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

    t0 = time.time()
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

    print("Training complete.")

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
    """
    Reads a NIfTI volume image and converts it to a PyTorch tensor.

    Args:
        file_path (Union[str, Path]): The absolute or relative path to the .nii or .nii.gz file.
        squeeze_dims (bool): Whether to drop singleton dimensions initially.

    Returns:
        torch.Tensor: The volumetric image data as a 32-bit float tensor.
            Expected shape: (C, Z, Y, X). For single-channel volumes, C=1.
    """
    img_array = nib.load(str(file_path)).get_fdata()
    img_array = np.transpose(img_array)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)

    if squeeze_dims:
        img_tensor = img_tensor.squeeze()

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    print(f"Loaded volume shape: {img_tensor.shape}")
    return img_tensor


def process_oasis_dataset(oasis_dir: Path, results_dir: Path, device: torch.device,
                          enable_logging: bool = True) -> None:
    for subject_path in oasis_dir.iterdir():
        if not subject_path.is_dir():
            continue

        subject_id = subject_path.name
        nii_file = subject_path / "aligned_orig.nii.gz"

        if not nii_file.exists():
            print(f"Skipping {subject_id}: 'aligned_orig.nii.gz' not found.")
            continue

        subject_out_dir = results_dir / subject_id
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
            image_dims=spatial_shape,
            coord_bounds=(0, spatial_shape),
            min_spatial_scale=cfg.min_spatial_scale,
            max_spatial_scale=cfg.max_spatial_scale
        ).to(device)
        gs.initialize_parameters_from_image(img_tensor)

        print(f"Image shape: {img_tensor.shape}, num voxels: {num_voxels}")
        print(f"Original Params: {original_params} -> Gaussians Params: {num_gaussians * params_per_gaussian}")
        print(f"Calculated Num gaussians: {gs.num_gaussians} for compression factor {cfg.compression_factor}")

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = subject_out_dir / run_id
        run_dir.mkdir(exist_ok=True, parents=True)

        final_psnr = train(run_dir, gs, img_tensor, cfg, device, enable_logging=enable_logging)

        # Print the final PSNR prior to storing representations as requested
        print(f"Saving representations for {subject_id}. Final PSNR: {final_psnr:.2f} dB")

        # Save standard representation state dict
        param_save_path = subject_out_dir / "learned_params.pth"
        torch.save(gs.state_dict(), param_save_path)
        print(f"Saved learned parameters to: {param_save_path}")

        # Extract and save PyG Graph Data
        pyg_data = extract_pyg_data(gs, k_graph=cfg.k_graph, psnr=final_psnr)
        pyg_save_path = subject_out_dir / "graph_data.pt"
        torch.save(pyg_data, pyg_save_path)
        print(f"Saved PyG graph data to: {pyg_save_path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    oasis_path = Path("/home/nil/data/OASIS/")
    results_path = Path(r"./results/")

    if not oasis_path.exists():
        raise FileNotFoundError(f"OASIS dataset path does not exist: {oasis_path}")

    # Set enable_logging=False here to train optimally without drawing or saving volume visualizations
    process_oasis_dataset(oasis_path, results_path, device, enable_logging=False)