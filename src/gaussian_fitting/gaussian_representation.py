"""
Defines GaussianIntensityField, an n-dimensional mixture-of-Gaussians model that represents
a volumetric intensity field (e.g. an MRI scan) as a compact set of anisotropic Gaussians
with learnable positions, covariances, and per-channel intensities, including structure-aware
initialization from an input image.
"""

import time
from typing import Tuple, Union, Optional

import faiss
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



@torch.compile(mode="default")
def fused_mahalanobis_distance_nd(precision_matrices_nk: torch.Tensor, x_minus_mu_nk: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Mahalanobis distance v^T Lambda v for each (point, Gaussian) pair.
    Compiled to fuse operations in memory, avoiding explicit intermediate allocations.
    Automatically generalizes to any dimensionality D.
    """
    M_v = torch.einsum('nkij,nkj->nki', precision_matrices_nk, x_minus_mu_nk)
    return torch.sum(x_minus_mu_nk * M_v, dim=2)


class GaussianIntensityField(nn.Module):
    """"
    Gaussian representations generalizable to n-dimensional intensity fields.
    Uses Cholesky L-matrix lower triangular matrix formulation.
    Handles any number of channels (eg. RGB).
    """
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
                 coord_bounds: Tuple[
                     Union[int, float, tuple, list, torch.Tensor], Union[int, float, tuple, list, torch.Tensor]],
                 min_spatial_scale: float = 0.5,
                 max_spatial_scale: float = 40.0):
        """
        Initializes the n-dimensional Gaussian Intensity Field.

        Args:
            num_gaussians (int): The total number of Gaussians used to represent the field.
            in_dim (int): The spatial dimensionality of the coordinate space (e.g., 2 for 2D images, 3 for 3D volumes).
            out_dim (int): The dimensionality of the output features (e.g., 1 for grayscale intensity, 3 for RGB).
            coord_bounds (Tuple[Union[int, float, tuple, list, torch.Tensor], Union[int, float, tuple, list, torch.Tensor]]):
                The (minimum, maximum) bounds for the spatial coordinates of the Gaussians (`mus`).
                If passing `torch.Tensor`s, the expected shape for each tensor is `(in_dim,)`.
            min_spatial_scale (float, optional): The absolute minimum spatial extent (standard deviation proxy)
                allowed for any Gaussian. Enforced algebraically to prevent Gaussians from collapsing into
                infinitely thin points or needles (infinite precision). Defaults to 0.5.
            max_spatial_scale (float, optional): The absolute maximum spatial extent allowed for any Gaussian.
                Enforced algebraically to prevent Gaussians from expanding infinitely and washing out
                details (zero precision). Defaults to 40.0.
        """
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

        # Softplus unconstrained formulation
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

        mus_local = self.mus[top_k_idcs]  # (N, K, D)
        precisions_local = precisions_all[top_k_idcs]  # (N, K, D, D)

        intensities_local = self.intensities[top_k_idcs]  # (N, K, out_dim)
        x_minus_mu = coords.unsqueeze(1) - mus_local  # (N, K, D)

        inner_term = fused_mahalanobis_distance_nd(precisions_local, x_minus_mu)  # (N, K)

        weights = torch.exp(-0.5 * inner_term)  # (N, K)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)

        out = torch.sum(weights.unsqueeze(-1) * intensities_local, dim=1)  # (N, out_dim)
        return out

    def initialize_parameters_from_image(self, image: torch.Tensor) -> None:
        print("Running structure-aware and space-filling initialization...")
        start = time.time()
        device = self.mus.device
        spatial_shape = image.shape[1:]
        D_spat = len(spatial_shape)

        if self.D != D_spat:
            raise ValueError(f"Model spatial dimension ({self.D}) does not match image spatial dimensions ({D_spat}).")

        # 1. Gradient and Intensity Maps
        grads = torch.gradient(image, dim=list(range(1, image.ndim)))
        grad_tensor = torch.stack(grads, dim=-1)
        grad_mag = torch.linalg.norm(grad_tensor, dim=-1).mean(dim=0)
        img_mag = image.mean(dim=0)

        # 2. Mixed Probability Sampling
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

        sampled_intensities = image.view(image.shape[0], -1)[:, sampled_indices].T

        grad_tensor_mean = grad_tensor.mean(dim=0)
        sampled_grads = grad_tensor_mean.view(-1, D_spat)[sampled_indices]

        sampled_grad_mag = torch.linalg.norm(sampled_grads, dim=-1, keepdim=True)
        directions = sampled_grads / (sampled_grad_mag + 1e-8)

        max_grad = grad_mag.max() + 1e-8
        norm_mag = (sampled_grad_mag / max_grad).clamp(0.0, 1.0)

        # 3. K-Nearest Neighbors for Adaptive Base Scaling
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
        base_scale = base_scale.unsqueeze(-1)  # (N, 1)

        # 4. Adaptive Anisotropy based on Structure
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
