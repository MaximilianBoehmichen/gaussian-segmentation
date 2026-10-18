import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn

from config.types import CoordsBatch, FloatIndices, IntIndices, MuTensor, RotationMatrix, VolumeTensor

type ScalingsInvTensor = Float[torch.Tensor, "num_gaussians 3"]
type QuaternionsTensor = Float[torch.Tensor, "num_gaussians 4"]
type ColorTensor = Float[torch.Tensor, "num_gaussians"]

type TopKIndices = Int[torch.Tensor, "N, KNN"]
type ColorPredictions = Float[torch.Tensor, "N 1"]


class GaussianRepresentation3D(nn.Module):
    """Learnable Gaussian 3D Representation of a Volume."""

    SCALING_STD_OF_MEAN_INIT: float = 0.25
    SCALING_MIN_INIT: float = 1.00

    spatial_dims: Int[torch.Tensor, "3"]
    min_color: float
    max_color: float

    ground_truth: VolumeTensor

    mu: MuTensor
    scalings_inv: ScalingsInvTensor
    quaternions: QuaternionsTensor
    colors: ColorTensor

    @jaxtyped(typechecker=beartype)
    def __init__(
            self,
            volume: VolumeTensor,
            num_gaussians: int,
            gradient_init_percentage: float,
    ):
        """Initialize a 3D Gaussian Representation of a Volume.

        Args:
            num_gaussians (int): Number of Gaussians to fit.
            gradient_init_percentage (float): Percentage of Gaussians to initialize to from the image gradient.
        """
        super().__init__()

        device = volume.device

        min_color = 0.0
        max_color = 1.0

        # scalings = sigma^2
        scaling_mean = max(2.0, max(volume.shape) * 0.05)
        scaling_std = scaling_mean * self.SCALING_STD_OF_MEAN_INIT
        noise = torch.randn((num_gaussians, 3)) * scaling_std
        scalings = (torch.full((num_gaussians, 3), scaling_mean) + noise).clamp(min=self.SCALING_MIN_INIT)

        self.ground_truth = volume
        self.spatial_dims = torch.tensor(volume.shape).to(device)
        self.min_color = min_color
        self.max_color = max_color
        self.mu = nn.Parameter(torch.rand((num_gaussians, 3)) * (self.spatial_dims - 1)).to(device)
        self.scalings_inv = nn.Parameter(torch.log(torch.sqrt(1 / scalings))).to(device)
        self.quaternions = nn.Parameter(torch.zeros((num_gaussians, 4))).to(device)
        self.quaternions.data[:, 0] = 1.0  # (w=1, x=0, y=0, z=0) quaternions

        colors = self.color_from_grid(self.mu)
        self.colors = -torch.log(1.0 / colors - 1.0)

        self.from_image_gradient(gradient_init_percentage)

    def color_from_grid(self, indices: IntIndices | FloatIndices) -> ColorTensor:
        """Gives the color of the closest voxel by L1.

        Args:
            indices (IntVector | FloatVector): The mu position of each Gaussian.

        Returns:
            The color of the closest voxel.
        """
        indices = indices.round().type(torch.int64)

        return (self.ground_truth[indices.unbind(dim=1)] - self.min_color) / (self.max_color - self.min_color)

    def from_image_gradient(self, gradient_init_percentage: float):
        """Initialize the given percentage of Gaussians by image gradient.

        Args:
            gradient_init_percentage (float): Percentage of Gaussians to initialize to from the image gradient.
        """
        if gradient_init_percentage == 0.0:
            return

        k = int(self.num_gaussian * gradient_init_percentage)

        # gradients from image
        grads = torch.gradient(self.ground_truth)
        grad_tensor = torch.stack(grads, dim=-1)
        grad_mag = torch.sqrt(torch.sum(grad_tensor ** 2, dim=-1))
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)

        grad_sum = grad_mag.sum()
        grad_prob = (grad_mag / grad_sum) if grad_sum > 0 else (torch.ones_like(grad_mag) / grad_mag.numel())

        pixels_per_gaussian = self.ground_truth.numel() / self.num_gaussian
        k_size = max(3, int(pixels_per_gaussian ** (1 / 3)))

        if k_size % 2 == 0:
            k_size += 1

        pad = k_size // 2
        grad_view = grad_prob.view(1, 1, *self.ground_truth.shape)
        grad_max = torch.nn.functional.max_pool3d(grad_view, kernel_size=k_size, stride=1, padding=pad)
        is_peak = (grad_view == grad_max) & (grad_view > 1e-6)  # local max in kernel area and not flat
        grad_prob_nms = torch.where(is_peak.view_as(grad_prob), grad_prob, grad_prob * 1e-4)

        p_init_flat = grad_prob_nms.flatten().double()
        p_init_flat = torch.nan_to_num(p_init_flat, nan=0.0, posinf=0.0, neginf=0.0)
        p_init_flat = p_init_flat + p_init_flat.max() * 1e-12
        p_init_flat /= p_init_flat.sum()
        sampled_indices = torch.multinomial(p_init_flat, num_samples=k, replacement=False)

        grid_axes = [torch.arange(s, device=self.device, dtype=torch.float32) for s in self.ground_truth.shape]
        grid = torch.stack(torch.meshgrid(*grid_axes, indexing='ij'), dim=-1).view(-1, 3)
        new_mus = grid[sampled_indices]

        grad_flat = grad_tensor.view(-1, 3)
        sampled_grads = grad_flat[sampled_indices]
        sampled_dirs = torch.nn.functional.normalize(sampled_grads, p=2, dim=-1, eps=1e-8)

        with torch.no_grad():
            # shortest-arc quaternion
            v0, v1, v2 = sampled_dirs[:, 0], sampled_dirs[:, 1], sampled_dirs[:, 2]
            q = torch.zeros(self.num_gaussian, 4, device=self.device)
            q[:, 0] = 1.0 + v0
            q[:, 1] = 0.0
            q[:, 2] = -v2
            q[:, 3] = v1

            degenerate_mask = (q[:, 0] < 1e-6)
            q[degenerate_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            new_rotations = torch.nn.functional.normalize(q, p=2, dim=-1)

            scaling_mean = max(1., torch.tensor(self.ground_truth.shape).amax().item() * 0.05)
            sampled_grad_mag = grad_mag_norm.flatten()[sampled_indices]

            new_scalings = torch.zeros((k, 3), device=self.device)

            perp_scale = scaling_mean / (1.0 + 8.0 * sampled_grad_mag)
            par_scale = scaling_mean * (1.0 + 7.0 * sampled_grad_mag)

            new_scalings[:, 0] = perp_scale
            new_scalings[:, 1:] = par_scale.unsqueeze(-1)

            noise = torch.randn_like(new_scalings) * (scaling_mean * 0.02)
            new_scalings = (new_scalings + noise).clamp(min=0.15)
            new_scalings_inv = torch.log(torch.sqrt(1 / new_scalings))

            img_flat = self.ground_truth.flatten()
            sampled_colors = img_flat[sampled_indices]
            new_colors = -torch.log(1.0 / sampled_colors - 1.0)

            # overwrite the k differently initialized gaussians
            self.mu.data[-k:] = new_mus
            self.quaternions.data[-k:] = new_rotations
            self.scalings_inv.data[-k:] = new_scalings_inv
            self.colors.data[-k:] = new_colors

    def compute_sigma_inverse(self):
        r"""Computes :math:`\Sigma^{-1}`

        Returns:
            :math:`\Sigma^{-1}
        """
        r, rt = self.rotation_matrices()
        s_inv = torch.diag_embed(self.scalings_inv ** 2)

        return torch.bmm(r, torch.bmm(s_inv, rt))

    def rotation_matrices(self) -> tuple[RotationMatrix, RotationMatrix]:
        """Builds the 3D rotation matrices from quaternions.

        Returns:
            R, R^(-1)=R^T
        """
        quaternions = self.quaternions
        q_norm = torch.nn.functional.normalize(quaternions, p=2, dim=1)

        w, x, y, z = q_norm.unbind(1)

        k = q_norm.shape[0]
        r = torch.empty((k, 3, 3), device=q_norm.device, dtype=q_norm.dtype)

        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
        r[:, 0, 1] = 2.0 * (xy - wz)
        r[:, 0, 2] = 2.0 * (xz + wy)

        r[:, 1, 0] = 2.0 * (xy + wz)
        r[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
        r[:, 1, 2] = 2.0 * (yz - wx)

        r[:, 2, 0] = 2.0 * (xz - wy)
        r[:, 2, 1] = 2.0 * (yz + wx)
        r[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)

        return r, r.mT

    @property
    def device(self) -> torch.Device:
        return self.mu.device

    @property
    def num_gaussian(self) -> int:
        return int(self.mu.shape[0])

    def forward(
            self,
            coords: CoordsBatch,
            top_k_indices: TopKIndices,
    ) -> ColorPredictions:
        """Forward pass.

        Args:
            coords (CoordsBatch): The coordinates of this batch.
            top_k_indices (TopKIndices): The k nearest neighbor Gaussians per voxel of ``coords``.

        Returns:
            The predicted color per voxel of ``coords``.
        """
        voxel_knn_mus = self.mu[top_k_indices]
        x_minus_mu = coords.unsqueeze(1) - voxel_knn_mus

        sigmas_inv = self.compute_sigma_inverse()[top_k_indices]

        # m_v = sigma_inv @ (x - mu)
        m_v = torch.einsum("nkij,nkj->nki", sigmas_inv, x_minus_mu)

        inner_term = torch.sum(x_minus_mu * m_v, dim=2)
        gaussian_weights = torch.exp(-0.5 * inner_term)
        weights_normalized = gaussian_weights / (torch.sum(gaussian_weights, dim=1, keepdim=True) + 1e-8)

        colors = self.colors[top_k_indices]
        colors = torch.sigmoid(colors) * (self.max_color - self.min_color) + self.min_color

        return torch.sum(weights_normalized * colors, dim=1)
