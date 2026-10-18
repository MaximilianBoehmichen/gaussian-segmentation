import time
from typing import Callable, Literal

import kornia
from beartype import beartype
from jaxtyping import Float, Int, Int64, jaxtyped
import faiss
import faiss.contrib.torch_utils
import torch

from config.config import GSTrainingConfig
from config.types import CoordsBatch, MuTensor, OasisVolumeTensor
from gaussian_fitting.gaussian_representation import GaussianRepresentation3D

type LossFn = Callable[[Float[torch.Tensor, "*batch"], Float[torch.Tensor, "*batch"]], Float[torch.Tensor, ""]]
FAISS_GPU_MAX_K = 2048


def build_faiss_index():
    cpu_index = faiss.IndexFlatL2(3)

    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index), res

    return cpu_index, None


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def compute_top_k(
        gs: GaussianRepresentation3D,
        coords: CoordsBatch,
        faiss_index,
        k: int,
        metric: Literal["l2", "mahalanobis"] = "l2",
        candidate_factor: int = 4,
        chunk_size: int = 32768
) -> Int[torch.Tensor, "p k_out"]:
    mus: MuTensor = gs.mu.detach()
    n_gauss = mus.shape[0]
    k = min(k, n_gauss)

    faiss_index.reset()
    faiss_index.add(mus.contiguous())

    if metric == "l2":
        _, top_k = faiss_index.search(coords, k)

        return top_k

    k_cand = min(max(k * candidate_factor, k), n_gauss, FAISS_GPU_MAX_K)
    _, cand = faiss_index.search(coords, k_cand)
    cand: Int[torch.Tensor, "p k_cand"] = cand.to(coords.device)
    sigmas_inv: Float[torch.Tensor, "g d d"] = gs.compute_sigma_inverse().detach()
    top_k = torch.empty((coords.shape[0], k), dtype=cand.dtype, device=coords.device)

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        c: Int[torch.Tensor, "b k_cand"] = cand[start:end]
        x_minus_mu: Float[torch.Tensor, "b k_cand d"] = coords[start:end, None, :] - mus[c]
        m_v: Float[torch.Tensor, "b k_cand d"] = torch.einsum("nkij,nkj->nki", sigmas_inv[c], x_minus_mu)
        d2: Float[torch.Tensor, "b k_cand"] = torch.sum(x_minus_mu * m_v, dim=2)
        best: Int[torch.Tensor, "b k_out"] = torch.topk(d2, k, dim=1, largest=False).indices
        top_k[start:end] = torch.gather(c, 1, best)

    return top_k


def compute_regularization_losses(
        gs_model: GaussianRepresentation3D,
        spatial_dims: torch.Tensor,
        lambda_pos=0.0, lambda_scale=0.0,
        min_scale=None, max_scale=None
):
    """
    Penalizes Gaussians that drift outside the image bounds
    or shrink/grow beyond stable scaling thresholds.
    """
    # 1. Position Regularization
    if lambda_pos > 0.0:
        # Penalize any mu value that exceeds image bounds
        # relu(x - 1.0) is >0 only when x > 1.0
        out_of_bounds_pos = torch.nn.functional.relu(torch.abs(gs_model.mu - spatial_dims / 2) - spatial_dims / 2)
        loss_pos = out_of_bounds_pos.mean()
        loss_pos *= lambda_pos
    else:
        loss_pos = torch.tensor([0.0], dtype=gs_model.mu.dtype, device=gs_model.mu.device)

    # 2. Scale Regularization
    if min_scale is None:
        min_scale = 0.5  # Half a voxel
    if max_scale is None:
        max_scale = max(spatial_dims) / 2  # Proportion of the image
    if lambda_scale > 0.0:
        # Penalize if it gets too small (too wide) or too large (too narrow)
        too_wide = torch.nn.functional.relu(
            gs_model.scalings_inv - max_scale)  # If smaller than max_scale -> negative -> no gradient
        too_narrow = torch.nn.functional.relu(
            min_scale - gs_model.scalings_inv)  # If larger than min_scale -> negative -> no gradient
        loss_scale = (too_wide + too_narrow).mean()
        loss_scale *= lambda_scale
    else:
        loss_scale = torch.tensor([0.0], dtype=gs_model.mu.dtype, device=gs_model.mu.device)

    return loss_pos, loss_scale


@torch.no_grad()
def render_volume(
        gs: GaussianRepresentation3D,
        coords: CoordsBatch,
        top_k: Int[torch.Tensor, "p k"],
        chunk_size: int = 2 ** 16,
) -> torch.Tensor:
    pred = torch.empty(coords.shape[0], dtype=gs.ground_truth.dtype, device=coords.device)

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        pred[start:end] = gs.forward(coords[start:end], top_k[start:end])

    return pred


def train_gs(
        image: OasisVolumeTensor,
        params: GSTrainingConfig,
        loss_fn: LossFn,
        debug: bool = False
):
    gs = GaussianRepresentation3D(
        image,
        params.num_gaussians,
        params.gradient_init_percentage
    )
    device = image.device

    if loss_fn is None:
        loss_fn = torch.nn.L1Loss()

    shape_tensor = torch.tensor(image.shape, device=image.device)
    knn_metric = params.knn_metric
    knn_candidate_factor: int = getattr(params, "knn_candidate_factor", 4)

    D = 3
    device = image.device
    grid_axes = [torch.arange(s, device=device) for s in image.shape]
    coords_voxel_: Float[torch.Tensor, "p d"] = (
        torch.stack(torch.meshgrid(*grid_axes, indexing='ij'), dim=-1).view(-1, D).float())
    values_: Float[torch.Tensor, "p"] = image.flatten()
    total_coords = coords_voxel_.shape[0]
    batch_size = int(total_coords * params.point_sample_prop)

    optimizer = torch.optim.AdamW(gs.parameters(), lr=params.learning_rate_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.max_epochs, eta_min=params.learning_rate_end)

    faiss_index, _faiss_res = build_faiss_index()
    top_k_idcs: Int[torch.Tensor, "p k"] = compute_top_k(
        gs, coords_voxel_, faiss_index, params.k_neighborhood,
        metric=knn_metric, candidate_factor=knn_candidate_factor)

    t0 = time.time()
    progress_ims: list[torch.Tensor] = []
    progress_ims_epochs: list[int] = []
    progress_ims_psnrs: list[float] = []
    loss, loss_rec, loss_scale, loss_pos = [torch.tensor([torch.inf])] * 4
    epoch_iter = range(params.max_epochs)

    if debug:
        from tqdm import tqdm
        epoch_iter = tqdm(epoch_iter, desc="Optimizing", leave=True)

    for epoch in epoch_iter:
        t0 = time.time()

        rand_indices: Int64[torch.Tensor, "b"] = torch.randperm(total_coords, device=device)[:batch_size]
        batch_coords: Float[torch.Tensor, "b d"] = coords_voxel_[rand_indices]
        batch_values: Float[torch.Tensor, "b"] = values_[rand_indices]
        batch_top_k: Int64[torch.Tensor, "b k"] = top_k_idcs[rand_indices]

        optimizer.zero_grad()
        preds: Float[torch.Tensor, "b"] = gs.forward(batch_coords, batch_top_k)
        loss_rec = loss_fn(preds, batch_values)
        loss_pos, loss_scale = compute_regularization_losses(
            gs, shape_tensor, lambda_pos=params.reg_weight_pos, lambda_scale=params.reg_weight_scale)
        loss = loss_rec + loss_pos + loss_scale

        loss.backward()
        optimizer.step()
        scheduler.step()

        if debug:
            t1 = time.time()
            pred = render_volume(gs, coords_voxel_, top_k_idcs).reshape(image.shape)
            psnr = kornia.metrics.psnr(image[None], pred[None], max_val=gs.max_color).item()
            epoch_iter.set_postfix(epoch=epoch, loss=f"{loss.item():.6f}", psnr=f"{psnr:.2f}", time=f"{t1 - t0:.2f}")
            tqdm.write(str(epoch_iter))

        if (epoch + 1) % params.knn_update_rate == 0:
            with torch.no_grad():
                top_k_idcs = compute_top_k(
                    gs, coords_voxel_, faiss_index, params.k_neighborhood,
                    metric=knn_metric, candidate_factor=knn_candidate_factor)

    pred = render_volume(gs, coords_voxel_, top_k_idcs).reshape(image.shape)
    psnr = kornia.metrics.psnr(image[None], pred[None], max_val=gs.max_color).item()

    return gs, pred, coords_voxel_, top_k_idcs, psnr
