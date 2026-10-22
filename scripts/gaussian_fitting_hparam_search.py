import copy
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import argparse
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm

from config.loss import SSIMLoss
from gaussian_fitting import knn
from gaussian_fitting.gaussian_representation import GaussianIntensityField
from gaussian_fitting.oasis_l2r.dataset import OasisL2RDataset
from gaussian_fitting.train import TrainConfig, extract_pyg_data

SAMPLES_PER_SEARCH = 16
END_EPOCHS = [100, 200, 300, 500, 1000]
SEARCH_SPACE: dict[str, list] = {
    "params_percent": [1.0, 2.0, 5.0, 10.0],
    "loss": ["mse", "l1", "ssim"],
    "lr": [1e-1, 1e-2, 1e-3],
    "distance": ["l2", "mahalanobis"],
    "cell_size": [8, 12, 16],
    "knn_update_rate": [10, 50, 100, 500],
}


@dataclass
class SearchSpace:
    params_percent: float
    loss: str
    lr: float
    distance: str
    cell_size: int
    knn_update_rate: int


def get_loss_and_sampling(loss: str) -> tuple[torch.nn.Module, Callable]:
    losses = {
        "mse": (torch.nn.MSELoss(), sample_voxels),
        "l1": (torch.nn.L1Loss(), sample_voxels),
        "ssim": (SSIMLoss(data_range=1.0), sample_patches),
    }

    return losses[loss]


def sample_voxels(spatial_shape, batch_size: int, device: torch.device) -> tuple[torch.Tensor, tuple]:
    """Sample random voxels.

    Args:
        spatial_shape: The shape of the image.
        batch_size: How many voxels to sample.
        device: _.

    Returns:
        The sampled voxels and their shape
    """
    indices = torch.randperm(math.prod(spatial_shape), device=device)[:batch_size]

    return indices, (-1, 1)


def sample_patches(spatial_shape, batch_size: int, device: torch.device, patch_size: int = 48) -> tuple[
    torch.Tensor, tuple]:
    """Sample random contiguous patches.

    Args:
        spatial_shape: The shape of the image.
        batch_size: How many voxels to sample.
        device: _.
        patch_size: Cubed size of patch that is sampled.

    Returns:
        The n sampled voxels patches and their shape

    Note:
        # TODO: add some sort of probability readjustment for edges and voxels close to edges. Currently undersampled. Padding?
    """
    n = max(1, round(batch_size / patch_size ** 3))
    D, H, W = spatial_shape

    starts = torch.stack([torch.randint(0, s - patch_size + 1, (n,), device=device) for s in spatial_shape], dim=-1)
    in_patch = torch.arange(patch_size, device=device)
    in_patch_d, in_patch_h, in_patch_w = torch.meshgrid(in_patch, in_patch, in_patch, indexing="ij")

    indices = ((starts[:, 0, None, None, None] + in_patch_d) * H + (
                starts[:, 1, None, None, None] + in_patch_h)) * W + (starts[:, 2, None, None, None] + in_patch_w)

    return indices.reshape(-1), (n, 1, patch_size, patch_size, patch_size)


def process_search_space(
        train_config: TrainConfig,
        dataset_path: Path = Path("/vol/miltank/projects/L2R/OASIS_L2R"),
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        start: int = 0,
        end: int = 2 ** 31,
):
    """Process the search space in the given range.

    Args:
        train_config (TrainConfig): The generic training parameters.
        dataset_path (Path): Path to the dataset.
        device (torch.device): Device to run on.
        start (int, optional): Start of search space. Defaults to 0.
        end (int, optional): End of search space. Defaults to 2 ** 31.
    """
    Path("./hparams").mkdir(exist_ok=True)

    for i, space in enumerate(itertools.product(*SEARCH_SPACE.values())):
        if i < start:
            continue

        if i == end:
            return

        print(f"Search Space {i}: {str(space)}")

        search_config = SearchSpace(**dict(zip(SEARCH_SPACE, space)))
        statistics = train(train_config, search_config, dataset_path, device)

        output = {
            "hparams": asdict(search_config),
            "statistics": statistics,
        }
        data = json.dumps(output, indent=4)

        with open(f"./hparams/{i}", "w+") as f:
            f.write(data)


def train(
        train_config: TrainConfig,
        search_space: SearchSpace,
        dataset_path: Path,
        device: torch.device,
) -> dict[int, dict]:
    """Trains a subset of the training dataset to aggregate statistics.

    Args:
        search_space (dict[str, float | int | str]): The config used for training.
        dataset_path (Path): Path to the dataset.
        device (torch.device): Device to run on.
    """
    aggregates: dict[int, dict] = {}

    dataset = OasisL2RDataset(dataset_path, "train")

    for i in tqdm(range(SAMPLES_PER_SEARCH)):
        torch.cuda.reset_peak_memory_stats()

        sample = dataset[i]
        image, _, _ = sample["image"].to(device), sample["label"], sample["info"]
        image = image.squeeze()
        image = image.unsqueeze(0)
        image_flat = image.permute(*range(1, image.ndim), 0).reshape(-1, 1)

        total_params = math.prod(image.shape)
        params_per_gaussian = GaussianIntensityField.params_per_gaussian(len(image.shape[1:]), 1)
        num_gaussians = int((total_params * (search_space.params_percent / 100.0)) / params_per_gaussian)

        # start timed training --------------------------------
        torch.cuda.synchronize()
        t0 = time.time()
        representations: list[GaussianIntensityField] = []

        aggregates[i] = {}
        aggregates[i]["t0"] = t0

        gs = GaussianIntensityField(
            num_gaussians=num_gaussians,
            in_dim=len(image.shape[1:]),
            out_dim=1,
            coord_bounds=(0, image.shape[1:]),
            min_spatial_scale=train_config.min_spatial_scale,
            max_spatial_scale=train_config.max_spatial_scale,
        ).to(device)
        gs.initialize_parameters_from_image(image)
        gs = torch.compile(gs, mode="default")

        knn_index = knn.StructuredVoronoiFAISSKNN(
            train_config.K,
            dim=3,
            range_min=gs.coord_min,
            range_max=gs.coord_max,
            spacing=float(search_space.cell_size, ),
            nlist=int(math.sqrt(num_gaussians)),
            nprobe=16,  # I hope that's enough, 6% nlist < nprobe < 20% nlist in the search space
            k_candidates=train_config.K_candidates,
            device=device,
            metric=search_space.distance,
        )

        spatial_shape = image.shape[1:]
        grid_axes = [torch.arange(s, device=device) for s in spatial_shape]
        coords_voxel = torch.stack(
            torch.meshgrid(*grid_axes, indexing="ij"), dim=-1
        ).float()
        coords_voxel_flat = coords_voxel.view(-1, gs.D)

        knn_index.build(gs.mus.detach(), precisions=gs.get_precision_matrices)
        knn_cache_flat = knn_index.query_via_chunks(coords_voxel_flat)

        optimizer = torch.optim.Adam(gs.parameters(), lr=search_space.lr)
        scaler = torch.amp.GradScaler('cuda')
        loss_fn, sample_batch = get_loss_and_sampling(search_space.loss)
        loss = torch.full((1,), 0.0)
        batch_size = int(total_params * train_config.sample_prop)

        for epoch in range(max(END_EPOCHS) + 1):
            if epoch in END_EPOCHS:
                torch.cuda.synchronize()
                representations.append(copy.deepcopy(gs))
                aggregates[i][f"t_{epoch}_raw"] = time.time() - t0
                aggregates[i][f"vram_{epoch}"] = torch.cuda.max_memory_allocated(device)

                if epoch == max(END_EPOCHS):
                    break

            if epoch > 0 and epoch % search_space.knn_update_rate == 0:
                knn_index.build(gs.mus.data.detach(), precisions=gs.get_precision_matrices)
                knn_cache_flat = knn_index.query_via_chunks(coords_voxel_flat)

            indices, shape = sample_batch(spatial_shape, batch_size, device)
            batch_coords = coords_voxel_flat[indices]
            batch_target_vals = image_flat[indices]
            batch_knn = knn_cache_flat[indices]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred_intensities = gs(batch_coords, batch_knn)

            loss = loss_fn(pred_intensities.float().view(shape), batch_target_vals.view(shape))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # end timed training --------------------------------
        # start timed extraction --------------------------------
        for j, epoch in enumerate(END_EPOCHS):
            torch.cuda.synchronize()
            t1 = time.time()

            gs = representations[j]
            _ = extract_pyg_data(gs, train_config.k_graph, psnr=0.0)

            torch.cuda.synchronize()
            aggregates[i][f"t_{epoch}_full"] = aggregates[i].get(f"t_{epoch}_raw", 0.0) + time.time() - t1

        # end timed extraction --------------------------------
        # (unrelated to training) metric calculation not timed
        for j, epoch in enumerate(END_EPOCHS):
            mse_sum = 0.0
            chunk_size = 500_000
            preds: list[torch.Tensor] = []

            gs = representations[j]
            gs.eval()
            knn_index.build(gs.mus.detach(), precisions=gs.get_precision_matrices())
            knn_cache_flat = knn_index.query_via_chunks(coords_voxel_flat)

            with torch.no_grad():
                for k in range(0, total_params, chunk_size):
                    end = min(k + chunk_size, total_params)
                    b_coords = coords_voxel_flat[k:end]
                    b_knn = knn_cache_flat[k:end]
                    b_targets = image_flat[k:end]

                    with torch.amp.autocast('cuda'):
                        pred = gs(b_coords, b_knn)
                        mse_sum += F.mse_loss(pred, b_targets, reduction='sum').item()

                        preds.append(pred.float())

                pred_volume = torch.cat(preds).view(1, 1, *spatial_shape)
                full_mse = mse_sum / total_params
                final_psnr = 10 * math.log10((image.max().item() ** 2) / full_mse) if full_mse > 0 else 100.0
                final_ssim = ssim(pred_volume, image.unsqueeze(0), data_range=1.0).item()

                aggregates[i][f"mse_{epoch}"] = full_mse
                aggregates[i][f"psnr_{epoch}"] = final_psnr
                aggregates[i][f"ssim_{epoch}"] = final_ssim

        del (
            gs, representations, knn_index, knn_cache_flat, optimizer, scaler, coords_voxel, coords_voxel_flat, preds,
            pred_volume, image, image_flat, batch_coords, batch_knn, pred_intensities, loss, pred, b_coords, b_knn,
            b_targets
        )

    return aggregates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit Gaussian Intensity Fields to OASIS dataset splits.")
    parser.add_argument("--dataset_path", type=str, default="/vol/miltank/projects/L2R/OASIS_L2R",
                        help="Path to the root of the dataset containing train, val, and test subdirectories.")
    parser.add_argument("--start", type=int, default=0, help="At which search space iteration to continue.")
    parser.add_argument("--end", type=int, help="At which search space iteration to end.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")

    train_config = TrainConfig()

    process_search_space(
        train_config,
        dataset_path,
        device,
        args.start,
        args.end,
    )
