"""Fit Gaussian intensity fields to OASIS volumes with a fixed config.

For every sample and repetition one file is written to
``<out_path>/<split>/<sample_id>/<repetition>.pt``. Training runs once up to
``max(epochs)`` and a snapshot is added to that file at each epoch in ``epochs``.

File layout::

    {
        "config": dict,             # TrainingConfig as plain dict
        "split", "sample_id", "repetition", "seed",
        "model_args": dict,         # kwargs to rebuild GaussianIntensityField
        "mask": uint8 tensor,       # label volume from OasisL2RDataset
        "intensity_min", "intensity_max": float,  # original range; the image is
                                    # min-max scaled to [0, 1] before training
        "epochs": {
            <epoch>: {
                "state_dict": dict,            # Gaussian parameters (CPU)
                "train_time": float,           # seconds, cumulative, eval excluded
                "metrics": {"psnr", "ssim", "mse", "l1"},
                "previews": [png0, png1, png2],  # encoded PNG bytes (uint8 tensor),
                                                 # middle slice along dim 0/1/2,
                                                 # left: ground truth, right: prediction
            },
            ...
        },
    }
"""

import argparse
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.io import encode_png
from tqdm import tqdm

from config.config import TrainingConfig
from gaussian_fitting import knn
from gaussian_fitting.gaussian_representation import GaussianIntensityField
from gaussian_fitting.oasis_l2r.dataset import OasisL2RDataset
from gaussian_fitting.sampling import get_loss_and_sampling
from gaussian_fitting.train import TrainConfig

# Same as the hparam search (set there via the gaussian_fitting.train import).
torch.set_float32_matmul_precision("high")

EVAL_CHUNK_SIZE = 500_000
SEARCH_DEFAULTS = TrainConfig()


def config_to_dict(config: TrainingConfig) -> dict:
    """Converts the config into a dict without Path objects."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


def build_knn_index(
        gs: GaussianIntensityField, config: TrainingConfig, device: torch.device
) -> knn.StructuredVoronoiFAISSKNN:
    return knn.StructuredVoronoiFAISSKNN(
        config.k,
        dim=3,
        range_min=gs.coord_min,
        range_max=gs.coord_max,
        spacing=float(config.cell_size),
        nlist=int(math.sqrt(gs.num_gaussians)),
        nprobe=config.nprobe,
        k_candidates=config.k_candidates,
        device=device,
        metric=config.distance,
    )


@torch.no_grad()
def evaluate(
        model: GaussianIntensityField,
        knn_index: knn.StructuredVoronoiFAISSKNN,
        image: torch.Tensor,
        coords_flat: torch.Tensor,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    """Full-volume metrics and middle-slice previews after a fresh KNN rebuild.

    The fresh KNN cache is local, so the training cache and schedule stay untouched.
    Uses the uncompiled model: calling the compiled one with eval-sized chunks would
    recompile it with dynamic shapes and change the kernels used for training.

    Args:
        model: The uncompiled model.
        knn_index: The KNN index; rebuilt in place.
        image: Ground truth of shape (1, D, H, W).
        coords_flat: All voxel coordinates of shape (D*H*W, 3).

    Returns:
        The metrics and one encoded PNG per spatial dim.
    """
    spatial_shape = image.shape[1:]
    image_flat = image.reshape(-1, 1)

    knn_index.build(model.mus.detach(), precisions=model.get_precision_matrices)
    knn_cache = knn_index.query_via_chunks(coords_flat)

    preds = []
    for i in range(0, coords_flat.shape[0], EVAL_CHUNK_SIZE):
        end = min(i + EVAL_CHUNK_SIZE, coords_flat.shape[0])

        with torch.amp.autocast("cuda"):
            preds.append(model(coords_flat[i:end], knn_cache[i:end]).float())

    del knn_cache

    pred_flat = torch.cat(preds)
    pred_volume = pred_flat.view(1, 1, *spatial_shape)

    mse = F.mse_loss(pred_flat, image_flat).item()
    l1 = F.l1_loss(pred_flat, image_flat).item()
    psnr = 10 * math.log10(image.max().item() ** 2 / mse) if mse > 0 else 100.0
    ssim_value = ssim(pred_volume, image.unsqueeze(0), data_range=1.0).item()

    metrics = {"psnr": psnr, "ssim": ssim_value, "mse": mse, "l1": l1}

    return metrics, middle_slice_previews(image[0], pred_volume[0, 0])


def middle_slice_previews(gt: torch.Tensor, pred: torch.Tensor) -> list[torch.Tensor]:
    """Encodes the middle slice along each dim as PNG (ground truth | prediction)."""
    max_val = gt.max().clamp(min=1e-8)
    previews = []

    for dim in range(gt.ndim):
        mid = gt.shape[dim] // 2
        side_by_side = torch.cat([gt.select(dim, mid), pred.select(dim, mid)], dim=-1)

        img = (side_by_side / max_val).clamp(0, 1).mul(255).round().to(torch.uint8)
        previews.append(encode_png(img.unsqueeze(0).cpu()))

    return previews


def min_max_scale(sample: dict) -> dict:
    """Scales image intensities to [0, 1] and keeps the original range in info."""
    image = sample["image"]
    i_min, i_max = image.min(), image.max()
    sample["image"] = (image - i_min) / (i_max - i_min).clamp(min=1e-8)
    sample["info"]["intensity_min"] = i_min.item()
    sample["info"]["intensity_max"] = i_max.item()

    return sample


def save_atomic(obj: dict, path: Path) -> None:
    """Writes to a temp file first so a crash never leaves a broken file."""
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def train_sample(
        config: TrainingConfig,
        image: torch.Tensor,
        output: dict,
        out_file: Path,
        device: torch.device,
) -> None:
    """Trains one repetition on one volume and saves a snapshot at each config epoch.

    Args:
        config: The training config.
        image: Ground truth of shape (1, D, H, W) on device.
        output: The dict that is saved; snapshots are added to ``output["epochs"]``.
        out_file: Where to save ``output``.
        device: _.
    """
    epochs = sorted(set(config.epochs))
    spatial_shape = image.shape[1:]
    total_voxels = math.prod(spatial_shape)
    image_flat = image.reshape(-1, 1)

    grid_axes = [torch.arange(s, device=device) for s in spatial_shape]
    coords_flat = (
        torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=-1)
        .float()
        .view(-1, len(spatial_shape))
    )

    params_per_gaussian = GaussianIntensityField.params_per_gaussian(
        len(spatial_shape), 1
    )
    num_gaussians = int(total_voxels * (config.params / 100.0) / params_per_gaussian)

    model_args = {
        "num_gaussians": num_gaussians,
        "in_dim": len(spatial_shape),
        "out_dim": 1,
        "coord_bounds": (0, tuple(spatial_shape)),
        "min_spatial_scale": config.min_spatial_scale,
        "max_spatial_scale": config.max_spatial_scale,
    }
    output["model_args"] = model_args

    # start timed training --------------------------------
    torch.cuda.synchronize()
    t_start = time.time()

    model = GaussianIntensityField(**model_args).to(device)
    model.initialize_parameters_from_image(image)
    gs = torch.compile(model, mode="default")

    knn_index = build_knn_index(model, config, device)
    knn_index.build(model.mus.detach(), precisions=model.get_precision_matrices)
    knn_cache_flat = knn_index.query_via_chunks(coords_flat)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn, sample_batch = get_loss_and_sampling(config.loss)
    batch_size = int(total_voxels * config.sample_prop)

    train_time = 0.0

    for epoch in range(max(epochs) + 1):
        if epoch in epochs:
            torch.cuda.synchronize()
            train_time += time.time() - t_start
            # end timed training (paused) --------------------------------

            metrics, previews = evaluate(model, knn_index, image, coords_flat)
            output["epochs"][epoch] = {
                "state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "train_time": train_time,
                "metrics": metrics,
                "previews": previews,
            }
            save_atomic(output, out_file)
            tqdm.write(
                f"  epoch {epoch}: {train_time:.1f}s | "
                + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            )

            if epoch == max(epochs):
                break

            torch.cuda.synchronize()
            t_start = time.time()
            # resume timed training --------------------------------

        if epoch > 0 and epoch % config.knn_rate == 0:
            knn_index.build(model.mus.detach(), precisions=model.get_precision_matrices)
            knn_cache_flat = knn_index.query_via_chunks(coords_flat)

        indices, shape = sample_batch(spatial_shape, batch_size, device)
        batch_coords = coords_flat[indices]
        batch_targets = image_flat[indices]
        batch_knn = knn_cache_flat[indices]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            pred = gs(batch_coords, batch_knn)

        loss = loss_fn(pred.float().view(shape), batch_targets.view(shape))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def process_config(config: TrainingConfig, device: torch.device) -> None:
    """Run the training of the config for all splits, samples and repetitions."""
    for split in config.split:
        dataset = OasisL2RDataset(config.dataset_path, split, transforms=min_max_scale)
        indices = range(config.start, min(config.end, len(dataset)))

        for i in tqdm(indices, desc=split):
            sample = dataset[i]
            sample_id = sample["info"]["name"]
            sample_dir = config.out_path / split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image = sample["image"].squeeze().unsqueeze(0).to(device)
            mask = sample["label"].to(torch.uint8)

            for rep in range(config.repetitions):
                out_file = sample_dir / f"{rep}.pt"

                if out_file.exists():
                    done = torch.load(out_file, weights_only=False)["epochs"].keys()

                    if set(config.epochs) <= set(done):
                        tqdm.write(f"Skipping {split}/{sample_id}/{rep}: already done")
                        continue

                seed = config.seed + rep
                torch.manual_seed(seed)
                tqdm.write(f"{split}/{sample_id} rep {rep} (seed {seed})")

                output = {
                    "config": config_to_dict(config),
                    "split": split,
                    "sample_id": sample_id,
                    "repetition": rep,
                    "seed": seed,
                    "mask": mask,
                    "intensity_min": sample["info"]["intensity_min"],
                    "intensity_max": sample["info"]["intensity_max"],
                    "epochs": {},
                }
                train_sample(config, image, output, out_file, device)

                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Gaussian Intensity Fields to OASIS dataset splits.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/vol/miltank/projects/L2R/OASIS_L2R",
        help="Path to the root of the dataset containing train, val, and test subdirectories.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/out/run_x/",
        help="Path to output directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        required=True,
        help="Epochs at which to save a snapshot, e.g. --epochs 100 300 500.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="How many times to repeat the training.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First sample index of each split to process.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2 ** 31,
        help="Sample index (exclusive) at which to stop.",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="JSON file from the hparam search; its hparams override the flags below.",
    )
    parser.add_argument(
        "--loss", type=str, default="mse", choices=["mse", "l1", "ssim"]
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--params", type=float, default=1.0, help="Params in percent.")
    parser.add_argument(
        "--knn_rate", type=int, default=100, help="How often to recompute the KNN."
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=12,
        help="How large the Voronoi cells should be.",
    )
    parser.add_argument(
        "--distance", type=str, default="l2", choices=["l2", "mahalanobis"]
    )
    parser.add_argument(
        "--sample_prop",
        type=float,
        default=SEARCH_DEFAULTS.sample_prop,
        help="Fraction of voxels per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=848577,
        help="Base seed, repetition r uses seed + r.",
    )

    args = parser.parse_args()

    if args.hparams is not None:
        with open(args.hparams) as f:
            hp = json.load(f)["hparams"]
        args.params = hp["params_percent"]
        args.loss = hp["loss"]
        args.lr = hp["lr"]
        args.distance = hp["distance"]
        args.cell_size = hp["cell_size"]
        args.knn_rate = hp["knn_update_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")

    training_config = TrainingConfig(
        dataset_path=dataset_path,
        out_path=Path(args.out_path),
        split=args.splits,
        start=args.start,
        end=args.end,
        epochs=sorted(set(args.epochs)),
        repetitions=args.reps,
        loss=args.loss,
        learning_rate=args.lr,
        params=args.params,
        knn_rate=args.knn_rate,
        cell_size=args.cell_size,
        distance=args.distance,
        sample_prop=args.sample_prop,
        min_spatial_scale=SEARCH_DEFAULTS.min_spatial_scale,
        max_spatial_scale=SEARCH_DEFAULTS.max_spatial_scale,
        k=SEARCH_DEFAULTS.K,
        k_candidates=SEARCH_DEFAULTS.K_candidates,
        seed=args.seed,
    )
    print(training_config)

    process_config(training_config, device)
