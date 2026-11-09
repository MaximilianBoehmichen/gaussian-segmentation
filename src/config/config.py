from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from config.types import Split


@dataclass
class MetaConfig:
    start: int = 0
    end: int = 2 ** 16
    split: list[Split] = field(default_factory=list)


@dataclass
class GSTrainingConfig(MetaConfig):
    """The training config for the gaussian splatting.

    Taken from https://github.com/MaximilianBoehmichen/ADLM/blob/main/optimize_static_repr_fast.py
    """

    num_gaussians: int = 16  # Number of Gaussians to train
    gradient_init_percentage: float = (
        1.0  # Percentage of Gaussians to init by image gradient
    )
    max_epochs: int = 500  # Number of optimization epochs
    knn_metric: Literal["l2", "mahalanobis"] = "l2"
    k_neighborhood: int = (
        15  # Number of nearby Gaussians to consider (larger K -> slower)
    )
    knn_update_rate: int = 10  # Number of steps after which the KNN will be updated
    reg_weight_scale: float = (
        0.1  # Regularization weight for Gaussian scales being too small/big
    )
    reg_weight_pos: float = (
        0.0  # Regularization weight for Gaussian position leaving image bounds
    )
    learning_rate_start: float = 1e-2  # Starting learning rate for optimization
    learning_rate_end: float = 1e-5  # Ending learning rate for optimization
    point_sample_prop: float = (
        0.9  # Percentage of total coordinates to supervise in minibatch
    )
    repetitions: int = 1


@dataclass
class TrainingConfig(MetaConfig):
    """The training config for the gaussian splatting."""
    dataset_path: Path = Path("/vol/miltank/projects/L2R/OASIS_L2R")
    out_path: Path = Path("./data/out/run_x/")
    epochs: list[int] = field(default_factory=list)  # Epochs at which a snapshot is saved
    repetitions: int = 1
    loss: Literal["l1", "mse", "ssim"] = "mse"
    learning_rate: float = 1e-3
    params: float = 1.0  # Gaussian parameters in percent of voxel count
    knn_rate: int = 100  # Rebuild KNN every n epochs
    cell_size: int = 12  # Voronoi cell spacing
    distance: Literal["l2", "mahalanobis"] = "l2"
    min_spatial_scale: float = 1.0
    max_spatial_scale: float = 20.0
    sample_prop: float = 0.1  # Fraction of voxels per batch
    k: int = 24  # Neighbors per voxel in forward pass
    k_candidates: int = 256
    nprobe: int = 16
    seed: int = 848577  # Repetition r uses seed + r
