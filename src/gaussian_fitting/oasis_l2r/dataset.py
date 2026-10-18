from collections.abc import Callable
from linecache import cache
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from config.types import OasisLabel, Split, OasisVolume


class OasisL2RDataset(Dataset[OasisVolume]):
    """The dataset class for the Oasis L2R dataset."""

    root: Path
    split: Split
    transforms: Callable[[dict[str, OasisVolume | OasisLabel]], dict[str, OasisVolume | OasisLabel]]
    cache: bool
    seed: int
    filenames: tuple[str, str]

    _split_files: list[Path]
    _sample_cache: dict[int, tuple[OasisVolume, OasisLabel]] = {}

    def __init__(
            self,
            root: Path | str,
            split: Split,
            *,
            transforms: Callable[[dict[str, OasisVolume | OasisLabel]], dict[str, OasisVolume | OasisLabel]] = lambda
                    x: x,
            cache: bool = False,
            pre_load: bool = False,
            seed: int = 848577,
            filenames: tuple[str, str] = ("aligned_norm.nii.gz", "aligned_seg4.nii.gz"),
    ) -> None:
        """Initializes the dataset.

        Args:
            root (Path): Path to the root directory of the dataset.
            split (Split): Split of the dataset.
            transforms (Sequence[Callable[[OasisVolume], OasisVolume]]): The chosen transforms to apply at access.
            cache (bool): Whether to cache the dataset in memory.
            pre_load (bool): If cache is enabled, whether to load the samples at initialization.
            seed (int): Random seed if necessary.
            filenames (tuple[str, str]): Common file name of (image, label).
        """
        self.split_path = Path(root) / split
        self.transforms = transforms
        self.cache = cache
        self.seed = seed
        self.filenames = filenames

        self._split_files = sorted(d for d in self.split_path.iterdir() if d.is_dir())

        if cache and pre_load:
            for i, sample_path in enumerate(self._split_files):
                self._sample_cache[i] = self._load_sample(sample_path)

    def _load_sample(self, path: Path) -> tuple[OasisVolume, OasisLabel]:
        """Loads the sample image and label.

        Args:
            path (Path): Path to the sample.

        Returns:
            The dataset returnable sample.
        """
        image = nib.Nifti1Image.from_filename(path / self.filenames[0]).get_fdata(dtype=np.float32)
        label = nib.Nifti1Image.from_filename(path / self.filenames[1]).get_fdata(dtype=np.uint8)

        return image, label

    def __len__(self) -> int:
        """Gives the length of the dataset's current split.

        Returns:
            The length of the dataset.
        """
        return len(self._split_files)

    def __getitem__(self, index: int) -> dict[str, OasisVolume | OasisLabel]:
        """Gives the data for a given selected sample.

        Args:
            index (int): The index of the sample.

        Returns:
            The sample for the index after applied transforms.
        """
        image, label = self._sample_cache.get(index, (None, None))

        if image is None or label is None:
            image, label = self._load_sample(self._split_files[index])

            if cache:
                self._sample_cache[index] = (image, label)

        sample = {
            "image": image.copy(),
            "label": label.copy().astype(np.int64),
        }

        return self.transforms(sample)
