from typing import Literal

import numpy as np
import torch
from jaxtyping import Float32, Float64, Float, Int8, Int64, Int

type Split = Literal["train", "val", "test"]

type OasisLabel = Int8[torch.Tensor, "LR SI AP"]
type OasisLabelTensor = Int64[torch.Tensor, "LR SI AP"]
type OasisVolume = Float64[torch.Tensor, "LR SI AP"]
type OasisVolumeTensor = Float64[torch.Tensor, "LR SI AP"]

type OasisLabelBatch = Int8[OasisLabel, "B"]  # noqa: F821
type OasisVolumeBatch = Float64[OasisVolume, "B"]  # noqa: F821

type Label = OasisLabel
type LabelTensor = OasisLabelTensor
type Volume = OasisVolume
type VolumeTensor = OasisVolumeTensor

type RotationMatrix = Float64[torch.Tensor, "N 3 3"]

type IntIndices = Int[torch.Tensor, "V 3"]
type FloatIndices = Float[torch.Tensor, "V 3"]

type CoordsBatch = Float32[torch.Tensor, "N 3"]
type MuTensor = Float[torch.Tensor, "num_gaussians 3"]
