import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0):
        super().__init__()

        self.data_range = data_range

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - ssim(input, target, data_range=self.data_range)
