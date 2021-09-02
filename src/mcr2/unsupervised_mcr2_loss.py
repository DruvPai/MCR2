import torch
from mcr2.unsupervised_coding_rate_calculator import UnsupervisedCodingRateCalculator
import opt_einsum


class UnsupervisedMCR2Loss(torch.nn.Module):
    def __init__(self, eps_sq: float, gamma: float):
        super(UnsupervisedMCR2Loss, self).__init__()
        self.coding_rate: UnsupervisedCodingRateCalculator = UnsupervisedCodingRateCalculator(
            eps_sq=eps_sq
        )
        self.gamma: float = gamma

    def forward(self, Z: torch.Tensor, Pi: torch.Tensor):
        return self.gamma * opt_einsum.contract("ij, ij ->", Pi, Pi) / 2 \
               - self.coding_rate.compute_DeltaR(Z=Z, Pi=Pi)