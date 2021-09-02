import torch
from mcr2.supervised_coding_rate_calculator import SupervisedCodingRateCalculator

class SupervisedMCR2Loss(torch.nn.Module):
    def __init__(self, eps_sq: float, invariance_type: str):
        super(SupervisedMCR2Loss, self).__init__()
        self.coding_rate: SupervisedCodingRateCalculator = SupervisedCodingRateCalculator(
            eps_sq=eps_sq, invariance_type=invariance_type
        )

    def forward(self, Z: torch.Tensor, y: torch.Tensor):
        return -self.coding_rate.compute_DeltaR(Z=Z, y=y, input_in_fourier_basis=False)