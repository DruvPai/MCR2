import torch
import mcr2


class AbstractSupervisedMCR2Loss(torch.nn.Module):
    def __init__(self, eps_sq):
        super(AbstractSupervisedMCR2Loss, self).__init__()
        self.cr = self._get_coding_rate(eps_sq)

    def _get_coding_rate(self, eps_sq):
        raise NotImplementedError("Initialized abstract objective class")

    def forward(self, Z, Pi):
        return -self.cr.DeltaR(Z, Pi)

__all__ = ["AbstractSupervisedMCR2Loss"]
