import torch
import mcr2

class SupervisedMCR2Loss(torch.nn.Module):
    def __init__(self, eps_sq):
        super(SupervisedMCR2Loss, self).__init__()
        self.cr = self._get_coding_rate(eps_sq)

    def _get_coding_rate(self, eps_sq):
        raise NotImplementedError("Initialized abstract mcr2 class")

    def forward(self, Z, Pi):
        return -self.cr.DeltaR(Z, Pi)


class SupervisedVectorMCR2Loss(SupervisedMCR2Loss):
    def _get_coding_rate(self, eps_sq):
        return mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)


class SupervisedShiftInvariantMCR2Loss(SupervisedMCR2Loss):
    def _get_coding_rate(self, eps_sq):
        return mcr2.coding_rate.SupervisedShiftInvariantCodingRate(eps_sq)


class SupervisedTranslationInvariantMCR2Loss(SupervisedMCR2Loss):
    def _get_coding_rate(self, eps_sq):
        return mcr2.coding_rate.SupervisedTranslationInvariantCodingRate(eps_sq)


__all__ = ["SupervisedVectorMCR2Loss", "SupervisedShiftInvariantMCR2Loss", "SupervisedTranslationInvariantMCR2Loss"]
