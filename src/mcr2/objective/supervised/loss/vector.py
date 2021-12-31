import torch
import mcr2
from .abstract import AbstractSupervisedMCR2Loss

class VectorSupervisedMCR2Loss(AbstractSupervisedMCR2Loss):
    def _get_coding_rate(self, eps_sq):
        return mcr2.objective.supervised.coding_rate.VectorCodingRate(eps_sq)

__all__ = ["VectorSupervisedMCR2Loss"]
