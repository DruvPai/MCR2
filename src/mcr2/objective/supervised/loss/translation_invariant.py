import torch
import mcr2
from .abstract import AbstractSupervisedMCR2Loss

class TranslationInvariantSupervisedMCR2Loss(AbstractSupervisedMCR2Loss):
    def _get_coding_rate(self, eps_sq):
        return mcr2.objective.supervised.coding_rate.TranslationInvariantCodingRate(eps_sq)

__all__ = ["TranslationInvariantSupervisedMCR2Loss"]
