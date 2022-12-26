from torch import Tensor

from src.mcr2.primitives.coding_rate import DeltaR


def supervised_mcr2_loss(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the MCR2 loss from "Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction" by Yu et al.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (N, K); a {0, 1} matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the MCR2 loss of Z with respect to y.
    """
    return DeltaR(Z, y_onehot, eps)


__all__ = ["supervised_mcr2_loss"]
