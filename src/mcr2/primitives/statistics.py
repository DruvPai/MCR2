from torch import einsum, maximum, tensor, Tensor

from src.mcr2.primitives.products import tensorized_ZtZ, tensorized_ZtZ_class


def second_moment(Z: Tensor):
    """
    Computes the empirical second moment of the input matrices.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.

    Returns:
        Tensor of shape (*, D, D); the empirical second moment of Z.
    """
    N = tensor(Z.shape[-2], device=Z.device)  # ()
    N = maximum(N, tensor(1.0, device=Z.device))  # (*)
    ZtZ = tensorized_ZtZ(Z)  # (*, D, D)
    return ZtZ / N  # (*, D, D)


def second_moment_class(Z: Tensor, y_onehot: Tensor):
    """
    Computes the empirical second moment of each class of the input matrices.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.

    Returns:
        Tensor of shape (*, K, D, D); the empirical second moment of each class of Z.
    """
    ZtZ = tensorized_ZtZ_class(Z, y_onehot)  # (*, K, D, D)
    Nc = y_onehot.float().sum(dim=-2).unsqueeze(-1).unsqueeze(-1)  # (*, K, 1, 1)
    Nc = maximum(Nc, tensor(1.0, device=ZtZ.device))  # (*, K, 1, 1)
    return ZtZ / Nc  # (*, K, D, D)


def gramian(Z: Tensor):
    """
    Computes the Gramian of the rows of each input matrix.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices where each row is a data point and each column is a feature.

    Returns:
        Tensor of shape (*, N, N); the Gramian of the rows of Z.
    """
    ZZt = einsum("...ni, ...mi -> ...mn", Z, Z)  # (*, N, N)
    return ZZt  # (*, N, N)


__all__ = ["second_moment", "second_moment_class", "gramian"]
