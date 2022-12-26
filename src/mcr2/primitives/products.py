from torch import Tensor, einsum


def tensorized_ZtZ(Z: Tensor):
    """
    For a batch of matrices Z, computes a vectorized Z.T @ Z.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices.

    Returns:
        Tensor of shape (*, D, D); the matrices Z.T @ Z.
    """
    ZtZ = einsum("...ni, ...nj -> ...ij", Z, Z)  # (*, D, D)
    return ZtZ  # (*, D, D)


def tensorized_ZtZ_class(Z: Tensor, y_onehot: Tensor):
    """
    For a batch of matrices Z, computes a vectorized Zi.T @ Zi for each class i.

    Args:
        Z: Tensor of shape (*, N, D); a batch of matrices.
        y_onehot: Tensor of shape (*, N, K); a batch of {0, 1} matrices where each row is a data point and each column is a class.

    Returns:
        Tensor of shape (*, K, D, D); the matrices Zi.T @ Zi.
    """
    ZtZ = einsum("...ni, ...nj, ...nk -> ...kij", Z, Z, y_onehot.float())  # (*, K, D, D)
    return ZtZ  # (*, K, D, D)
