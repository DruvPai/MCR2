import torch

def normalize_frob(Z):  # (N, *)
    """
    Normalizes the tensor Z by dividing it by its Frobenius norm.
    Note: We use Frobenius normalization, not column-by-column normalization.
    This is for a variety of theoretical and empirical reasons.

    Args:
        Z: the data tensor (N, *)

    Returns:
        A normalized version of Z (N, *).
    """
    return Z / torch.linalg.norm(Z)

def normalize_vec(Z):  # (N, D)
    """
    Normalizes each row of the tensor Z by dividing it by its Frobenius (l2) norm.
    Args:
        Z: the data tensor (N, D)

    Returns:
        A normalized version of Z (N, D)
    """
    return torch.nn.functional.normalize(Z, dim=1)

def normalize_1d(Z):  # (N, C, T)
    """
    Normalizes each row of the tensor Z by dividing it by its Frobenius norm.
    Args:
        Z: the data tensor (N, C, T)

    Returns:
        A normalized version of Z (N, C, T)
    """
    N, C, T = Z.shape
    return torch.nn.functional.normalize(Z.reshape(N, C * T), dim=1).reshape(N, C, T)

def normalize_2d(Z):  # (N, C, H, W)
    """
    Normalizes each row of the tensor Z by dividing it by its Frobenius norm.
    Args:
        Z: the data tensor (N, C, H, W)

    Returns:
        A normalized version of Z (N, C, H, W)
    """
    N, C, H, W = Z.shape
    return torch.nn.functional.normalize(Z.reshape(N, C * H * W), dim=1).reshape(N, C, H, W)

__all__ = ["normalize_vec", "normalize_1d", "normalize_2d", "normalize_frob"]
