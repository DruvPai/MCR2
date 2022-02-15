import torch
import opt_einsum


def inner_product_vec(Z1, Z2=None):  # (M, D), (N, D)
    """
    Takes the pairwise inner products of all rows of Z1 and Z2.

    Args:
        Z1: first data matrix (M, D)
        Z2: second data matrix (N, D)

    Returns:
        A matrix containing the pairwise inner products (M, N)
    """
    if Z2 is None:
        Z2 = Z1
    return Z1.conj() @ Z2.T  # (M, N)


def inner_product_shift_invariant(Z1, Z2=None):  # (M, C, T), (N, C, T)
    """
    Takes the pairwise inner products of all rows of Z1 and Z2.

    Args:
        Z1: first data matrix (M, C, T)
        Z2: second data matrix (N, C, T)

    Returns:
        A matrix containing the pairwise inner products (M, N)
    """
    if Z2 is None:
        Z2 = Z1
    return opt_einsum.contract("mct, nct -> mn", Z1.conj(), Z2)  # (M, N)


def inner_product_translation_invariant(Z1, Z2=None):  # (M, C, H, W), (N, C, H, W)
    """
    Takes the pairwise inner products of all rows of Z1 and Z2.

    Args:
        Z1: first data matrix (M, C, H, W)
        Z2: second data matrix (N, C, H, W)

    Returns:
        A matrix containing the pairwise inner products (M, N)
    """
    if Z2 is None:
        Z2 = Z1
    return opt_einsum.contract("mchw, nchw -> mn", Z1.conj(), Z2)  # (M, N)


def gram_vec(Z):  # (N, D)
    """
    Computes the empirical second moment of a data matrix Z.
    Args:
        Z: a data matrix (N, D)

    Returns:
        The covariance matrix of Z data (D, D)
    """
    return Z.T @ Z.conj()  # (N, D) x (N, D) -> (D, D)

def gram_per_class_vec(Z, Pi):  # (N, D), (N, K)
    """
    Computes the empirical second moment of each class of a data matrix Z.
    Args:
        Z: a data matrix (N, D)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, D, D)
    """
    return opt_einsum.contract("ni, nj, nk -> kij", Z, Z.conj(), Pi.to(torch.float))  # (N, D) x (N, D) x (K, N) -> (K, D, D)

def gram_shift_invariant(Z):  # (N, C, T)
    """
    Computes the empirical second moment of a multi-channel 1D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last dimension.
    Args:
        Z: a data matrix (N, C, T)

    Returns:
        The covariance matrix of Z data (T, C, C)
    """
    return opt_einsum.contract("nct, ndt -> tcd", Z, Z.conj())  # (N, C, T) x (N, C, T) ->  (T, C, C)

def gram_per_class_shift_invariant(Z, Pi):  # (K, T, C, C),  (N, K)
    """
    Computes the empirical second moment of each class of a multi-channel 1D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last dimension.
    Args:
        Z: a data matrix (N, C, T)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, T, C, C)
    """
    return opt_einsum.contract("nct, ndt, nk -> ktcd", Z, Z.conj(), Pi.to(torch.float))  # (N, C, T) x (N, C, T) x (K, N) -> (K, T, C, C)

def gram_translation_invariant(Z):  # (N, C, H, W)
    """
    Computes the empirical second moment of a multi-channel 2D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last two dimensions.
    Args:
        Z: a data matrix (N, C, H, W)

    Returns:
        The covariance matrix of Z data (H, W, C, C)
    """
    return opt_einsum.contract("nchw, ndhw -> hwcd", Z, Z.conj())  # (N, C, H, W) x (N, C, H, W) -> (H, W, C, C)

def gram_per_class_translation_invariant(Z, Pi):  # (N, C, H, W), (N, K)
    """
    Computes the empirical second moment of each class of a multi-channel 2D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last two dimensions.
    Args:
        Z: a data matrix (N, C, H, W)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, H, W, C, C)
    """
    return opt_einsum.contract("nchw, ndhw, nk -> khwcd", Z, Z.conj(), Pi.to(torch.float))  # (N, C, H, W) x (N, C, H, W) x (N, K) -> (K, H, W, C, C)

__all__ = [
    "inner_product_vec", "inner_product_shift_invariant", "inner_product_translation_invariant",
    "gram_vec", "gram_per_class_vec",
    "gram_shift_invariant", "gram_per_class_shift_invariant",
    "gram_translation_invariant", "gram_per_class_translation_invariant"
]
