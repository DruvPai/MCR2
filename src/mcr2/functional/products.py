import opt_einsum
import torch


def similarity_vec(Z1, Z2=None):  # (M, D), (N, D)
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


def similarity_shift_invariant(Z1, Z2=None):  # (M, C, T), (N, C, T)
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


def similarity_translation_invariant(Z1, Z2=None):  # (M, C, H, W), (N, C, H, W)
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


def Sigma_hat_vec(Z):  # (N, D)
    """
    Computes the empirical covariance of a data matrix Z.
    Args:
        Z: a data matrix (N, D)

    Returns:
        The covariance matrix of Z data (D, D)
    """
    return Z.T.conj() @ Z / Z.shape[0]  # (N, D) x (N, D) -> (D, D)


def Sigma_hat_per_class_vec(Z, Pi):  # (N, D), (N, K)
    """
    Computes the empirical covariance  of each class of a data matrix Z.
    Args:
        Z: a data matrix (N, D)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, D, D)
    """
    K = Pi.shape[1]
    N_per_class = torch.sum(Pi, axis=0)  # (K, )
    weight_per_class = torch.where(  # stops divide by 0 errors
        N_per_class > 0,
        1 / N_per_class,  # (K, )
        torch.tensor(0.0)  # ()
    )  # (K, )
    ZPiZh = opt_einsum.contract("ni, nj, nk -> kij", Z.conj(), Z,
                                Pi.to(torch.float))  # (N, D) x (N, D) x (K, N) -> (K, D, D)
    return weight_per_class.view(K, 1, 1) * ZPiZh


def Sigma_hat_shift_invariant(Z):  # (N, C, T)
    """
    Computes the empirical covariance of a multi-channel 1D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last dimension.
    Args:
        Z: a data matrix (N, C, T)

    Returns:
        The covariance matrix of Z data (T, C, C)
    """
    return opt_einsum.contract("nct, ndt -> tcd", Z.conj(), Z) / Z.shape[0]  # (N, C, T) x (N, C, T) ->  (T, C, C)


def Sigma_hat_per_class_shift_invariant(Z, Pi):  # (K, T, C, C),  (N, K)
    """
    Computes the empirical covariance of each class of a multi-channel 1D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last dimension.
    Args:
        Z: a data matrix (N, C, T)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, T, C, C)
    """
    K = Pi.shape[1]
    N_per_class = torch.sum(Pi, axis=0)  # (K, )
    weight_per_class = torch.where(  # stops divide by 0 errors
        N_per_class > 0,
        1 / N_per_class,  # (K, )
        torch.tensor(0.0)  # ()
    )  # (K, )
    ZPiZh = opt_einsum.contract("nct, ndt, nk -> ktcd", Z.conj(), Z,
                                Pi.to(torch.cfloat))  # (N, C, T) x (N, C, T) x (K, N) -> (K, T, C, C)
    return weight_per_class.view(K, 1, 1, 1) * ZPiZh


def Sigma_hat_translation_invariant(Z):  # (N, C, H, W)
    """
    Computes the empirical covariance of a multi-channel 2D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last two dimensions.
    Args:
        Z: a data matrix (N, C, H, W)

    Returns:
        The covariance matrix of Z data (H, W, C, C)
    """
    return opt_einsum.contract("nchw, ndhw -> hwcd", Z.conj(), Z) / Z.shape[
        0]  # (N, C, H, W) x (N, C, H, W) -> (H, W, C, C)


def Sigma_hat_per_class_translation_invariant(Z, Pi):  # (N, C, H, W), (N, K)
    """
    Computes the empirical covariance of each class of a multi-channel 2D data matrix Z.
    Should be used in contexts where Z matrix is already FFTed in last two dimensions.
    Args:
        Z: a data matrix (N, C, H, W)
        Pi: a class information matrix (N, K)

    Returns:
        The class covariance tensor of Z data (K, H, W, C, C)
    """
    K = Pi.shape[1]
    N_per_class = torch.sum(Pi, axis=0)  # (K, )
    weight_per_class = torch.where(  # stops divide by 0 errors
        N_per_class > 0,
        1 / N_per_class,  # (K, )
        torch.tensor(0.0)  # ()
    )  # (K, )
    ZPiZh = opt_einsum.contract("nchw, ndhw, nk -> khwcd", Z.conj(), Z,
                                Pi.to(torch.cfloat))  # (N, C, H, W) x (N, C, H, W) x (N, K) -> (K, H, W, C, C)
    return weight_per_class.view(K, 1, 1, 1, 1) * ZPiZh


__all__ = [
    "similarity_vec", "similarity_shift_invariant", "similarity_translation_invariant",
    "Sigma_hat_vec", "Sigma_hat_per_class_vec",
    "Sigma_hat_shift_invariant", "Sigma_hat_per_class_shift_invariant",
    "Sigma_hat_translation_invariant", "Sigma_hat_per_class_translation_invariant"
]
