from torch import Tensor
from torch.linalg import cholesky


def logdet_hpd(M: Tensor):
    """
    Computes the log determinant of a Hermitian PD matrix or batch of Hermitian PD matrices.

    Args:
        M: Tensor of shape (*, D, D); a batch of Hermitian PD matrices.

    Returns:
        Tensor of shape (*); the log-determinants of such Hermitian PD matrices.
    """
    return 2 * cholesky(M).diagonal(dim1=-2, dim2=-1).log().sum(-1)


__all__ = ["logdet_hpd"]
