from torch import maximum, tensor, Tensor
from torch.linalg import cholesky, eigvalsh


def logdet_I_plus(M: Tensor):
    """
    Computes the log determinant of a matrix or batch of matrices of the form I + M where M is Hermitian PSD.

    Args:
        M: Tensor of shape (*, D, D); a batch of Hermitian PSD matrices.

    Returns:
        Tensor of shape (*); the log-determinant(s) of I + M.
    """
    ev = eigvalsh(M)  # (*, D)
    ev = maximum(ev, tensor(0.0, device=M.device))  # (*, D)  # Numerical precision to prevent blowup
    return (1 + ev).log().sum(dim=-1)  # (*)


__all__ = ["logdet_I_plus"]
