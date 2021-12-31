import torch

def logdet(H):
    """
    Takes the log determinant of a (possible batch of) Hermitian matrix.
    Args:
        H: A Hermitian matrix batch (*, D, D)

    Returns:
        The log determinant of H (*, )
    """
    L = torch.linalg.cholesky(H)  # (*, D, D)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (*, D)
    log_diag_L = torch.log(diag_L)  # (*, D)
    tr_log_L = torch.sum(log_diag_L, dim=-1)  # (*, )
    return 2 * tr_log_L  # (*, )

__all__ = ["logdet"]