import torch


def logdet(H):
    """
    Takes the log determinant of a (possible batch of) Hermitian matrix.
    Args:
        H: A Hermitian matrix batch (*, D, D)

    Returns:
        The log determinant of H (*, )
    """
    ld = torch.slogdet(H)
    return (ld[0] * ld[1]).real


__all__ = ["logdet"]
