import torch


def logdet(H, min_sv=1e-8):
    """
    Takes the log determinant of a (possible batch of) real symmetric positive definite matrix.
    Args:
        H: A Hermitian matrix batch (*, D, D)

    Returns:
        The log determinant of H (*, )
    """
    singular_values = torch.linalg.svdvals(H)
    thresholded_singular_values = torch.clamp(singular_values, min=min_sv, max=None)
    return torch.sum(torch.log(thresholded_singular_values), dim=-1)


def logdet_complex(H):
    """
    Takes the log determinant of a (possible batch of) complex positive definite Hermitian matrix.
    Args:
        H: A Hermitian matrix batch (*, D, D)

    Returns:
        The log determinant of H (*, )
    """
    ld = torch.slogdet(H)
    return (ld[0] * ld[1]).real


__all__ = ["logdet", "logdet_complex"]
