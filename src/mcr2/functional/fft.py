import torch

def fft(Z):
    """
    Takes 1d FFT of the last dimension of Z.
    Args:
        Z: a data matrix (*, T)

    Returns:
        A tensor with the last dimension equal to FFT of Z (*, T)
    """
    return torch.fft.fft(Z, norm="ortho")

def ifft(Z):
    """
    Takes 1d IFFT of the last dimension of Z.
    Args:
        Z: an FFTed data matrix (*, T)

    Returns:
        A tensor with the last dimension equal to IFFT of Z (*, T)
    """
    return torch.fft.ifft(Z, norm="ortho")

def fft2(Z):
    """
    Takes 2d FFT of the last two dimensions of Z.
    Args:
        Z: a data matrix (*, H, W)

    Returns:
        A tensor with the last two dimensions equal to 2d FFT of Z (*, H, W)
    """
    return torch.fft.fft2(Z, norm="ortho")

def ifft2(Z):
    """
    Takes 2d IFFT of the last two dimensions of Z.
    Args:
        Z: an FFTed data matrix (*, H, W)

    Returns:
        A tensor with the last two dimensions equal to 2d IFFT of Z (*, H, W)
    """
    return torch.fft.ifft2(Z, norm="ortho")

__all__ = ["fft", "fft2", "ifft", "ifft2"]
