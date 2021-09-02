import torch
import typing
from mcr2.invariance import InvarianceSpecification
from mcr2.validity import Z_valid_order_2, Z_valid_order_3, Z_valid_order_4


def logdet(Z: torch.Tensor) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
    sgn, logabsdet = torch.slogdet(Z)
    return sgn * logabsdet


def fft_input(Z: torch.Tensor, invariance_type: str) -> torch.Tensor:
    if len(Z.shape) == 2:  # (M, N)
        assert (Z_valid_order_2(Z=Z, invariance_type=invariance_type))
        if InvarianceSpecification.no_invariance(invariance_type=invariance_type):
            return torch.fft.fft(input=Z, dim=1, norm="ortho")
    elif len(Z.shape) == 3:  # (M, C, T)
        assert (Z_valid_order_3(Z=Z, invariance_type=invariance_type))
        if InvarianceSpecification.shift_invariance(invariance_type=invariance_type):
            return torch.fft.fft(input=Z, dim=2, norm="ortho")
    elif len(Z.shape) == 4:  # (M, C, H, W)
        assert (Z_valid_order_4(Z=Z, invariance_type=invariance_type))
        if InvarianceSpecification.translation_invariance(invariance_type=invariance_type):
            return torch.fft.fft2(input=Z, dim=(2, 3), norm="ortho")
