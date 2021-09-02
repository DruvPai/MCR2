import mcr2
import torch


def Z_y_or_Pi_valid(Z: torch.Tensor, y_or_Pi: torch.Tensor,
                    invariance_type: str) -> bool:  # (M, dims) and (M, )
    return Z_y_valid(Z=Z, y=y_or_Pi, invariance_type=invariance_type) \
           or Z_Pi_valid(Z=Z, Pi=y_or_Pi, invariance_type=invariance_type)


def Z_y_valid(Z: torch.Tensor, y: torch.Tensor,
              invariance_type: str) -> bool:  # (M, dims) and (M, )
    return Z_valid(Z=Z, invariance_type=invariance_type) \
           and y_valid(y) \
           and Z.shape[0] == y.shape[0]


def Z_Pi_valid(Z: torch.Tensor, Pi: torch.Tensor,
               invariance_type: str) -> bool:  # (M, dims) and (M, M)
    return Z_valid(Z=Z, invariance_type=invariance_type) \
           and Pi_valid(Pi) \
           and Z.shape[0] == Pi.shape[0] == Pi.shape[1]


def Z_valid(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, N) or (M, C, T) or (M, C, H, W)
    return Z_valid_order_2(Z=Z, invariance_type=invariance_type) \
           or Z_valid_order_3(Z=Z, invariance_type=invariance_type) \
           or Z_valid_order_4(Z=Z, invariance_type=invariance_type)


def Z_valid_order_2(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, N)
    return Z_valid_order_2_no_invariance(Z=Z, invariance_type=invariance_type)


def Z_valid_order_3(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, C, T)
    return Z_valid_order_3_shift_invariance(Z=Z, invariance_type=invariance_type)


def Z_valid_order_4(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, C, H, W)
    return Z_valid_order_4_translation_invariance(Z=Z, invariance_type=invariance_type)


def Z_valid_order_2_no_invariance(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, N)
    return len(Z.shape) == 2 \
           and mcr2._invariance.InvarianceSpecification.no_invariance(
        invariance_type=invariance_type)


def Z_valid_order_3_shift_invariance(Z: torch.Tensor, invariance_type: str) -> bool:  # (M, C, T)
    return len(Z.shape) == 3 \
           and mcr2._invariance.InvarianceSpecification.shift_invariance(
        invariance_type=invariance_type)


def Z_valid_order_4_translation_invariance(Z: torch.Tensor,
                                           invariance_type: str) -> bool:  # (M, C, H, W)
    return len(Z.shape) == 4 \
           and mcr2._invariance.InvarianceSpecification.translation_invariance(
        invariance_type=invariance_type)


def y_valid(y: torch.Tensor) -> bool:  # (M, )
    return len(y.shape) == 1


def Pi_valid(Pi: torch.Tensor) -> bool:  # (M, M)
    return len(Pi.shape) == 2 and Pi.shape[0] == Pi.shape[1]
