import torch
import opt_einsum

from mcr2.validity import Z_valid_order_2_no_invariance, Pi_valid
from mcr2.computation_primitives import logdet


class UnsupervisedCodingRateCalculator:
    def __init__(self, eps_sq: float):
        self.eps_sq: float = eps_sq

    def compute_DeltaR(self, Z: torch.Tensor, Pi: torch.Tensor):
        assert (Z_valid_order_2_no_invariance(Z=Z, invariance_type="none") and Pi_valid(Pi=Pi))
        return self.compute_R(Z=Z) - self.compute_Rc(Z=Z, Pi=Pi)

    def compute_R(self, Z: torch.Tensor) -> torch.Tensor:
        assert (Z_valid_order_2_no_invariance(Z=Z, invariance_type="none"))
        M: int = int(Z.shape[0])
        N: int = int(Z.shape[1])
        alpha: float = N / (M * self.eps_sq)
        I: torch.Tensor = torch.eye(n=N, device=Z.device)  # (N, N)
        cov: torch.Tensor = opt_einsum.contract("ji, jk -> ik", Z.conj(), Z)  # (N, N)
        shifted_cov: torch.Tensor = I + alpha * cov  # (N, N)
        return logdet(Z=shifted_cov) / 2  # ()

    def compute_Rc(self, Z: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        assert (Z_valid_order_2_no_invariance(Z=Z, invariance_type="none") and Pi_valid(Pi=Pi))
        M: int = int(Z.shape[0])
        N: int = int(Z.shape[1])
        alpha: float = N / self.eps_sq
        I: torch.Tensor = torch.eye(n=N, device=Z.device).unsqueeze(0)  # (1, N, N)
        cov: torch.Tensor = opt_einsum.contract("ij, ik, im -> kjm", Z, Pi, Z.conj())  # (M, N, N)
        shifted_cov: torch.Tensor = I + alpha * cov  # (M, N, N)
        return opt_einsum.contract("i -> ", logdet(Z=shifted_cov)) / 2  # ()

__all__ = ["UnsupervisedCodingRateCalculator"]
