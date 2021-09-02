import mcr2
import opt_einsum
import torch


class SupervisedCodingRateCalculator:
    def __init__(self, eps_sq: float, invariance_type: str = None):
        self.eps_sq: float = eps_sq
        self.invariance_type: str = invariance_type

    def compute_DeltaR(
            self,
            Z: torch.Tensor,
            y: torch.Tensor,
            input_in_fourier_basis: bool = False
    ):
        assert (mcr2._validity.Z_y_valid(Z=Z, y=y, invariance_type=self.invariance_type))
        V: torch.Tensor = self._fft_input_if_necessary(
            Z=Z, input_in_fourier_basis=input_in_fourier_basis
        )
        return self.compute_R(Z=V, input_in_fourier_basis=True) \
               - self.compute_Rc(Z=V, y=y, input_in_fourier_basis=True)

    def compute_R(self, Z: torch.Tensor, input_in_fourier_basis: bool = False) -> torch.Tensor:
        assert (mcr2._validity.Z_valid(Z=Z, invariance_type=self.invariance_type))
        V: torch.Tensor = self._fft_input_if_necessary(
            Z=Z,
            input_in_fourier_basis=input_in_fourier_basis
        )
        if mcr2._validity.Z_valid_order_2_no_invariance(Z=Z, invariance_type=self.invariance_type):
            return self._compute_R_order_2_no_invariance(V=V)
        elif mcr2._validity.Z_valid_order_3_shift_invariance(
                Z=Z, invariance_type=self.invariance_type
        ):
            return self._compute_R_order_3_shift_invariance(V=V)
        elif mcr2._validity.Z_valid_order_4_translation_invariance(
                Z=Z, invariance_type=self.invariance_type
        ):
            return self._compute_R_order_4_translation_invariance(V=V)

    def compute_Rc(
            self,
            Z: torch.Tensor,
            y: torch.Tensor,
            input_in_fourier_basis: bool = False,
    ) -> torch.Tensor:
        assert (mcr2._validity.Z_y_valid(Z=Z, y=y, invariance_type=self.invariance_type))
        V: torch.Tensor = self._fft_input_if_necessary(
            Z=Z, input_in_fourier_basis=input_in_fourier_basis
        )
        M: int = int(V.shape[0])
        unique_outputs: typing.Tuple[torch.Tensor] = torch.unique(input=y, return_counts=True)
        classes: torch.Tensor = unique_outputs[0]  # (K, )
        counts: torch.Tensor = unique_outputs[1]  # (K, )
        gamma: torch.Tensor = counts / M  # (K, )
        return sum(
            gamma[i] *
            self.compute_R(  # ()
                Z=V[y == classes[i]],  # (M_classes[j], dims),
                input_in_fourier_basis=True
            ) for i in range(classes.shape[0])
        )  # ()

    def _compute_R_order_2_no_invariance(self, V: torch.Tensor) -> torch.Tensor:
        M: int = int(V.shape[0])
        N: int = int(V.shape[1])
        alpha: float = N / (M * self.eps_sq)
        I: torch.Tensor = torch.eye(n=N, device=V.device)  # (N, N)
        cov: torch.Tensor = opt_einsum.contract("ji, jk -> ik", V, V.conj())  # (N, N)
        shifted_cov: torch.Tensor = I + alpha * cov  # (N, N)
        return mcr2._computation_primitives.logdet(Z=shifted_cov) / 2  # ()

    def _compute_R_order_3_shift_invariance(self, V: torch.Tensor) -> torch.Tensor:
        M: int = int(V.shape[0])
        C: int = int(V.shape[1])
        T: int = int(V.shape[2])
        alpha: float = C / (M * self.eps_sq)
        I: torch.Tensor = torch.eye(n=C, device=V.device).unsqueeze(0)  # (1, C, C)
        cov: torch.Tensor = opt_einsum.contract("jil, jkl -> lik", V, V.conj())  # (T, C, C)
        shifted_cov: torch.Tensor = I + alpha * cov  # (T, C, C)
        return opt_einsum.contract(
            "i -> ",
            mcr2._computation_primitives.logdet(Z=shifted_cov)
        ).real / (2 * T)  # ()

    def _compute_R_order_4_translation_invariance(self, V: torch.Tensor) -> torch.Tensor:
        M: int = int(V.shape[0])
        C: int = int(V.shape[1])
        H: int = int(V.shape[2])
        W: int = int(V.shape[3])
        alpha: float = C / (M * self.eps_sq)
        I: torch.Tensor = torch.eye(n=C, device=V.device).unsqueeze(0)  # (1, 1, C, C)
        cov: torch.Tensor = opt_einsum.contract("jihw, jkhw -> hwik", V, V.conj())  # (H, W, C, C)
        shifted_cov: torch.Tensor = I + alpha * cov  # (H, W, C, C)
        return opt_einsum.contract(
            "ij -> ",
            mcr2._computation_primitives.logdet(Z=shifted_cov)
        ).real / (2 * H * W)  # ()

    def _fft_input_if_necessary(self, Z: torch.Tensor,
                                input_in_fourier_basis: bool = False) -> torch.Tensor:
        if input_in_fourier_basis or len(Z.shape) == 2:  # fourier transform is orthogonal
            return Z
        else:
            return mcr2._computation_primitives.fft_input(Z=Z, invariance_type=self.invariance_type)


class SupervisedMCR2Loss(torch.nn.Module):
    def __init__(self, eps_sq: float, invariance_type: str):
        super(SupervisedMCR2Loss, self).__init__()
        self.coding_rate: SupervisedCodingRateCalculator = SupervisedCodingRateCalculator(
                eps_sq=eps_sq, invariance_type=invariance_type
            )

    def forward(self, Z: torch.Tensor, y: torch.Tensor):
        return -self.coding_rate.compute_DeltaR(Z=Z, y=y, input_in_fourier_basis=False)


__all__ = ["SupervisedCodingRateCalculator", "SupervisedMCR2Loss"]
