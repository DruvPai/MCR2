import torch
import mcr2
import mcr2.functional as F
import opt_einsum

from .abstract import ReduLayer


class ShiftInvariantReduLayer(ReduLayer):
    def __init__(self, eta, eps_sq, lmbda, num_classes, dimensions, differentiable):
        super(ShiftInvariantReduLayer, self).__init__(eta, eps_sq, lmbda, num_classes, differentiable)

        self.C, self.T = dimensions

        self.gam = torch.nn.Parameter(
            torch.ones(self.K) / self.K,  # (K, )
            requires_grad=False
        )
        self.E = torch.nn.parameter.Parameter(
            torch.empty(size=(self.T, self.C, self.C)),  # (T, C, C)
            requires_grad=self.diff
        )
        self.Cs = torch.nn.parameter.Parameter(
            torch.empty(size=(self.K, self.T, self.C, self.C)),  # (K, T, C, C)
            requires_grad=self.diff
        )

        self.cr = mcr2.coding_rate.SupervisedShiftInvariantCodingRate(eps_sq)

    def forward(self, V):  # (N, C, T)
        with self.diff_ctx():
            expd = opt_einsum.contract("nct, tcd -> ndt", V, self.E.conj())  # (N, C, T) x (T, C, C) -> (N, C, T)
            comp = opt_einsum.contract("nct, ktcd -> kndt", V, self.Cs.conj())  # (N, C, T) x (K, T, C, C) -> (K, N, C, T)
            clus = self.nonlinear(comp)  # (N, C, T)
            V = V + self.eta * (expd - clus)  # (N, C, T)
            return F.normalize_1d(V)  # (N, C, T)

    def nonlinear(self, Cz):  # (K, N, C, T)
        K, N, C, T = Cz.shape  # (K, N, C, T)
        norm = torch.linalg.norm(Cz.flatten(start_dim=2), axis=2).clamp(min=1e-8)  # (K, N)
        pred = torch.nn.functional.softmax(-self.lmbda * norm, dim=0).view(K, N, 1, 1)  # (K, N, 1, 1)
        gam = self.gam.view(K, 1, 1, 1)  # (K, 1, 1, 1)
        out = torch.sum(gam * Cz * pred, axis=0)  # sum((K, 1, 1, 1) x (K, N, C, T) x (K, N, 1, 1) -> (K, N, C, T)) -> (N, C, T)
        return out  # (N, C, T)

    def compute_gam(self, Pi):  # (N, K)
        with torch.no_grad():
            N, K = Pi.shape  # (N, K)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            return gamma_per_class  # (K, )

    def compute_E(self, V):  # (N, C, T)
        with self.diff_ctx():
            N, C, T = V.shape  # (N, C,T)
            VTV = F.gram_shift_invariant(V)  # (T, C, C)
            alpha = C / (N * self.eps_sq)
            I = torch.eye(C).view(1, C, C)  # (1, C, C)
            Sigma_hat = I + alpha * VTV  # (T, C, C)
            return alpha * torch.linalg.inv(Sigma_hat)  # (T, C, C)

    def compute_Cs(self, V, Pi):
        with self.diff_ctx():
            N, C, T = V.shape  # (N, C, T)
            N, K = Pi.shape  # (N, K)
            VTV_per_class = F.gram_per_class_shift_invariant(V, Pi)  # (K, T, C, C)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            alpha_per_class = torch.where(  # stops divide by 0 errors
                N_per_class > 0.0,
                C / (self.eps_sq * N_per_class),  # (K, )
                torch.tensor(0.0)  # ()
            )  # (K, )
            alpha_per_class = alpha_per_class.view(K, 1, 1, 1)  # (K, 1, 1, 1)
            I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
            Sigma_hat_per_class = I + alpha_per_class * VTV_per_class  # (K, T, C, C)
            return alpha_per_class * torch.linalg.inv(Sigma_hat_per_class)  # (K, T, C, C)

    def preprocess(self, V):
        with self.diff_ctx():
            return F.normalize_1d(F.fft(V))

    def postprocess(self, V):
        with self.diff_ctx():
            return F.normalize_1d(F.ifft(V).real.to(torch.float))

    def compute_mcr2(self, Z, Pi):
        with torch.no_grad():
            return (self.cr.DeltaR(Z, Pi), self.cr.R(Z), self.cr.Rc(Z, Pi))

__all__ = ["ShiftInvariantReduLayer"]
