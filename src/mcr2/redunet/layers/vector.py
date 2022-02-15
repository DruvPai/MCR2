import torch
import mcr2
import mcr2.functional as F
import opt_einsum

from .abstract import ReduLayer


class VectorReduLayer(ReduLayer):
    def __init__(self, eta, eps_sq, lmbda, num_classes, dimensions, differentiable):
        super(VectorReduLayer, self).__init__(eta, eps_sq, lmbda, num_classes, differentiable)

        self.D = dimensions

        self.gam = torch.nn.Parameter(
            torch.ones(self.K) / self.K,  # (K, )
            requires_grad=False
        )
        self.E = torch.nn.parameter.Parameter(
            torch.empty(size=(self.D, self.D)),  # (D, D)
            requires_grad=self.diff
        )
        self.Cs = torch.nn.parameter.Parameter(
            torch.empty(size=(self.K, self.D, self.D)),  # (K, D, D)
            requires_grad=self.diff
        )

        self.cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)

    def forward(self, V):  # (N, D)
        with self.diff_ctx():
            expd = opt_einsum.contract('ni, ij -> nj', V, self.E.conj())  # (N, D) x (D, D) -> (N, D)
            comp = opt_einsum.contract('ni, kij -> knj', V, self.Cs.conj())  # (N, D) x (K, D, D) -> (K, N, D)
            clus = self.nonlinear(comp)  # (N, D)
            V = V + self.eta * (expd - clus)  # (N, D)
            return F.normalize_vec(V)  # (N, D)

    def nonlinear(self, Cz):  # (K, N, D)
        K, N, D = Cz.shape  # (K, N, D)
        norm = torch.linalg.norm(Cz, axis=2).clamp(min=1e-8)  # (K, N)
        pred = torch.nn.functional.softmax(-self.lmbda * norm, dim=0).view(K, N, 1)  # (K, N, 1)
        gam = self.gam.view(K, 1, 1)  # (K, 1, 1)
        out = torch.sum(gam * Cz * pred, axis=0)  # sum((K, 1, 1) * (K, N, D) x (K, N, 1) -> (K, N, D)) -> (N, D)
        return out  # (N, D)

    def compute_gam(self, Pi):  # (N, K)
        with torch.no_grad():
            N, K = Pi.shape  # (N, K)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            return gamma_per_class  # (K, )

    def compute_E(self, V):
        with self.diff_ctx():
            N, D = V.shape  # (N, D)
            VTV = F.gram_vec(V)  # (D, D)
            alpha = D / (N * self.eps_sq)
            I = torch.eye(D)  # (D, D)
            Sigma_hat = I + alpha * VTV  # (D, D)
            return alpha * torch.linalg.inv(Sigma_hat)  # (D, D)

    def compute_Cs(self, V, Pi):
        with self.diff_ctx():
            N, D = V.shape  # (N, D)
            N, K = Pi.shape  # (N, K)
            VTV_per_class = F.gram_per_class_vec(V, Pi)  # (K, D, D)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            alpha_per_class = torch.where(  # stops divide by 0 errors
                N_per_class > 0.0,
                D / (self.eps_sq * N_per_class),  # (K, )
                torch.tensor(0.0)  # ()
            )  # (K, )
            alpha_per_class = alpha_per_class.view(K, 1, 1)  # (K, 1, 1)
            I = torch.eye(D).view(1, D, D)  # (1, D, D)
            Sigma_hat_per_class = I + alpha_per_class * VTV_per_class  # (K, D, D)
            return alpha_per_class * torch.linalg.inv(Sigma_hat_per_class)  # (K, D, D)

    def preprocess(self, V):
        with self.diff_ctx():
            return F.normalize_vec(V)

    def postprocess(self, V):
        with self.diff_ctx():
            return F.normalize_vec(V)

    def compute_mcr2(self, Z, Pi):
        with torch.no_grad():
            return (self.cr.DeltaR(Z, Pi), self.cr.R(Z), self.cr.Rc(Z, Pi))

__all__ = ["VectorReduLayer"]
