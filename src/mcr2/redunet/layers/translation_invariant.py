import torch
import mcr2
import mcr2.functional as F
import opt_einsum

from .abstract import ReduLayer


class TranslationInvariantReduLayer(ReduLayer):
    def __init__(self, eta, eps_sq, lmbda, num_classes, dimensions, differentiable):
        super(TranslationInvariantReduLayer, self).__init__(eta, eps_sq, lmbda, num_classes, differentiable)

        self.C, self.H, self.W = dimensions

        self.gam = torch.nn.Parameter(
            torch.ones(self.K) / self.K,  # (K, )
            requires_grad=False
        )
        self.E = torch.nn.parameter.Parameter(
            torch.empty(size=(self.H, self.W, self.C, self.C)),  # (D, D)
            requires_grad=self.diff
        )
        self.Cs = torch.nn.parameter.Parameter(
            torch.empty(size=(self.K, self.H, self.W, self.C, self.C)),  # (K, D, D)
            requires_grad=self.diff
        )

        self.cr = mcr2.coding_rate.SupervisedTranslationInvariantCodingRate(eps_sq)

    def forward(self, V):  # (N, C, H, W)
        with self.diff_ctx():
            expd = opt_einsum.contract("nchw, hwcd -> ndhw", V, self.E.conj())  # (N, C, H, W) x (H, W, C, C) -> (N, C, H, W)
            comp = opt_einsum.contract("nchw, khwcd -> kndhw", V, self.Cs.conj())  # (N, C, H, W) x (K, H, W, C, C) -> (K, N, C, H, W)
            clus = self.nonlinear(comp)  # (N, C, H, W)
            V = V + self.eta * (expd - clus)  # (N, C, H, W)
            return F.normalize_2d(V)  # (N, C, H, W)

    def nonlinear(self, Cz):  # (K, N, C, H, W)
        K, N, C, H, W = Cz.shape  # (K, N, C, H, W)
        norm = torch.linalg.norm(Cz.flatten(start_dim=2), axis=2)  # (K, N)
        pred = torch.nn.functional.softmax(-self.lmbda * norm, dim=0).view(K, N, 1, 1, 1)  # (K, N, 1, 1, 1)
        gam = self.gam.view(K, 1, 1, 1, 1)  # (K, 1, 1, 1, 1)
        out = torch.sum(gam * Cz * pred, axis=0)  # sum((K, 1, 1, 1, 1) x (K, N, C, H, W) x (K, N, 1, 1, 1) -> (K, N, C, H, W)) -> (N, C, H, W)
        return out  # (N, C, H, W)

    def compute_gam(self, Pi):  # (K, N)
        with torch.no_grad():
            K, N = Pi.shape  # (K, N)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            return gamma_per_class  # (K, )

    def compute_E(self, V):  # (N, C, H, W)
        with self.diff_ctx():
            N, C, H, W = V.shape  # (N, C, H, W)
            VTV = F.gram_translation_invariant(V)  # (H, W, C, C)
            alpha = C / (N * self.eps_sq)
            I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
            Sigma_hat = I + alpha * VTV  # (H, W, C, C)
            return alpha * torch.linalg.inv(Sigma_hat)  # (H, W, C, C)

    def compute_Cs(self, V, Pi):  # (N, C, H, W), (K, N)
        with self.diff_ctx():
            N, C, H, W = V.shape  # (N, C, H, W)
            N, K = Pi.shape  # (K, N)
            VTV_per_class = F.gram_per_class_translation_invariant(V, Pi)  # (K, H, W, C, C)
            N_per_class = torch.sum(Pi, axis=0)  # (K, )
            gamma_per_class = N_per_class / N  # (K, )
            alpha_per_class = torch.where(  # stops divide by 0 errors
                N_per_class > 0.0,
                C / (self.eps_sq * N_per_class),  # (K, )
                torch.tensor(0.0)  # ()
            )  # (K, )
            alpha_per_class = alpha_per_class.view(K, 1, 1, 1, 1)  # (K, 1, 1, 1, 1)
            I = torch.eye(C).view(1, 1, 1, C, C)  # (1, 1, 1, C, C)
            Sigma_hat_per_class = I + alpha_per_class * VTV_per_class  # (K, H, W, C, C)
            return alpha_per_class * torch.linalg.inv(Sigma_hat_per_class)  # (K, H, W, C, C)

    def preprocess(self, V):
        with self.diff_ctx():
            return F.normalize_2d(F.fft2(V))

    def postprocess(self, V):
        with self.diff_ctx():
            return F.normalize_2d(F.ifft2(V).real.to(torch.float))

    def compute_mcr2(self, Z, Pi):
        with torch.no_grad():
            return (self.cr.DeltaR(Z, Pi), self.cr.R(Z), self.cr.Rc(Z, Pi))

__all__ = ["TranslationInvariantReduLayer"]
