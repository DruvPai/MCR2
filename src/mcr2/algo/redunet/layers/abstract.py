import torch
import pytorch_lightning


class ReduLayer(pytorch_lightning.LightningModule):
    def __init__(self, eta, eps_sq, lmbda, num_classes, differentiable):
        super(ReduLayer, self).__init__()
        self.E = None
        self.Cs = None
        self.gam = None

        self.eta = eta
        self.eps_sq = eps_sq
        self.lmbda = lmbda

        self.K = num_classes

        self.diff = differentiable
        self.diff_ctx = lambda: torch.no_grad() if not self.diff else torch.enable_grad()

    def forward(self, V):
        raise NotImplementedError("Attempted to call forward with abstract ReduLayer")

    def zero(self):
        self.set_params(torch.zeros_like(self.E), torch.zeros_like(self.Cs))

    def init(self, V, Pi):
        gam = self.compute_gam(Pi)
        E = self.compute_E(V)
        Cs = self.compute_Cs(V, Pi)
        self.set_params(E, Cs, gam)

    def update(self, V, Pi, tau):
        E_ref, Cs_ref = self.E, self.Cs
        E_new = self.compute_E(V)
        Cs_new = self.compute_Cs(V, Pi)
        E_update = E_ref + tau * (E_new - E_ref)
        Cs_update = Cs_ref + tau * (Cs_new - Cs_ref)
        self.set_params(E_update, Cs_update)

    def set_params(self, E, Cs, gam=None):
        self.E = torch.nn.Parameter(E, requires_grad=self.E.requires_grad)
        self.Cs = torch.nn.Parameter(Cs, requires_grad=self.Cs.requires_grad)
        if gam is not None:
            self.gam = torch.nn.Parameter(gam, requires_grad=False)

__all__ = ["ReduLayer"]
