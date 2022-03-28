import mcr2
import torch


class SupervisedLDRLoss(torch.nn.Module):
    def __init__(self, eps_sq):
        super(SupervisedLDRLoss, self).__init__()
        self.cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)


class SupervisedBinaryLDRLoss(SupervisedLDRLoss):
    def forward(self, Z, Z_hat):
        return self.cr.DeltaR_distance(Z, Z_hat)


class SupervisedMulticlassLDRLoss(SupervisedLDRLoss):
    def forward(self, Z, Z_hat, Pi):
        N = Z.shape[0]
        N_per_class = torch.sum(Pi, axis=0)
        gamma_per_class = N_per_class / N

        R_Z = self.cr.R(Z)  # ()
        R_Zhat = self.cr.R(Z_hat)  # ()
        R_Z_per_class = self.cr.R_per_class(Z, Pi)  # (K, )
        R_Zhat_per_class = self.cr.R_per_class(Z_hat, Pi)  # (K, )

        R_Zi_Zhati = sum(self.cr.R(
            torch.cat(tensors=(Z[Pi[:, i] == 1], Z_hat[Pi[:, i] == 1]), dim=0)
        ) for i in range(Pi.shape[1]))  # TODO: optimize out CPU access

        return R_Z + R_Zhat + R_Zi_Zhati - torch.sum((gamma_per_class + 0.5) * (R_Z_per_class + R_Zhat_per_class))  # ()


__all__ = ["SupervisedBinaryLDRLoss", "SupervisedMulticlassLDRLoss"]
