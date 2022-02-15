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
        K = Pi.shape[1]
        RZ = self.cr.R(Z)
        RZ_hat = self.cr.R(Z_hat)
        RZ_Z_hat_concat = self.cr.R(torch.cat(tensors=(Z, Z_hat), dim=0))
        Z_per_class = [Z[Pi[:, i] == 1] for i in range(K)]
        Z_hat_per_class = [Z_hat[Pi[:, i] == 1] for i in range(K)]
        return RZ + RZ_hat + RZ_Z_hat_concat - sum(
            ((Z_per_class[i].shape[0] / N) + (1 / 2)) * self.cr.DeltaR_distance(Z_per_class[i], Z_hat_per_class[i]) for
            i in range(K)
        )

    def forward_unoptimized(self, Z, Z_hat, Pi):
        Z_per_class = [Z[Pi[:, i] == 1] for i in range(Pi.shape[1])]
        Z_hat_per_class = [Z_hat[Pi[:, i] == 1] for i in range(Pi.shape[1])]
        return self.cr(Z) + self.cr(Z_hat) + \
               sum(self.cr.DeltaR_distance(Z_per_class[i], Z_hat_per_class[i]) for i in range(Pi.shape[1]))


__all__ = ["SupervisedBinaryLDRLoss", "SupervisedMulticlassLDRLoss"]
