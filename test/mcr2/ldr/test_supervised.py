import unittest

import mcr2
import mcr2.functional as F
import torch

D = 500
N = 2000
K = 10

eps_sq = 0.5

cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)


def naive_binary_LDR_loss(Z, Z_hat):
    return cr.DeltaR_distance(Z, Z_hat)


def naive_multiclass_LDR_loss(Z, Z_hat, Pi):
    return cr.DeltaR(Z, Pi) + cr.DeltaR(Z_hat, Pi) + torch.sum(
        torch.tensor([cr.DeltaR_distance(Z[Pi[:, i] == 1], Z_hat[Pi[:, i] == 1]) for i in range(Pi.shape[1])]))


class TestSupervisedBinaryLDRLoss(unittest.TestCase):
    def test_forward(self):
        Z = torch.randn((N, D))
        Z_hat = torch.randn((N, D))
        loss = mcr2.ldr.SupervisedBinaryLDRLoss(eps_sq)
        self.assertTrue(torch.allclose(loss.forward(Z, Z_hat), naive_binary_LDR_loss(Z, Z_hat)))


class TestSupervisedMulticlassLDRLoss(unittest.TestCase):
    def test_forward(self):
        Z = torch.randn((N, D))
        Z_hat = torch.randn((N, D))
        Pi = F.y_to_pi(torch.randint(low=0, high=K, size=(N,)))
        loss = mcr2.ldr.SupervisedMulticlassLDRLoss(eps_sq)
        self.assertTrue(torch.allclose(loss.forward(Z, Z_hat, Pi), naive_multiclass_LDR_loss(Z, Z_hat, Pi)))


if __name__ == '__main__':
    unittest.main()
