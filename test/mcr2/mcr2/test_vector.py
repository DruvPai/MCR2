import unittest

import mcr2
import mcr2.functional as F
import torch

D = 500
N = 2000
K = 10

eps_sq = 0.5

cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)


def naive_MCR2_loss(Z, Pi):
    return -cr.DeltaR(Z, Pi)


class TestSupervisedVectorMCR2Loss(unittest.TestCase):
    def test_forward(self):
        Z = torch.randn((N, D))
        Pi = F.y_to_pi(torch.randint(low=0, high=K, size=(N,)))
        loss = mcr2.mcr2.SupervisedVectorMCR2Loss(eps_sq)
        self.assertTrue(torch.allclose(loss.forward(Z, Pi), naive_MCR2_loss(Z, Pi)))


if __name__ == '__main__':
    unittest.main()
