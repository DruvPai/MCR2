import unittest

import mcr2
import mcr2.functional as F
import torch

C = 5
T = 10
N = 200
K = 10

eps_sq = 0.5

cr = mcr2.coding_rate.SupervisedShiftInvariantCodingRate(eps_sq)


def naive_MCR2_loss(Z, Pi):
    return -cr.DeltaR(Z, Pi)


class TestSupervisedVectorMCR2Loss(unittest.TestCase):
    def test_forward(self):
        Z = F.fft(torch.randn((N, C, T)))
        Pi = F.y_to_pi(torch.randint(low=0, high=K, size=(N,)))
        loss = mcr2.mcr2.SupervisedShiftInvariantMCR2Loss(eps_sq)
        self.assertTrue(torch.allclose(loss.forward(Z, Pi), naive_MCR2_loss(Z, Pi)))


if __name__ == '__main__':
    unittest.main()
