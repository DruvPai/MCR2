import unittest

import mcr2
import mcr2.functional as F
import torch

C = 5
H = 3
W = 3
N = 200
K = 10

eps_sq = 0.5

cr = mcr2.coding_rate.SupervisedTranslationInvariantCodingRate(eps_sq)


def naive_MCR2_loss(Z, Pi):
    return -cr.DeltaR(Z, Pi)


class TestSupervisedTranslationInvariantMCR2Loss(unittest.TestCase):
    def test_forward(self):
        Z = F.fft2(torch.randn((N, C, H, W)))
        Pi = F.y_to_pi(torch.randint(low=0, high=K, size=(N,)))
        loss = mcr2.mcr2.SupervisedTranslationInvariantMCR2Loss(eps_sq)
        self.assertTrue(torch.allclose(loss.forward(Z, Pi), naive_MCR2_loss(Z, Pi)))


if __name__ == '__main__':
    unittest.main()
