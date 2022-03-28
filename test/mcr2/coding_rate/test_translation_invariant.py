import unittest

import mcr2
import mcr2.functional as F
import torch

N = 50
C = 3
H = 3
W = 3
K = 10
eps_sq = 0.5


def naive_R(Z):
    n, c, h, w = Z.shape
    I = torch.eye(c).unsqueeze(0).unsqueeze(0)
    if n == 0:
        return F.logdet(I)
    alpha = c / (n * eps_sq)
    return 0.5 * torch.sum(F.logdet_complex(I + alpha * torch.einsum("nchw, ndhw -> hwcd", Z.conj(), Z)))


def naive_Rc(Z, Pi):
    n, c, h, w = Z.shape
    y = F.pi_to_y(Pi)
    out = torch.tensor(0.0)
    for i in range(torch.max(y) + 1):
        n_i = torch.sum(y == i)
        if n_i > 0:
            out += (n_i / n) * naive_R(Z[y == i])
    return out


def naive_DeltaR(Z, Pi):
    return naive_R(Z) - naive_Rc(Z, Pi)


def naive_DeltaR_distance(Z1, Z2):
    return naive_R(torch.cat((Z1, Z2), dim=0)) \
           - (Z1.shape[0] / (Z1.shape[0] + Z2.shape[0])) * naive_R(Z1) \
           - (Z2.shape[0] / (Z1.shape[0] + Z2.shape[0])) * naive_R(Z2)


def naive_DeltaR_distance2(Z1, Z2):
    Pi = torch.zeros((Z1.shape[0] + Z2.shape[0], 2))
    Pi[0:Z1.shape[0], 0] = 1
    Pi[Z1.shape[0]:Z1.shape[0] + Z2.shape[0], 1] = 1
    return naive_DeltaR(torch.cat(tensors=(Z1, Z2), dim=0), Pi)


class TestTranslationInvariantCodingRate(unittest.TestCase):
    cr = mcr2.coding_rate.SupervisedTranslationInvariantCodingRate(eps_sq)

    def test_R(self):
        Z = F.fft2(torch.randn((N, C, H, W)))
        self.assertTrue(torch.allclose(self.cr.R(Z), naive_R(Z), atol=1e-3))

    def test_Rc(self):
        Z = F.fft2(torch.randn((N, C, H, W)))
        y = torch.randint(low=0, high=K, size=(N,))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(self.cr.Rc(Z, Pi), naive_Rc(Z, Pi), atol=1e-3))

    def test_DeltaR(self):
        Z = F.fft2(torch.randn((N, C, H, W)))
        y = torch.randint(low=0, high=K, size=(N,))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(self.cr.DeltaR(Z, Pi), naive_DeltaR(Z, Pi), atol=1e-3))

    def test_DeltaR_distance(self):
        Z1 = F.fft2(torch.randn((N + 1, C, H, W)))
        Z2 = F.fft2(torch.randn((N - 1, C, H, W)))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance(Z1, Z2), atol=1e-3))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance2(Z1, Z2), atol=1e-3))
        Z1 = F.fft2(torch.randn((N, C, H, W)))
        Z2 = F.fft2(torch.randn((N, C, H, W)))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance(Z1, Z2), atol=1e-3))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance2(Z1, Z2), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
