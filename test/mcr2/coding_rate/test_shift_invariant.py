import torch
import mcr2
import mcr2.functional as F
import unittest


N = 50
C = 30
T = 20
K = 10


def naive_R(Z, eps_sq):
    n, c, t = Z.shape
    I = torch.eye(c).unsqueeze(0)
    alpha = c / (n * eps_sq)
    return 0.5 * torch.sum(F.logdet(I + alpha * torch.einsum("nct, ndt -> tcd", Z, Z.conj())))


def naive_Rc(Z, Pi, eps_sq):
    n, c, t = Z.shape
    y = F.pi_to_y(Pi)
    out = torch.tensor(0.0)
    for i in range(torch.max(y) + 1):
        n_i = torch.sum(y == i)
        if n_i > 0:
            out += (n_i / n) * naive_R(Z[y == i], eps_sq)
    return out


def naive_DeltaR(Z, Pi, eps_sq):
    return naive_R(Z, eps_sq) - naive_Rc(Z, Pi, eps_sq)


def naive_DeltaR_distance(Z1, Z2, eps_sq):
    return naive_R(torch.cat((Z1, Z2), dim=0), eps_sq) \
           - (Z1.shape[0] / (Z1.shape[0] + Z2.shape[0])) * naive_R(Z1, eps_sq) \
           - (Z2.shape[0] / (Z1.shape[0] + Z2.shape[0])) * naive_R(Z2, eps_sq)


def naive_DeltaR_distance2(Z1, Z2, eps_sq):
    Pi = torch.zeros((Z1.shape[0] + Z2.shape[0], 2))
    Pi[0:Z1.shape[0], 0] = 1
    Pi[Z1.shape[0]:Z1.shape[0] + Z2.shape[0], 1] = 1
    return naive_DeltaR(torch.cat(tensors=(Z1, Z2), dim=0), Pi, eps_sq)


class TestShiftInvariantCodingRate(unittest.TestCase):
    eps_sq = 0.5
    cr = mcr2.coding_rate.SupervisedShiftInvariantCodingRate(eps_sq)

    def test_R(self):
        Z = torch.randn((N, C, T))
        self.assertTrue(torch.allclose(self.cr.R(Z), naive_R(Z, self.eps_sq)))

    def test_Rc(self):
        Z = torch.randn((N, C, T))
        y = torch.randint(low=0, high=K, size=(N, ))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(self.cr.Rc(Z, Pi), naive_Rc(Z, Pi, self.eps_sq)))

    def test_DeltaR(self):
        Z = torch.randn((N, C, T))
        y = torch.randint(low=0, high=K, size=(N, ))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(self.cr.DeltaR(Z, Pi), naive_DeltaR(Z, Pi, self.eps_sq)))

    def test_DeltaR_distance(self):
        Z1 = torch.randn((N + 1, C, T))
        Z2 = torch.randn((N - 1, C, T))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance(Z1, Z2, self.eps_sq)))
        self.assertTrue(torch.allclose(self.cr.DeltaR_distance(Z1, Z2), naive_DeltaR_distance2(Z1, Z2, self.eps_sq)))


if __name__ == '__main__':
    unittest.main()
