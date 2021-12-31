from mcr2.functional import logdet as pkg_logdet
from torch.linalg import slogdet
import unittest
import torch


def naive_logdet(X):
    return slogdet(X)[0] * slogdet(X)[1]

class TestLogdet(unittest.TestCase):
    n = 50
    d = 30
    k = 10
    alpha = 0.1

    def test_logdet_psd(self):
        Z = torch.randn((self.n, self.d))
        X = torch.einsum("ni, nj -> ij", Z, Z.conj())
        logdet_fast = pkg_logdet(X)
        self.assertTrue(torch.allclose(naive_logdet(X), logdet_fast))

    def test_logdet_batch_psd(self):
        Z = torch.randn((self.k, self.n, self.d))
        X = torch.einsum("kni, knj -> kij", Z, Z.conj())
        logdet_fast = pkg_logdet(X)
        self.assertTrue(torch.allclose(naive_logdet(X), logdet_fast))

    def test_logdet_pd(self):
        Z = torch.randn((self.n, self.d))
        X = torch.eye(self.d) + self.alpha * torch.einsum("ni, nj -> ij", Z, Z.conj())
        logdet_fast = pkg_logdet(X)
        self.assertTrue(torch.allclose(naive_logdet(X), logdet_fast))

    def test_logdet_batch_pd(self):
        Z = torch.randn((self.k, self.n, self.d))
        X = torch.eye(self.d).unsqueeze(0) + self.alpha * torch.einsum("kni, knj -> kij", Z, Z.conj())
        logdet_fast = pkg_logdet(X)
        self.assertTrue(torch.allclose(naive_logdet(X), logdet_fast))


if __name__ == '__main__':
    unittest.main()
