import unittest

import torch
from mcr2.functional import normalize_frob as pkg_normalize, normalize_vec, normalize_1d, normalize_2d
from torch.linalg import norm


class TestNormalize(unittest.TestCase):
    n = 50
    d = 30
    c = 10
    t = 20
    h = 32
    w = 32

    def test_normalize_order_2(self):
        Z = torch.randn((self.n, self.d))
        self.assertTrue(torch.allclose(norm(pkg_normalize(Z)), torch.tensor(1.0)))

    def test_normalize_order_3(self):
        Z = torch.randn((self.n, self.c, self.t))
        self.assertTrue(torch.allclose(norm(pkg_normalize(Z)), torch.tensor(1.0)))

    def test_normalize_order_4(self):
        Z = torch.randn((self.n, self.c, self.h, self.w))
        self.assertTrue(torch.allclose(norm(pkg_normalize(Z)), torch.tensor(1.0)))

    def test_normalize_vec(self):
        Z = torch.randn((self.n, self.d))
        self.assertTrue(torch.allclose(norm(normalize_vec(Z)[0]), torch.tensor(1.0)))

    def test_normalize_1d(self):
        Z = torch.randn((self.n, self.c, self.t))
        self.assertTrue(torch.allclose(norm(normalize_1d(Z)[0]), torch.tensor(1.0)))

    def test_normalize_2d(self):
        Z = torch.randn((self.n, self.c, self.h, self.w))
        self.assertTrue(torch.allclose(norm(normalize_2d(Z)[0]), torch.tensor(1.0)))


if __name__ == '__main__':
    unittest.main()
