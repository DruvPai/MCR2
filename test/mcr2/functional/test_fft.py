import unittest

import torch
from mcr2.functional.fft import *


class TestFFT(unittest.TestCase):
    n = 50
    c = 20
    t = 10
    h = 32
    w = 32

    def test_fft_invertible_1d(self):
        Z = torch.randn((self.n, self.c, self.t))
        self.assertTrue(torch.allclose(Z, ifft(fft(Z)).real, atol=1e-6, rtol=1e-3))
        self.assertTrue(torch.allclose(Z, fft(ifft(Z)).real, atol=1e-6, rtol=1e-3))

    def test_fft_invertible_2d(self):
        Z = torch.randn((self.n, self.c, self.h, self.w))
        self.assertTrue(torch.allclose(Z, ifft2(fft2(Z)).real, atol=1e-6, rtol=1e-3))
        self.assertTrue(torch.allclose(Z, fft2(ifft2(Z)).real, atol=1e-6, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
