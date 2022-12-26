import unittest
from unittest import TestCase

from torch import eye, logdet, randn
from mcr2.primitives.logdet import logdet_I_plus
from mcr2.primitives.statistics import second_moment
from test.utils import seed, assert_tensors_almost_equal

B = 6
N = 10
D = 5
eps = 0.5


def naive_logdet_I_plus(H):
    return logdet(eye(H.shape[0]) + H)

def naive_logdet_I_plus_batched(H):
    return logdet(eye(H.shape[1]).unsqueeze(0) + H)


class TestLogdet(TestCase):
    def test_logdet(self):
        seed()
        Z = randn((N, D))
        Q = (D / (eps ** 2)) * second_moment(Z)
        assert_tensors_almost_equal(self, naive_logdet_I_plus(Q), logdet_I_plus(Q))

    def test_logdet_batch(self):
        seed()
        Z = randn((B, N, D))
        Q = (D / (eps ** 2)) * second_moment(Z)
        assert_tensors_almost_equal(self, naive_logdet_I_plus_batched(Q), logdet_I_plus(Q))

    def test_logdet_zero(self):
        seed()
        Z = randn((0, D))
        Q = (D / (eps ** 2)) * second_moment(Z)
        assert_tensors_almost_equal(self, naive_logdet_I_plus(Q), logdet_I_plus(Q))

    def test_logdet_batch_zero(self):
        seed()
        Z = randn((B, 0, D))
        Q = (D / (eps ** 2)) * second_moment(Z)
        assert_tensors_almost_equal(self, naive_logdet_I_plus_batched(Q), logdet_I_plus(Q))

        Z = randn((0, 0, D))
        Q = (D / (eps ** 2)) * second_moment(Z)
        assert_tensors_almost_equal(self, naive_logdet_I_plus_batched(Q), logdet_I_plus(Q))


if __name__ == '__main__':
    unittest.main()
