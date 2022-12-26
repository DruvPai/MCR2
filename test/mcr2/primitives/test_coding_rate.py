import unittest
from unittest import TestCase

from torch import cat, eye, randint, randn
from torch.nn.functional import one_hot

from mcr2.primitives.statistics import second_moment
from mcr2.primitives.logdet import logdet_I_plus
from mcr2.primitives.coding_rate import R, Rc, DeltaR, DeltaR_diff, DeltaR_cdiff
from test.utils import seed, assert_tensors_almost_equal



B = 6
N = 10
M = 12
D = 5
K = 3
eps = 0.5

def naive_R(Z, eps=eps):
    N, D = Z.shape
    return 0.5 * logdet_I_plus(D / (eps ** 2) * second_moment(Z))


def naive_Rc(Z, y, eps=eps, K=K):
    N, D = Z.shape
    return sum(len(Z[y == k]) / N * naive_R(Z[y == k], eps) for k in range(K))


def naive_DeltaR(Z, y, eps=eps, K=K):
    return naive_R(Z, eps) - naive_Rc(Z, y, eps, K)



def naive_DeltaR_diff(Z1, Z2, eps=eps):
    N, D = Z1.shape
    M, D = Z2.shape
    T = max(M + N, 1)
    return naive_R(cat((Z1, Z2), dim=0), eps) - (N / T) * naive_R(Z1, eps) - (M / T) * naive_R(Z2, eps)


def naive_DeltaR_cdiff(Z1, Z2, y1, y2, eps=eps, K=K):
    return sum(naive_DeltaR_diff(Z1[y1 == k], Z2[y2 == k], eps) for k in range(K))


class TestR(TestCase):
    def test_R(self):
        seed()
        Z = randn((N, D))
        assert_tensors_almost_equal(self, naive_R(Z, eps), R(Z, eps))

    def test_R_zero(self):
        seed()
        Z = randn((0, D))
        assert_tensors_almost_equal(self, naive_R(Z, eps), R(Z, eps))


class TestRc(TestCase):
    def test_Rc(self):
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_Rc(Z, y, eps, K), Rc(Z, y_onehot, eps))

    def test_Rc_zero(self):
        K = 100
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_Rc(Z, y, eps, K), Rc(Z, y_onehot, eps))


class TestDeltaR(TestCase):
    def test_DeltaR(self):
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_DeltaR(Z, y, eps, K), DeltaR(Z, y_onehot, eps))

    def test_DeltaR_zero(self):
        K = 100
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_DeltaR(Z, y, eps, K), DeltaR(Z, y_onehot, eps))


class TestDeltaRDiff(TestCase):
    def test_DeltaR_diff(self):
        seed()
        Z1 = randn((N, D))
        Z2 = randn((M, D))
        assert_tensors_almost_equal(self, naive_DeltaR_diff(Z1, Z2, eps), DeltaR_diff(Z1, Z2, eps))

    def test_DeltaR_diff_zero(self):
        seed()
        Z1 = randn((N, D))
        Z2 = randn((0, D))
        assert_tensors_almost_equal(self, naive_DeltaR_diff(Z1, Z2, eps), DeltaR_diff(Z1, Z2, eps))
        Z1 = randn((0, D))
        Z2 = randn((M, D))
        assert_tensors_almost_equal(self, naive_DeltaR_diff(Z1, Z2, eps), DeltaR_diff(Z1, Z2, eps))
        Z1 = randn((0, D))
        Z2 = randn((0, D))
        assert_tensors_almost_equal(self, naive_DeltaR_diff(Z1, Z2, eps), DeltaR_diff(Z1, Z2, eps))


class TestDeltaRCDiff(TestCase):
    def test_DeltaR_cdiff(self):
        seed()
        Z1 = randn((N, D))
        Z2 = randn((M, D))
        y1 = randint(low=0, high=K, size=(N, ))
        y2 = randint(low=0, high=K, size=(M, ))
        y1_onehot = one_hot(y1, num_classes=K)
        y2_onehot = one_hot(y2, num_classes=K)
        assert_tensors_almost_equal(self, naive_DeltaR_cdiff(Z1, Z2, y1, y2, eps, K), DeltaR_cdiff(Z1, Z2, y1_onehot, y2_onehot, eps))


    def test_DeltaR_cdiff_zero(self):
        seed()
        K = 100
        Z1 = randn((N, D))
        Z2 = randn((M, D))
        y1 = randint(low=0, high=K, size=(N, ))
        y2 = randint(low=0, high=K, size=(M, ))
        y1_onehot = one_hot(y1, num_classes=K)
        y2_onehot = one_hot(y2, num_classes=K)
        assert_tensors_almost_equal(self, naive_DeltaR_cdiff(Z1, Z2, y1, y2, eps, K), DeltaR_cdiff(Z1, Z2, y1_onehot, y2_onehot, eps))


if __name__ == '__main__':
    unittest.main()
