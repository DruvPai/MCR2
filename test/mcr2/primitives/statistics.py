import unittest
from unittest import TestCase

from torch import randint, randn, stack, zeros
from torch.nn.functional import one_hot
from src.mcr2.primitives.products import tensorized_ZtZ
from src.mcr2.primitives.statistics import second_moment, second_moment_class, gramian
from test.utils import seed, assert_tensors_almost_equal


B = 6
N = 10
D = 5
K = 3


def naive_second_moment(Z):
    N, D = Z.shape
    if N == 0:
        return zeros((D, D))
    else:
        return tensorized_ZtZ(Z) / N


def naive_second_moment_batched(Z):
    B, N, D = Z.shape
    sm = []
    for b in range(B):
        if N == 0:
            sm.append(zeros((D, D)))
        else:
            sm.append(tensorized_ZtZ(Z[b]) / N)
    return stack(sm, dim=0)


def naive_second_moment_class(Z, y, K=K):
    N, D = Z.shape
    sm = []
    for k in range(K):
        Zk = Z[y == k]
        Nk = Zk.shape[0]
        if Nk == 0:
            sm.append(zeros((D, D)))
        else:
            sm.append(tensorized_ZtZ(Zk) / Nk)
    return stack(sm, dim=0)


def naive_second_moment_class_batched(Z, y, K=K):
    B, N, D = Z.shape
    sm = []
    for b in range(B):
        sm_b = []
        for k in range(K):
            Zk = Z[b][y[b] == k]
            Nk = Zk.shape[0]
            if Nk == 0:
                sm_b.append(zeros((D, D)))
            else:
                sm_b.append(tensorized_ZtZ(Zk) / Nk)
        sm.append(stack(sm_b, dim=0))
    return stack(sm, dim=0)


def naive_gramian(Z):
    N, D = Z.shape
    return Z @ Z.T

def naive_gramian_batched(Z):
    B, N, D = Z.shape
    g = []
    if B == 0:
        return zeros(size=(B, N, N))
    for b in range(B):
        g.append(Z[b] @ Z[b].T)
    return stack(g, dim=0)


class TestSecondMoment(TestCase):
    def test_second_moment(self):
        seed()
        Z = randn((N, D))
        assert_tensors_almost_equal(self, naive_second_moment(Z), second_moment(Z))

    def test_second_moment_batched(self):
        seed()
        Z = randn((B, N, D))
        assert_tensors_almost_equal(self, naive_second_moment_batched(Z), second_moment(Z))


    def test_second_moment_zero(self):
        seed()
        Z = randn((0, D))
        assert_tensors_almost_equal(self, naive_second_moment(Z), second_moment(Z))

    def test_second_moment_batched_zero(self):
        seed()
        Z = randn((B, 0, D))
        assert_tensors_almost_equal(self, naive_second_moment_batched(Z), second_moment(Z))


class TestSecondMomentClass(TestCase):
    def test_second_moment_class(self):
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_second_moment_class(Z, y, K=K), second_moment_class(Z, y_onehot))

    def test_second_moment_class_batched(self):
        seed()
        Z = randn((B, N, D))
        y = randint(low=0, high=K, size=(B, N))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_second_moment_class_batched(Z, y, K=K), second_moment_class(Z, y_onehot))


    def test_second_moment_class_zero(self):
        K = 100
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_second_moment_class(Z, y, K=K), second_moment_class(Z, y_onehot))

    def test_second_moment_class_batched_zero(self):
        K = 100
        seed()
        Z = randn((B, N, D))
        y = randint(low=0, high=K, size=(B, N))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_second_moment_class_batched(Z, y, K=K), second_moment_class(Z, y_onehot))


class TestGramian(TestCase):
    def test_gramian(self):
        seed()
        Z = randn((N, D))
        assert_tensors_almost_equal(self, naive_gramian(Z), gramian(Z))

    def test_gramian_batched(self):
        seed()
        Z = randn((B, N, D))
        assert_tensors_almost_equal(self, naive_gramian_batched(Z), gramian(Z))


    def test_gramian_zero(self):
        seed()
        Z = randn((0, D))
        assert_tensors_almost_equal(self, naive_gramian(Z), gramian(Z))

    def test_gramian_batched_zero(self):
        seed()
        Z = randn((B, 0, D))
        assert_tensors_almost_equal(self, naive_gramian_batched(Z), gramian(Z))


if __name__ == '__main__':
    unittest.main()
