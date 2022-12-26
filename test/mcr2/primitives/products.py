import unittest
from unittest import TestCase

from torch import randint, randn, stack, zeros
from torch.nn.functional import one_hot
from src.mcr2.primitives.products import tensorized_ZtZ, tensorized_ZtZ_class
from test.utils import seed, assert_tensors_almost_equal

B = 6
N = 10
D = 5
K = 3


def naive_ZtZ(Z):
    N, D = Z.shape
    return Z.T @ Z

def naive_ZtZ_batched(Z):
    B, N, D = Z.shape
    return stack([
        Z[b].T @ Z[b] for b in range(B)
    ], dim=0)

def naive_ZtZ_class(Z, y, K=K):
    N, D = Z.shape
    return stack([
        Z[y == k].T @ Z[y == k] for k in range(K)
    ], dim=0)

def naive_ZtZ_class_batched(Z, y, K=K):
    B, N, D = Z.shape
    return stack([
        stack([
            Z[b][y[b] == k].T @ Z[b][y[b] == k] for k in range(K)
        ], dim=0)
        for b in range(B)
    ], dim=0)



class TestZtZ(TestCase):
    def test_ZtZ(self):
        seed()
        Z = randn((N, D))
        assert_tensors_almost_equal(self, naive_ZtZ(Z), tensorized_ZtZ(Z))


    def test_ZtZ_batch(self):
        seed()
        Z = randn((B, N, D))
        assert_tensors_almost_equal(self, naive_ZtZ_batched(Z), tensorized_ZtZ(Z))


    def test_ZtZ_zero(self):
        seed()
        Z = randn((0, D))
        assert_tensors_almost_equal(self, zeros((D, D)), tensorized_ZtZ(Z))


    def test_ZtZ_batch_zero(self):
        seed()
        Z = randn((B, 0, D))
        assert_tensors_almost_equal(self, zeros((B, D, D)), tensorized_ZtZ(Z))
        Z = randn((0, 0, D))
        assert_tensors_almost_equal(self, zeros((0, D, D)), tensorized_ZtZ(Z))


class TestZtZ_class(TestCase):
    def test_ZtZ_class(self):
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_ZtZ_class(Z, y, K=K), tensorized_ZtZ_class(Z, y_onehot))


    def test_ZtZ_class_batch(self):
        seed()
        Z = randn((B, N, D))
        y = randint(low=0, high=K, size=(B, N))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_ZtZ_class_batched(Z, y, K=K), tensorized_ZtZ_class(Z, y_onehot))


    def test_ZtZ_class_zero(self):
        K = 100
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_ZtZ_class(Z, y, K=K), tensorized_ZtZ_class(Z, y_onehot))


    def test_ZtZ_class_batch_zero(self):
        K = 100
        seed()
        Z = randn((B, N, D))
        y = randint(low=0, high=K, size=(B, N))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_ZtZ_class_batched(Z, y, K=K), tensorized_ZtZ_class(Z, y_onehot))


if __name__ == '__main__':
    unittest.main()
