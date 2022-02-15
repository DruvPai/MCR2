import torch
import mcr2.functional as F
import unittest


def naive_inner_product_vec(Z1, Z2):
    return Z1 @ Z2.T.conj()


def naive_inner_product_shift_invariant(Z1, Z2):
    return torch.einsum("mct, nct -> mn", Z1, Z2)


def naive_inner_product_translation_invariant(Z1, Z2):
    return torch.einsum("mchw, nchw -> mn", Z1, Z2)


def naive_gram_vec(X):
    return X.T @ X.conj()


def naive_gram_per_class_vec(X, Pi):
    N, D = X.shape
    N, K = Pi.shape
    Sigma = torch.zeros((K, D, D))
    y = F.pi_to_y(Pi)
    for i in range(K):
        Xi = X[y == i]
        Sigma[i] = Xi.T @ Xi.conj()
    return Sigma


def naive_gram_shift_invariant(X):
    return torch.einsum("nct, ndt -> tcd", X, X.conj())


def naive_gram_per_class_shift_invariant(X, Pi):
    N, C, T = X.shape
    N, K = Pi.shape
    Sigma_per_class = torch.zeros((K, T, C, C))
    y = F.pi_to_y(Pi)
    for i in range(K):
        Xi = X[y == i]
        Sigma_per_class[i] = naive_gram_shift_invariant(Xi)
    return Sigma_per_class


def naive_gram_translation_invariant(X):
    return torch.einsum("nchw, ndhw -> hwcd", X, X.conj())


def naive_gram_per_class_translation_invariant(X, Pi):
    N, C, H, W = X.shape
    N, K = Pi.shape
    Sigma_per_class = torch.zeros((K, H, W, C, C))
    y = F.pi_to_y(Pi)
    for i in range(K):
        Xi = X[y == i]
        Sigma_per_class[i] = naive_gram_translation_invariant(Xi)
    return Sigma_per_class


class TestInnerProducts(unittest.TestCase):
    def test_inner_vec(self):
        M, N, D = 20, 30, 40
        Z1 = torch.randn((M, D))
        Z2 = torch.randn((N, D))
        self.assertTrue(torch.allclose(naive_inner_product_vec(Z1, Z2), F.inner_product_vec(Z1, Z2), rtol=1e-3))
        self.assertTrue(torch.allclose(naive_inner_product_vec(Z1, Z1), F.inner_product_vec(Z1), rtol=1e-3))

    def test_inner_shift_invariant(self):
        M, N, C, T = 20, 30, 5, 20
        Z1 = torch.randn((M, C, T))
        Z2 = torch.randn((N, C, T))
        self.assertTrue(torch.allclose(naive_inner_product_shift_invariant(Z1, Z2), F.inner_product_shift_invariant(Z1, Z2), rtol=1e-3))
        self.assertTrue(torch.allclose(naive_inner_product_shift_invariant(Z1, Z1), F.inner_product_shift_invariant(Z1), rtol=1e-3))

    def test_inner_translation_invariant(self):
        M, N, C, H, W = 20, 30, 5, 3, 3
        Z1 = torch.randn((M, C, H, W))
        Z2 = torch.randn((N, C, H, W))
        self.assertTrue(torch.allclose(naive_inner_product_translation_invariant(Z1, Z2), F.inner_product_translation_invariant(Z1, Z2), rtol=1e-3))
        self.assertTrue(torch.allclose(naive_inner_product_translation_invariant(Z1, Z1), F.inner_product_translation_invariant(Z1), rtol=1e-3))

class TestGramian(unittest.TestCase):
    def test_gramian_vec(self):
        N, D, K = 30, 40, 10
        Z = torch.randn((N, D))
        y = torch.randint(low=0, high=K, size=(N, ))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(naive_gram_vec(Z), F.gram_vec(Z)))
        self.assertTrue(torch.allclose(naive_gram_per_class_vec(Z, Pi), F.gram_per_class_vec(Z, Pi)))

    def test_gramian_shift_invariant(self):
        N, C, T, K = 30, 5, 10, 10
        Z = torch.randn((N, C, T))
        y = torch.randint(low=0, high=K, size=(N, ))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(naive_gram_shift_invariant(Z), F.gram_shift_invariant(Z)))
        self.assertTrue(torch.allclose(naive_gram_per_class_shift_invariant(Z, Pi), F.gram_per_class_shift_invariant(Z, Pi)))

    def test_gramian_translation_invariant(self):
        N, C, H, W, K = 30, 5, 32, 32, 10
        Z = torch.randn((N, C, H, W))
        y = torch.randint(low=0, high=K, size=(N, ))
        Pi = F.y_to_pi(y, K)
        self.assertTrue(torch.allclose(naive_gram_translation_invariant(Z), F.gram_translation_invariant(Z)))
        self.assertTrue(torch.allclose(naive_gram_per_class_translation_invariant(Z, Pi), F.gram_per_class_translation_invariant(Z, Pi)))

if __name__ == '__main__':
    unittest.main()
