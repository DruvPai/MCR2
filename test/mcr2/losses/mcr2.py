import unittest
from unittest import TestCase

from torch import randn, randint
from torch.nn.functional import one_hot

from src.mcr2.primitives.coding_rate import DeltaR
from src.mcr2.losses.mcr2 import supervised_mcr2_loss
from test.utils import seed, assert_tensors_almost_equal

N = 10
D = 5
K = 3
eps = 0.5


def naive_supervised_mcr2_loss(Z, y_onehot, eps):
    return DeltaR(Z, y_onehot, eps)


class TestSupervisedMCR2Loss(TestCase):
    def test_supervised_mcr2_loss(self):
        seed()
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_supervised_mcr2_loss(Z, y_onehot, eps),
                                    supervised_mcr2_loss(Z, y_onehot, eps))

    def test_supervised_mcr2_loss_zero(self):
        seed()
        K = 100
        Z = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self, naive_supervised_mcr2_loss(Z, y_onehot, eps),
                                    supervised_mcr2_loss(Z, y_onehot, eps))



if __name__ == '__main__':
    unittest.main()
