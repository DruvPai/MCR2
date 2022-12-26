import unittest
from unittest import TestCase

from torch import randn, randint
from torch.nn.functional import one_hot

from mcr2.primitives.coding_rate import DeltaR, DeltaR_diff, DeltaR_cdiff
from mcr2.losses.ctrl import supervised_ctrl_loss, unsupervised_ctrl_loss
from test.utils import seed, assert_tensors_almost_equal


N = 10
D = 5
K = 3
eps = 0.5

def naive_supervised_ctrl_loss(Z, Zhat, y_onehot, eps):
    return DeltaR(Z, y_onehot, eps) + DeltaR(Zhat, y_onehot, eps) + DeltaR_cdiff(Z, Zhat, y_onehot, y_onehot, eps)


def naive_unsupervised_ctrl_loss(Z, Zhat, eps):
    return DeltaR_diff(Z, Zhat, eps)


class TestSupervisedCTRLLoss(TestCase):
    def test_supervised_ctrl_loss(self):
        seed()
        Z = randn((N, D))
        Zhat = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self,
                                    supervised_ctrl_loss(Z, Zhat, y_onehot, eps),
                                    naive_supervised_ctrl_loss(Z, Zhat, y_onehot, eps)
                                    )

    def test_supervised_ctrl_loss_zero(self):
        seed()
        K = 100
        Z = randn((N, D))
        Zhat = randn((N, D))
        y = randint(low=0, high=K, size=(N, ))
        y_onehot = one_hot(y, num_classes=K)
        assert_tensors_almost_equal(self,
                                    supervised_ctrl_loss(Z, Zhat, y_onehot, eps),
                                    naive_supervised_ctrl_loss(Z, Zhat, y_onehot, eps)
                                    )

class TestUnsupervisedCTRLLoss(TestCase):
    def test_unsupervised_ctrl_loss(self):
        seed()
        Z = randn((N, D))
        Zhat = randn((N, D))
        assert_tensors_almost_equal(self,
                                    unsupervised_ctrl_loss(Z, Zhat, eps),
                                    naive_unsupervised_ctrl_loss(Z, Zhat, eps))


if __name__ == '__main__':
    unittest.main()
