from unittest import TestCase

from torch import manual_seed, Tensor
from torch.linalg import norm


def seed():
    manual_seed(1)


def assert_tensors_almost_equal(test_case: TestCase, x1: Tensor, x2: Tensor):
    test_case.assertAlmostEqual(norm(x1 - x2).item(), 0.0, places=3)

__all__ = ["seed", "assert_tensors_almost_equal"]
