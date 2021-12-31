from mcr2.functional.class_info import y_to_pi as pkg_y_to_pi, pi_to_y as pkg_pi_to_y
import torch
import unittest

def naive_y_to_pi(y, K=-1):
    if K == -1:
        K = torch.max(y) + 1
    N = y.shape[0]
    Pi = torch.zeros(size=(N, K), dtype=torch.long)
    for i in range(N):
        Pi[i][y[i]] = 1
    return Pi

def naive_pi_to_y(Pi):
    N, K = Pi.shape
    y = torch.zeros(size=(N,), dtype=torch.long)
    for i in range(N):
        for j in range(K):
            if Pi[i][j] == 1:
                y[i] = j
    return y

class TestClassInfo(unittest.TestCase):
    def test_y_to_pi(self):
        n = 50
        K_informative = 10
        K_missingclasses = 100
        y_informative = torch.randint(low=0, high=K_informative, size=(n, ))
        y_missingclasses = torch.randint(low=0, high=K_missingclasses, size=(n, ))
        self.assertTrue(torch.allclose(naive_y_to_pi(y_informative), pkg_y_to_pi(y_informative)))
        self.assertTrue(torch.allclose(naive_y_to_pi(y_missingclasses), pkg_y_to_pi(y_missingclasses)))
        self.assertTrue(torch.allclose(naive_y_to_pi(y_informative, K_informative), pkg_y_to_pi(y_informative, K_informative)))
        self.assertTrue(torch.allclose(naive_y_to_pi(y_missingclasses, K_missingclasses), pkg_y_to_pi(y_missingclasses, K_missingclasses)))

    def test_pi_to_y(self):
        n = 50
        K_informative = 10
        K_missingclasses = 100
        y_informative = torch.randint(low=0, high=K_informative, size=(n, ))
        y_missingclasses = torch.randint(low=0, high=K_missingclasses, size=(n, ))
        Pi_informative = naive_y_to_pi(y_informative)
        Pi_missingclasses = naive_y_to_pi(y_missingclasses)
        self.assertTrue(torch.allclose(naive_pi_to_y(Pi_informative), pkg_pi_to_y(Pi_informative)))
        self.assertTrue(torch.allclose(naive_pi_to_y(Pi_missingclasses), pkg_pi_to_y(Pi_missingclasses)))


if __name__ == '__main__':
    unittest.main()
