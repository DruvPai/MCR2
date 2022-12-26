from torch import cat, eye, maximum, stack, tensor, Tensor

from mcr2.primitives.logdet import logdet_hpd
from mcr2.primitives.statistics import second_moment, second_moment_class


def R(Z: Tensor, eps: float):
    """
    Computes the "coding rate" of the input matrix with respect to the given quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate of Z.
    """
    N, D = Z.shape

    P = second_moment(Z)  # (D, D)

    Q = eye(D, device=Z.device) + D / (eps ** 2) * P  # (D, D)
    ld_Q = logdet_hpd(Q)   # ()

    return 0.5 * ld_Q  # ()


def Rc(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the "segmented coding rate" of the input matrix with respect to the given class assignment and quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian conditional on their class.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (N, K); a {0, 1} matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the segmented coding rate of Z with respect to y.
    """
    N, D = Z.shape

    Nc = y_onehot.float().sum(dim=0)  # (K, )
    pi = Nc / N  # (K, )

    P = second_moment_class(Z, y_onehot)  # (K, D, D)
    Q = eye(D, device=Z.device).unsqueeze(0) + D / (eps ** 2) * P  # (K, D, D)
    ld_Q = logdet_hpd(Q)  # (K, )
    return 0.5 * (pi * ld_Q).sum()  # ()


def DeltaR(Z: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the "coding rate reduction" of the input matrix with respect to the given class assignment and quantization error, assuming that the rows of the input matrix are distributed according to a zero-mean Gaussian conditional on their class.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (N, K); a {0, 1} matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate reduction of Z with respect to y.
    """
    N, D = Z.shape

    Nc = y_onehot.float().sum(dim=0).unsqueeze(-1).unsqueeze(-1)  # (K, )
    pi = Nc / N  # (K, )

    P = second_moment_class(Z, y_onehot)  # (K, D, D)
    P_com = (pi * P).sum(dim=0, keepdims=True)  # (1, D, D)
    P_tot = cat((P, P_com), dim=0)  # (K + 1, D, D)

    Q_tot = eye(D, device=Z.device).unsqueeze(0) + D / (eps ** 2) * P_tot  # (K + 1, D, D)
    ld_Q = logdet_hpd(Q_tot)  # (K + 1, )

    pi = pi.squeeze(-1).squeeze(-1)  # (K, )
    return 0.5 * (ld_Q[-1] - (pi * ld_Q[:-1]).sum())  # ()


def DeltaR_diff(Z1: Tensor, Z2: Tensor, eps: float):
    """
    Computes the "coding rate reduction" difference between the two input matrices with respect to the given quantization error, assuming that the rows of the input matrices are distributed according to a zero-mean Gaussian.

    Args:
        Z1: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        Z2: Tensor of shape (M, D); a matrix where each row is a data point and each column is a feature.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate reduction difference between Z and Zhat.
    """
    N, D = Z1.shape
    M, D = Z2.shape

    N = tensor(float(N), device=Z1.device).unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    M = tensor(float(M), device=Z1.device).unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    T = maximum(M + N, tensor(1.0, device=Z1.device))  # (1, 1)

    P_Z1 = second_moment(Z1)  # (D, D)
    P_Z2 = second_moment(Z2)  # (D, D)
    P_com = (N * P_Z1 + M * P_Z2) / T  # (D, D)
    P_tot = stack((P_Z1, P_Z2, P_com), dim=0)  # (3, D, D)

    Q_tot = eye(D, device=Z1.device).unsqueeze(0) + D / (eps ** 2) * P_tot  # (3, D, D)
    ld_Q = logdet_hpd(Q_tot)  # (3, )

    N = N.squeeze(-1).squeeze(-1)  # ()
    M = M.squeeze(-1).squeeze(-1)  # ()
    T = T.squeeze(-1).squeeze(-1)  # ()

    return 0.5 * (ld_Q[2] - (N / T) * ld_Q[0] - (M / T) * ld_Q[1])  # ()


def DeltaR_cdiff(Z1: Tensor, Z2: Tensor, y1_onehot: Tensor, y2_onehot: Tensor, eps: float):
    """
    Computes the "coding rate reduction" difference between the two input matrices with respect to the given class assignments and quantization error, assuming that the rows of the input matrices are distributed according to a zero-mean Gaussian conditioned on class.

    Args:
        Z1: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        Z2: Tensor of shape (M, D); a matrix where each row is a data point and each column is a feature.
        y1_onehot: Tensor of shape (N, K); a matrix where each row is a data point and each column is a class.
        y2_onehot: Tensor of shape (M, K); a matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the coding rate reduction difference between Z and Zhat with respect to y1 and y2.
    """
    N, D = Z1.shape
    M, D = Z2.shape

    Nc = y1_onehot.float().sum(dim=0).unsqueeze(-1).unsqueeze(-1)  # (K, 1, 1)
    Mc = y2_onehot.float().sum(dim=0).unsqueeze(-1).unsqueeze(-1)  # (K, 1, 1)
    Tc = maximum(Nc + Mc, tensor(1.0, device=Z1.device))  # (K, 1, 1)

    P_Z1 = second_moment_class(Z1, y1_onehot)  # (K, D, D)
    P_Z2 = second_moment_class(Z2, y2_onehot)  # (K, D, D)
    P_com = (Nc * P_Z1 + Mc * P_Z2) / Tc  # (K, D, D)
    P_tot = stack((P_Z1, P_Z2, P_com), dim=0)  # (3, K, D, D)

    Q_tot = eye(D, device=Z1.device).unsqueeze(0).unsqueeze(0) + D / (eps ** 2) * P_tot  # (3, K, D, D)
    ld_Q = logdet_hpd(Q_tot)  # (3, K)

    Nc = Nc.squeeze(-1).squeeze(-1)  # (K, )
    Mc = Mc.squeeze(-1).squeeze(-1)  # (K, )
    Tc = Tc.squeeze(-1).squeeze(-1)  # (K, 1, 1)

    return 0.5 * (ld_Q[2] - (Nc / Tc) * ld_Q[0] - (Mc / Tc) * ld_Q[1]).sum()  # ()


__all__ = ["R", "Rc", "DeltaR", "DeltaR_diff", "DeltaR_cdiff"]
