from torch import cat, eye, Tensor

from mcr2.primitives.logdet import logdet_hpd
from mcr2.primitives.statistics import second_moment_class
from mcr2.primitives.coding_rate import DeltaR_diff


def supervised_ctrl_loss(Z: Tensor, Zhat: Tensor, y_onehot: Tensor, eps: float):
    """
    Computes the supervised CTRL loss from "Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction" by Dai et al.

    Args:
        Z: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        Zhat: Tensor of shape (N, D); a matrix where each row is a data point and each column is a feature.
        y_onehot: Tensor of shape (N, K); a {0, 1} matrix where each row is a data point and each column is a class.
        eps: Float; quantization error of the rate distortion.

    Returns:
        Tensor of shape (); the supervised CTRL loss between Z and Zhat with respect to y.
    """
    N, D = Z.shape
    N, K = y_onehot.shape

    Nc = y_onehot.float().sum(dim=-2).unsqueeze(-1).unsqueeze(-1)  # (K, 1, 1)
    pi = Nc / N  # (K, 1, 1)

    Pc = second_moment_class(Z, y_onehot)  # (K, D, D)
    Phatc = second_moment_class(Zhat, y_onehot)  # (K, D, D)
    Pcomc = (Pc + Phatc) / 2.0  # (K, D, D)

    P = (pi * Pc).sum(dim=0, keepdims=True)  # (1, D, D)
    Phat = (pi * Phatc).sum(dim=0, keepdims=True)  # (1, D, D)

    P_tot = cat((P, Phat, Pc, Phatc, Pcomc), dim=0)  # (3K + 2, D, D)

    Q_tot = eye(D, device=Z.device).unsqueeze(0) + D / (eps ** 2) * P_tot  # (3K + 2, D, D)
    ld_Q = logdet_hpd(Q_tot)  # (3K + 2, )

    pi = pi.squeeze(-1).squeeze(-1)  # (K, )
    return 0.5 * (ld_Q[0] + ld_Q[1] - ((pi + 0.5) * (ld_Q[2:K+2] + ld_Q[K+2:2*K+2])).sum() + ld_Q[2*K+2:].sum())  # ()


def unsupervised_ctrl_loss(Z, Zhat, eps):
    return DeltaR_diff(Z, Zhat, eps)


__all__ = ["supervised_ctrl_loss", "unsupervised_ctrl_loss"]
