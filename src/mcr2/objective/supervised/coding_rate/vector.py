import mcr2.functional as F
import opt_einsum
import torch

from .abstract import CodingRate


class VectorCodingRate(CodingRate):
    def R(self, Z):
        """
        Computes the coding rate of Z.

        Args:
            Z: data matrix, (N x D)
            input_in_fourier_basis: whether the input is in Fourier basis; e.g. already FFTed

        Returns:
            The coding rate of Z.
        """
        N, D = Z.shape  # (N, D)
        ZTZ = F.gram_vec(Z)  # (D, D)
        alpha = D / (N * self.eps_sq)
        I = torch.eye(D)  # (D, D)
        Sigma_hat = I + alpha * ZTZ  # (D, D)
        return F.logdet(Sigma_hat) / 2.0  # ()

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.

        Args:
            Z: data matrix, (N x D)
            Pi: class information matrix, (K x N)
            input_in_fourier_basis: whether the input is in Fourier basis; e.g. already FFTed

        Returns:
            The segmented coding rate of Z with respect to Pi.
        """
        N, D = Z.shape  # (N, D)
        N, K = Pi.shape  # (N, K)
        ZTZ_per_class = F.gram_per_class_vec(Z, Pi)  # (K, D, D)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        alpha_per_class = torch.where(  # stops divide by 0 errors
            N_per_class > 0.0,
            D / (self.eps_sq * N_per_class),  # (K, )
            torch.tensor(0.0)  # ()
        )  # (K, )
        I = torch.eye(D).view(1, D, D)  # (D, D)
        Sigma_hat_per_class = I + alpha_per_class.view(K, 1, 1) * ZTZ_per_class  # (K, D, D)
        logdets_per_class = F.logdet(Sigma_hat_per_class)  # (K, )
        return torch.sum(gamma_per_class * logdets_per_class) / 2.0  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.

        Args:
            Z: data matrix, (N x D)
            Pi: class information matrix, (K x N)
            input_in_fourier_basis: whether the input is in Fourier basis; e.g. already FFTed

        Returns:
            The coding rate reduction of Z with respect to Pi.
        """
        return self.R(Z) - self.Rc(Z, Pi)

__all__ = ["VectorCodingRate"]
