import mcr2.functional as F
import torch


class SupervisedCodingRate:
    def __init__(self, eps_sq):
        self.eps_sq = eps_sq

    def R(self, Z):
        raise NotImplementedError("Called R in basic coding rate class")

    def Rc(self, Z, Pi):
        raise NotImplementedError("Called Rc in basic coding rate class")

    def DeltaR(self, Z, Pi):
        return self.R(Z) - self.Rc(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        N1 = Z1.shape[0]
        N2 = Z2.shape[0]
        N = N1 + N2
        return self.R(torch.cat(tensors=(Z1, Z2), dim=0)) - (N1 / N) * self.R(Z1) - (N2 / N) * self.R(Z2)


class SupervisedVectorCodingRate(SupervisedCodingRate):
    def R(self, Z):
        """
        Computes the coding rate of Z.

        Args:
            Z: data matrix, (N x D)

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

        Returns:
            The coding rate reduction of Z with respect to Pi.
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.

        Args:
            Z1: data matrix, (N1 x D)
            Z2: data matrix, (N2 x D)

        Returns:
            The DeltaR distance between Z1 and Z2.
        """
        return super().DeltaR_distance(Z1, Z2)


class SupervisedShiftInvariantCodingRate(SupervisedCodingRate):
    def R(self, Z):
        """
        Computes the coding rate of Z.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N, C, T)

        Returns:
            The coding rate of Z.
        """
        N, C, T = Z.shape  # (N, C, T)
        ZTZ = F.gram_shift_invariant(Z)  # (T, C, C)
        alpha = C / (N * self.eps_sq)
        I = torch.eye(C).view(1, C, C)  # (1, C, C)
        Sigma_hat = I + alpha * ZTZ  # (T, C, C)
        return torch.sum(F.logdet(Sigma_hat)) / 2.0  # ()

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N, C, T)
            Pi: class information matrix, (N, K)

        Returns:
            The segmented coding rate of Z with respect to Pi.
        """
        N, C, T = Z.shape  # (N, C, T)
        N, K = Pi.shape  # (N, K)
        ZTZ_per_class = F.gram_per_class_shift_invariant(Z, Pi)  # (K, T, C, C)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        alpha_per_class = torch.where(  # stops divide by 0 errors
            N_per_class > 0,
            C / (self.eps_sq * N_per_class),  # (K, )
            torch.tensor(0.0)  # ()
        )  # (K, )
        I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
        Sigma_hat_per_class = I + alpha_per_class.view(K, 1, 1, 1) * ZTZ_per_class  # (K, T, C, C)
        logdets_per_class = torch.sum(
            F.logdet(Sigma_hat_per_class),  # (K, T)
            dim=1
        )  # (K, )
        return torch.sum(gamma_per_class * logdets_per_class) / 2.0  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N x C x T)
            Pi: class information matrix, (K x N)

        Returns:
            The coding rate reduction of Z with respect to Pi.
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.
        NOTE: Only an accurate measure of the distance if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z1: data matrix, (N1 x C x T)
            Z2: data matrix, (N2 x C x T)

        Returns:
            The DeltaR distance between Z1 and Z2.
        """
        return super().DeltaR_distance(Z1, Z2)


class SupervisedTranslationInvariantCodingRate(SupervisedCodingRate):
    def R(self, Z):
        """
        Computes the coding rate of Z.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)

        Returns:
            The coding rate of Z.
        """
        N, C, H, W = Z.shape  # (N, C, H, W)
        ZTZ = F.gram_translation_invariant(Z)  # (H, W, C, C)
        alpha = C / (N * self.eps_sq)
        I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
        Sigma_hat = I + alpha * ZTZ  # (H, W, C, C)
        return torch.sum(F.logdet(Sigma_hat)) / 2.0  # ()

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)
            Pi: class information matrix, (N, K)

        Returns:
            The segmented coding rate of Z with respect to Pi.
        """
        N, C, H, W = Z.shape  # (N, C, H, W)
        N, K = Pi.shape  # (N, K)
        ZTZ_per_class = F.gram_per_class_translation_invariant(Z, Pi)  # (K, H, W, C, C)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        alpha_per_class = torch.where(  # stops divide by 0 errors
            N_per_class > 0,
            C / (self.eps_sq * N_per_class),  # (K, )
            torch.tensor(0.0)  # ()
        )  # (K, )
        I = torch.eye(C).view(1, 1, 1, C, C)  # (1, 1, 1, C, C)
        Sigma_hat_per_class = I + alpha_per_class.view(K, 1, 1, 1, 1) * ZTZ_per_class  # (K, H, W, C, C)
        logdets_per_class = torch.sum(
            F.logdet(Sigma_hat_per_class),  # (K, H, W)
            dim=(1, 2)
        )  # (K, )
        return torch.sum(gamma_per_class * logdets_per_class) / 2.0  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)
            Pi: class information matrix, (N, K)

        Returns:
            The coding rate reduction of Z with respect to Pi.
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.
        NOTE: Only an accurate measure of the distance if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z1: data matrix, (N1 x C x H x W)
            Z2: data matrix, (N2 x C x H x W)

        Returns:
            The DeltaR distance between Z1 and Z2.
        """
        return super().DeltaR_distance(Z1, Z2)


__all__ = [
    "SupervisedVectorCodingRate",
    "SupervisedShiftInvariantCodingRate",
    "SupervisedTranslationInvariantCodingRate"
]
