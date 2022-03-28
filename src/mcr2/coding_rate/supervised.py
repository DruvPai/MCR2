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
            Z: data matrix, (N, D)

        Returns:
            The coding rate of Z, ().
        """
        N, D = Z.shape  # (N, D)
        Sigma_hat_data = F.Sigma_hat_vec(Z)  # (D, D)
        beta = D / self.eps_sq  # ()
        I = torch.eye(D)  # (D, D)
        Sigma_hat_distorted = I + beta * Sigma_hat_data  # (D, D)
        return F.logdet(Sigma_hat_distorted) / 2.0  # ()

    def R_per_class(self, Z, Pi):
        """
        Computes the coding rate of each class of Z as given by class information matrix Pi.

        Args:
            Z: data matrix, (N, D)
            Pi: class matrix, (N, K)

        Returns:
            The coding rate of each class of Z, (K, ).
        """
        N, D = Z.shape  # (N, D)
        Sigma_hat_data_per_class = F.Sigma_hat_per_class_vec(Z, Pi)  # (K, D, D)
        beta = D / self.eps_sq  # ()
        I = torch.eye(D).view(1, D, D)  # (D, D)
        Sigma_hat_distorted_per_class = I + beta * Sigma_hat_data_per_class  # (K, D, D)
        logdets_per_class = F.logdet(Sigma_hat_distorted_per_class)  # (K, )
        return logdets_per_class / 2.0  # ()

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.

        Args:
            Z: data matrix, (N, D)
            Pi: class information matrix, (N, K)

        Returns:
            The segmented coding rate of Z with respect to Pi, ().
        """
        N, K = Pi.shape  # (N, K)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        Rs_per_class = self.R_per_class(Z, Pi)  # (K, )
        return torch.sum(gamma_per_class * Rs_per_class)  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.

        Args:
            Z: data matrix, (N, D)
            Pi: class information matrix, (K, N)

        Returns:
            The coding rate reduction of Z with respect to Pi, ().
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.

        Args:
            Z1: data matrix, (N1, D)
            Z2: data matrix, (N2, D)

        Returns:
            The DeltaR distance between Z1 and Z2, ().
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
            The coding rate of Z, ().
        """
        N, C, T = Z.shape  # (N, C, T)
        Sigma_hat_data = F.Sigma_hat_shift_invariant(Z)  # (T, C, C)
        beta = C / self.eps_sq  # ()
        I = torch.eye(C).view(1, C, C)  # (1, C, C)
        Sigma_hat_distorted = I + beta * Sigma_hat_data  # (T, C, C)
        return torch.sum(F.logdet(Sigma_hat_distorted)) / 2.0  # ()

    def R_per_class(self, Z, Pi):
        """
        Computes the coding rate of each class of Z as given by class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N, C, T)
            Pi: class matrix, (N, K)

        Returns:
            The coding rate of each class of Z, (K, ).
        """
        N, C, T = Z.shape  # (N, C, T)
        Sigma_hat_data_per_class = F.Sigma_hat_per_class_shift_invariant(Z, Pi)  # (K, T, C, C)
        beta = C / self.eps_sq  # ()
        I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
        Sigma_hat_distorted_per_class = I + beta * Sigma_hat_data_per_class  # (K, T, C, C)
        logdets_per_class = torch.sum(
            F.logdet(Sigma_hat_distorted_per_class),  # (K, T)
            dim=1
        )  # (K, )
        return logdets_per_class / 2.0  # ()

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N, C, T)
            Pi: class information matrix, (N, K)

        Returns:
            The segmented coding rate of Z with respect to Pi, ().
        """
        N, K = Pi.shape  # (N, K)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        Rs_per_class = self.R_per_class(Z, Pi)  # (K, )
        return torch.sum(gamma_per_class * Rs_per_class)  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z: data matrix, (N, C, T)
            Pi: class information matrix, (N, K)

        Returns:
            The coding rate reduction of Z with respect to Pi, ().
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.
        NOTE: Only an accurate measure of the distance if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft(Z).

        Args:
            Z1: data matrix, (N1, C, T)
            Z2: data matrix, (N2, C, T)

        Returns:
            The DeltaR distance between Z1 and Z2, ().
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
            The coding rate of Z, ().
        """
        N, C, H, W = Z.shape  # (N, C, H, W)
        Sigma_hat_data = F.Sigma_hat_translation_invariant(Z)  # (H, W, C, C)
        beta = C / self.eps_sq
        I = torch.eye(C).view(1, 1, C, C)  # (1, 1, C, C)
        Sigma_hat_distorted = I + beta * Sigma_hat_data  # (H, W, C, C)
        return torch.sum(F.logdet(Sigma_hat_distorted)) / 2.0  # ()

    def R_per_class(self, Z, Pi):
        """
        Computes the coding rate of each class of Z as given by class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)
            Pi: class matrix, (N, K)

        Returns:
            The coding rate of each class of Z, (K, ).
        """
        N, C, H, W = Z.shape  # (N, C, H, W)
        Sigma_hat_data_per_class = F.Sigma_hat_per_class_translation_invariant(Z, Pi)  # (K, H, W, C, C)
        beta = C / self.eps_sq  # ()
        I = torch.eye(C).view(1, 1, 1, C, C)  # (1, 1, 1, C, C)
        Sigma_hat_distorted_per_class = I + beta * Sigma_hat_data_per_class  # (K, H, W, C, C)
        logdets_per_class = torch.sum(
            F.logdet(Sigma_hat_distorted_per_class),  # (K, H, W)
            dim=(1, 2)
        )  # (K, )
        return logdets_per_class / 2.0  # (K, )

    def Rc(self, Z, Pi):
        """
        Computes the segmented coding rate of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)
            Pi: class information matrix, (N, K)

        Returns:
            The segmented coding rate of Z with respect to Pi, ().
        """
        N, K = Pi.shape  # (N, K)
        N_per_class = torch.sum(Pi, axis=0)  # (K, )
        gamma_per_class = N_per_class / N  # (K, )
        Rs_per_class = self.R_per_class(Z, Pi)  # (K, )
        return torch.sum(gamma_per_class * Rs_per_class)  # ()

    def DeltaR(self, Z, Pi):
        """
        Computes the coding rate reduction of Z with respect to a class information matrix Pi.
        NOTE: Only an accurate measure of the coding rate if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z: data matrix, (N, C, H, W)
            Pi: class information matrix, (N, K)

        Returns:
            The coding rate reduction of Z with respect to Pi, ().
        """
        return super().DeltaR(Z, Pi)

    def DeltaR_distance(self, Z1, Z2):
        """
        Computes the DeltaR distance between the two data point sets Z1 and Z2.
        NOTE: Only an accurate measure of the distance if the data is FFTed beforehand.
        You can do this with mcr2.functional.fft2(Z).

        Args:
            Z1: data matrix, (N1, C, H, W)
            Z2: data matrix, (N2, C, H, W)

        Returns:
            The DeltaR distance between Z1 and Z2, ().
        """
        return super().DeltaR_distance(Z1, Z2)


__all__ = [
    "SupervisedVectorCodingRate",
    "SupervisedShiftInvariantCodingRate",
    "SupervisedTranslationInvariantCodingRate"
]
