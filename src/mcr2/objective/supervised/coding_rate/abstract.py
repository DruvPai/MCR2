class CodingRate:
    def __init__(self, eps_sq):
        self.eps_sq = eps_sq

    def R(self, Z):
        raise NotImplementedError("Called R in basic coding rate class")

    def Rc(self, Z, Pi):
        raise NotImplementedError("Called Rc in basic coding rate class")

    def DeltaR(self, Z, Pi):
        raise NotImplementedError("Called DeltaR in basic coding rate class")

__all__ = ["CodingRate"]
