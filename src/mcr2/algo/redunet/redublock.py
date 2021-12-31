import torch
import mcr2.functional as F

class ReduBlock(torch.nn.Module):
    # ReduNet Architecture. This class is used to stack a series of ReduLayers.

    def __init__(self, *layers):
        super(ReduBlock, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.losses = []

    def init(self, X, Pi):
        # Initialize the network. Using inputs and labels, it constructs
        # the parameters E and Cs throughout each ReduLayer.
        return self.forward(X, Pi,
                            init=True,
                            loss=True)

    def update(self, X, Pi, tau=0.1):
        # Update the network parameters E and Cs by
        # performing a moving average.
        return self.forward(X, Pi, tau=tau,
                            update=True,
                            loss=True)

    def zero(self):
        # Set every network parameters E and Cs to a zero matrix.
        for layer in self.layers:
            layer.zero()

    def forward(self, X, Pi=None, tau=0.1,
                init=False,
                update=False,
                loss=False):
        for idx, layer in enumerate(self.layers):
            # preprocess for redunet layers
            if idx == 0:
                X = layer.preprocess(X)

            # If init is set to True, then initialize
            # layer using inputs and labels
            if init:
                layer.init(X, Pi)

            # If update is set to True, then update
            # layer using inputs and labels
            if update:
                layer.update(X, Pi, tau)

            # Perform a forward pass
            X = layer.forward(X)

            # compute MCR quantities for redunet layer
            if loss:
                mcr_losses = layer.compute_mcr2(X, Pi)
                self._log_losses(idx, *mcr_losses)

            # postprocess for redunet layers
            if idx == len(self.layers) - 1:
                X = layer.postprocess(X)
        return X

    def _log_losses(self, layer_idx, DeltaR, R, Rc):
        self.losses.append(
            {"layer_idx": layer_idx, "DeltaR": DeltaR, "R": R, "Rc": Rc}
        )

    def get_losses_per_layer(self):
        return self.losses

__all__ = ["ReduBlock"]
