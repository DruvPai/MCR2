import matplotlib.pyplot as plt
import torch
import mcr2
import mcr2.functional as F
import pytorch_lightning as pl

N = 1000
C = 7
H = 32
W = 32
L = 200
K = 2


class ReduNetTranslationInvariantModel(pl.LightningModule):
    def __init__(self):
        super(ReduNetTranslationInvariantModel, self).__init__()
        self.redu_layers = mcr2.redunet.ReduBlock(*[
            mcr2.redunet.layers.TranslationInvariantReduLayer(5.0, 0.5, 500.0, K, (C, H, W), False) for
            _ in range(L)
        ])
        self.initialized = False
        self.loss = mcr2.mcr2.SupervisedTranslationInvariantMCR2Loss(0.5)

    def forward(self, x):
        if self.initialized:
            return F.normalize_2d(self.redu_layers(x))
        else:
            raise RuntimeError("calling forward without initialization")

    def training_step(self, batch, batch_idx):
        if not self.initialized:
            x, y = batch["full"]
            x = F.normalize_2d(x)
            Pi = mcr2.functional.y_to_pi(y, K)
            self.redu_layers.init(x, Pi)
            self.initialized = True
        else:
            x, y = batch["mini"]
            z = self(x)
            Pi = mcr2.functional.y_to_pi(y, K)
            loss = self.loss(z, Pi)
            return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        mu_0 = 5 * torch.ones(size=(C, H, W))
        X_0 = torch.randn((N, C, H, W)) + mu_0
        y_0 = torch.zeros((N,), dtype=torch.long)
        mu_1 = -5 * torch.ones(size=(C, H, W))
        X_1 = torch.randn((N, C, H, W)) + mu_1
        y_1 = torch.ones((N,), dtype=torch.long)
        X = torch.cat((X_0, X_1), dim=0)
        y = torch.cat((y_0, y_1), dim=0)
        self.dataset = torch.utils.data.TensorDataset(X, y)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return {
            "full": torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset),
                                                shuffle=True),
            # "mini": torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        }

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def plot_training_DeltaR(model):
    losses = model.redu_layers.get_losses_per_layer()
    DeltaRs = [entry["DeltaR"] for entry in losses]
    Rs = [entry["R"] for entry in losses]
    Rcs = [entry["Rc"] for entry in losses]
    layers = [i + 1 for i in range(len(DeltaRs))]
    plt.plot(layers, Rs, label="R")
    plt.plot(layers, Rcs, label="Rc")
    plt.plot(layers, DeltaRs, label="Delta R")
    plt.legend()
    plt.show()
    plt.close()


data_module = SyntheticDataModule(200)
model = ReduNetTranslationInvariantModel()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, data_module)
plot_training_DeltaR(model)
