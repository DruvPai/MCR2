import matplotlib.pyplot as plt
import torch
import mcr2
import mcr2.functional as F
import pytorch_lightning as pl

C = 20
H = 32
W = 32
d = 500
N = 2000


class SyntheticGaussianModel(pl.LightningModule):
    def __init__(self):
        super(SyntheticGaussianModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(C * H * W, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, C * H * W),
            torch.nn.Unflatten(dim=1, unflattened_size=(C, H, W))
        )
        self.loss = mcr2.objective.supervised.loss.TranslationInvariantSupervisedMCR2Loss(0.5)
        self.DeltaR_while_training = []

    def forward(self, x):
        return F.normalize_2d(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        Pi = F.y_to_pi(y, 2)
        z = self(x)
        loss = self.loss(F.fft2(z), Pi)
        self.DeltaR_while_training.append(-loss.item())
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
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def plot_DeltaR(model):
    DeltaRs = model.DeltaR_while_training
    steps = list(range(len(DeltaRs)))
    plt.plot(steps, DeltaRs, label="DeltaR")
    plt.legend()
    plt.show()
    plt.close()


data_module = SyntheticDataModule(200)
model = SyntheticGaussianModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, data_module)

plot_DeltaR(model)
