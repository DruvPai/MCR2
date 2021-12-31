import matplotlib.pyplot as plt
import torch
import mcr2
import mcr2.functional as F
import pytorch_lightning as pl

D = 2
d = 500
N = 2000


class SyntheticGaussianModel(pl.LightningModule):
    def __init__(self):
        super(SyntheticGaussianModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, D),
        )
        self.loss = mcr2.objective.supervised.loss.VectorSupervisedMCR2Loss(0.5)
        self.DeltaR_while_training = []

    def forward(self, x):
        return F.normalize_vec(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        Pi = F.y_to_pi(y, 2)
        loss = self.loss(z, Pi)
        self.DeltaR_while_training.append(-loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        mu_0 = 5 * torch.ones(size=(D, ))
        X_0 = torch.randn((N, D)) + mu_0
        y_0 = torch.zeros((N,), dtype=torch.long)
        mu_1 = -5 * torch.ones(size=(D, ))
        X_1 = torch.randn((N, D)) + mu_1
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

def plot_dataset(x, y):
    t = torch.linspace(start=0, end=2 * 3.14159, steps=1000)
    x0 = x[y == 0]
    x1 = x[y == 1]
    mu_0 = torch.mean(x0, dim=0)
    mu_1 = torch.mean(x1, dim=0)
    ang = torch.arccos(torch.dot(mu_0, mu_1) / (torch.linalg.norm(mu_0) * torch.linalg.norm(mu_1)))
    print(ang)
    plt.scatter(x0[:, 0], x0[:, 1], label="x0")
    plt.scatter(x1[:, 0], x1[:, 1], label="x1")
    plt.plot(torch.cos(t), torch.sin(t), linewidth=1)
    plt.legend()
    plt.show()
    plt.close()

data_module = SyntheticDataModule(200)
model = SyntheticGaussianModel()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)

plot_DeltaR(model)
x, y = data_module.dataset[:]
z = model(x).detach()
plot_dataset(z, y)
