import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from controllers import probabilistic_forward_dynamics as pfd


def gaussian_nll(mu_model, log_var_model, y_target):
    normalized_mse = (mu_model - y_target) ** 2 / torch.exp(log_var_model)
    return torch.mean(0.5 * (normalized_mse + log_var_model).sum(dim=1))


class EnsembleTrainer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_size: int,
        num_ensembles: int,
        lr: float,
        weight_decay: float = 0,
        batch_size: int = 32,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_size = history_size
        self.models = [
            pfd.ProbabilisticForwardDynamics(
                input_features=state_dim * history_size + action_dim * history_size,
                hidden_dims=[64, 64],
            )
            for _ in range(num_ensembles)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            for model in self.models
        ]
        self.batch_size = batch_size

    def train(self, num_epochs: int, train_dataset: Dataset):
        dataloaders = [
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for _ in self.models
        ]
        train_losses = [[] for _ in self.models]
        for epoch in range(num_epochs):
            for i, model in enumerate(self.models):
                for batch in tqdm(dataloaders[i], desc=f"Model: {i}, Epoch: {epoch}"):
                    x, y = batch
                    y_mean, y_log_var = model(x)
                    loss = gaussian_nll(y_mean, y_log_var, y.unsqueeze(1))
                    optimizer = self.optimizers[i]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses[i].append(loss.item())

        fig, ax = plt.subplots()
        for i, loss_vals in enumerate(train_losses):
            ax.plot(loss_vals, label=f"model {i}")
        ax.set_xlabel("step")
        ax.set_ylabel("Gaussian NLL loss")
        fig.savefig("ensemble_train_loss.png")

    def save_models(self):
        for i, model in enumerate(self.models):
            model.save_checkpoint(f"./models/dynamics_model{i}.pth")
