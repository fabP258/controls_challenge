import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ProbabilisticForwardDynamics(nn.Module):

    def __init__(
        self,
        input_features: int,
        hidden_dims: list | tuple,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_dims = hidden_dims
        layers = []
        prev_dim = input_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(Swish())
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 2)

    def parameter_count(self) -> int:
        num_parameters = 0
        attributes = ["mlp", "output_layer"]
        for attribute in attributes:
            attr = getattr(self, attribute)
            if isinstance(attr, nn.Module):
                num_parameters += sum(
                    p.numel() for p in attr.parameters() if p.requires_grad
                )
        return num_parameters

    def forward(self, x):
        h = self.mlp(x)
        h = self.output_layer(h)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    def save_checkpoint(self, checkpoint_path: str):
        checkpoint = {}
        checkpoint.update(
            {
                "args": {
                    "input_features": self.input_features,
                    "hidden_dims": self.hidden_dims,
                }
            }
        )
        checkpoint.update({"state_dict": self.state_dict()})
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        args = checkpoint["args"]
        model = cls(**args)
        model.load_state_dict(checkpoint["state_dict"])
        return model
