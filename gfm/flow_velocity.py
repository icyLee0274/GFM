import torch
from torch import nn, Tensor

__all__ = ["FlowVelocity", "MlpVelocityField"]


class MlpVelocityField(nn.Module):

    def __init__(self, dim: int = 2, width: int = 64, depth: int = 2, activation: str = "ELU"):
        """
        Create a nn velocity field that computes $v(t, x)$.

        :param dim: Dimension of the input x.
        :param width: Width of hidden layers.
        :param depth: Number of hidden layers.
        :param activation: Activation function.
        """
        super().__init__()
        activation = getattr(nn, activation)
        hidden = [nn.Linear(dim + 1, width), activation()]
        for _ in range(depth):
            hidden.append(nn.Linear(width, width))
            hidden.append(activation())
        hidden.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*hidden)

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        if t.dim() == 1: t = t.view(-1, 1)
        return self.net(torch.cat((t, x_t), dim=1))


class FlowVelocity(nn.Module):

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        return self.net(torch.cat((t, x_t), dim=-1))


class FlowVelocityDeep3(nn.Module):
    def __init__(self, dim: int = 2, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        return self.net(torch.cat((t, x_t), dim=-1))
