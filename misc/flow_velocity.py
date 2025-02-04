import torch
from torch import nn, Tensor

__all__ = ["FlowVelocity"]


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
    def __init__(self, dim: int = 2, h: int = 64):
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
