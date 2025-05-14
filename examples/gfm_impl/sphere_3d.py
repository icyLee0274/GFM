import logging
from itertools import product
from math import sqrt

from cvxpy import NonPos
from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
import cvxpy as cp
import geoopt

import gfm
import examples
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class Sphere3D(examples.GfmExampleBase):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.manifold: geoopt.Manifold | None = None

    def get_manifold(self):
        if self.manifold is None:
            self.manifold = geoopt.Sphere(torch.eye(3, device=self.device))
        return self.manifold

    @torch.no_grad()
    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        A = tensor([
            [-2., 0., 0.],
            [2., 4., 0.],
            [2., -4., 0.],
        ], device=self.device)
        b = tensor([1., 1., 1.], device=self.device)
        domain = gfm.LinearConstraint(A, b)  # A x <= b
        ip = tensor([0., 0., 1.], device=self.device)
        return domain, ip

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        domain, ip = self._init_domain()
        data = torch.zeros(n, 3, device=self.device)
        vs = torch.zeros(n, 3, device=self.device)
        dist = gfm.SumDistribution(
            MultivariateNormal(torch.zeros(2, device=self.device), .1 * torch.eye(2, device=self.device)),
            MultivariateNormal(tensor([.1, .1], device=self.device), .1 * torch.eye(2, device=self.device)),
            MultivariateNormal(tensor([.1, -.1], device=self.device), .1 * torch.eye(2, device=self.device)),
            MultivariateNormal(tensor([-.1, .1], device=self.device), .15 * torch.eye(2, device=self.device)),
            MultivariateNormal(tensor([-.1, -.1], device=self.device), .15 * torch.eye(2, device=self.device)),
            weights=tensor([.1, .2, .2, .25, .25], device=self.device),
            device=self.device,
        )
        fs = torch.zeros(n, dtype=torch.bool, device=self.device)
        manifold = self.get_manifold()

        while not torch.all(fs):
            n_gen = n - fs.sum().item()
            vs[~fs, :2] = dist.sample([n_gen])
            data[~fs] = manifold.expmap(ip.expand(n_gen, -1), vs[~fs])
            fs = domain.check_feasibility_v(data, self.device)

        return data

    def transform(self, data: Tensor) -> Tensor:
        manifold = self.get_manifold()
        ip = self.get_interior_point().to(data)
        vs = manifold.logmap(ip.expand(data.shape[0], -1), data)
        zs = super().transform(vs)
        return zs

    def inverse_transform(self, zs: Tensor) -> Tensor:
        vs = super().inverse_transform(zs)
        manifold = self.get_manifold()
        ip = self.get_interior_point().to(vs)
        xs = manifold.expmap(ip.expand(zs.shape[0], -1), vs)
        return xs
