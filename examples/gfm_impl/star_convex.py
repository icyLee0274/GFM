from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor

import gfm
import examples
from gfm import ConstrainedSet


class StarDomain(gfm.ConstrainedSet):

    def __init__(self, alpha: float, n_tips: int):
        super().__init__()
        self.alpha = alpha
        self.n_tips = n_tips

    def gamma(self, xs: Tensor) -> Tensor:
        return 1 + self.alpha * torch.sin(self.n_tips * torch.atan2(xs[:, 1], xs[:, 0]))

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        lhs = torch.linalg.vector_norm(points, dim=-1)
        rhs = self.gamma(points)
        return lhs <= rhs

    def check_feasibility(self, point: Tensor) -> bool:
        return self.check_feasibility_v(point.view(1, -1)).item()

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device()) -> Tensor:
        return self.gamma(vs)


class Star(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        return StarDomain(self.cfg.example.alpha, self.cfg.example.n_tips), \
            torch.zeros(2, device=self.cfg.accelerator, dtype=torch.float32)

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        dim = 2
        device = self.cfg.accelerator

        thetas = torch.linspace(0, 2 * torch.pi, self.cfg.example.n_tips + 1, device=device)[:-1]
        avgs = torch.stack([
            torch.cos(thetas),
            torch.sin(thetas),
        ], dim=1)

        cov = (self.cfg.example.scale * torch.eye(dim, device=device))

        dists = [torch.distributions.MultivariateNormal(torch.zeros(dim, device=device), cov)]
        for avg in avgs:
            dists.append(torch.distributions.MultivariateNormal(avg, cov))

        data_dist = gfm.TruncatedDistribution(gfm.SumDistribution(*dists), self.get_domain())
        data = data_dist.sample(torch.Size([n]))

        return data
