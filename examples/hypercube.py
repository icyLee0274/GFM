import torch, time, os
from torch import tensor, Tensor
from torchdiffeq import odeint

from .example import Example
from misc import *
from CvxIneq import *


class Hypercube(Example):

    def __init__(self, args, dim: int):
        super().__init__(args, dim)
        self.init_domain()

    def init_domain(self):
        self.domain = LinearConstraint(
            torch.vstack([torch.eye(self.dim), -torch.eye(self.dim)]).to(self.device),
            torch.ones(2 * self.dim, device=self.device),
        )
        self.gauge_map = GaugeMap(self.domain, torch.zeros(self.dim, device=self.device))
        self.init_prior(-1, 2)
        if self.verbose:
            print("Domain initialized.")

    def data_distribution(self):
        nz = torch.linspace(0, self.dim, self.dim, dtype=torch.int)
        dists = []
        for i in range(self.dim):
            loc = torch.full([self.dim], 0.9, device=self.device)
            loc[0:nz[i]] = -0.9
            dists.append(torch.distributions.MultivariateNormal(
                loc, .3 * torch.eye(self.dim, device=self.device)
            ))
        data_dist = TruncatedDistribution(SumDistribution(*dists), self.domain)
        return data_dist

    def gen0(self) -> float:
        z_0 = self.prior_dist.sample(torch.Size([self.n_gen]))
        ts = torch.linspace(0, 1, self.n_step, device=self.device)
        start = time.time()
        match self.method:
            case "vanilla":
                x_1 = odeint(self.velocity, z_0, ts)[-1]
            case "reflect":
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=cube_reflect)[-1]
            case "project":
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=cube_project)[-1]
            case "gauge_vanilla":
                z_1 = odeint(self.velocity, z_0, ts)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_reflect":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=ball_reflect)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_project":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=ball_project)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_mirror":
                y_1 = odeint(self.velocity, z_0, ts)[-1]
                z_1 = unit_ball_dual_map(y_1)
                x_1 = self.gauge_map.from_disk(z_1)
            case _:
                raise RuntimeError(f"Unsupported method: {self.method}")
        end = time.time()
        self.gen_x_1 = x_1
        return end - start
