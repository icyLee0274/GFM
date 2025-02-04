import torch, time, os
from sympy.stats.crv_types import UniformDistribution
from torch import tensor, Tensor
from torch.cpu import device_count
from torch.xpu import device
from torchdiffeq import odeint

from CvxIneq import GaugeMap, ball_reflect, ball_projection, unit_ball_mirror_map, unit_ball_dual_map
from CvxIneq.odeint_reflect import odeint_reflect
from .example import Example
from misc import *


class Hypercube(Example):

    def __init__(self, args, dim):
        super().__init__(args)
        self.dim = dim
        self.init_domain()

    def init_domain(self):
        self.domain = LinearConstraint(
            torch.vstack([torch.eye(self.dim), -torch.eye(self.dim)]).to(self.device),
            torch.ones(2 * self.dim, device=self.device),
        )
        self.gauge_map = GaugeMap(self.domain, torch.zeros(self.dim, device=self.device))
        if self.method == "gauge_mirror":
            self.prior_dist = torch.distributions.MultivariateNormal(
                torch.zeros(self.dim, device=self.device),
                torch.eye(self.dim, device=self.device),
            )
        elif self.method.startswith("gauge"):
            self.prior_dist = UnitBallUniform(self.dim)
        else:
            self.prior_dist = HyperBoxUniform(
                -torch.ones(self.dim, device=self.device),
                2 * torch.ones(self.dim, device=self.device)
            )

    def init_training(self):
        if self.gen_sample:
            nz = torch.linspace(0, self.dim, self.dim, dtype=torch.int)
            dists = []
            for i in range(self.dim):
                loc = torch.full([self.dim], 0.9, device=self.device)
                loc[0:nz[i]] = -0.9
                dists.append(torch.distributions.MultivariateNormal(
                    loc, .3 * torch.eye(self.dim, device=self.device)
                ))
            data_dist = TruncatedDistribution(SumDistribution(*dists), self.domain)
            self.true_samples = data_dist.sample(torch.Size([self.n_sample]))
        else:
            self.true_samples = torch.load(os.path.join(self.output, f"true_samples.pt"))
        if self.method == "gauge_mirror":
            self.training_samples = unit_ball_mirror_map(self.gauge_map.to_disk(self.true_samples))
        elif self.method.startswith("gauge"):
            self.training_samples = self.gauge_map.to_disk(self.true_samples)
        else:
            self.training_samples = self.true_samples
        self.velocity = FlowVelocity(self.dim, self.hidden)

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
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=ball_projection)[-1]
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


def cube_project(o: Tensor, v: Tensor) -> Tensor:
    e = o + v
    e[e > 1] = 1
    e[e < -1] = -1
    return e


def cube_reflect(o: Tensor, v: Tensor) -> Tensor:
    e = o + v
    e[e > 1] = 2 - e[e > 1]
    e[e < -1] = -2 - e[e < -1]
    return e
