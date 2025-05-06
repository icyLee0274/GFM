import torch, os, time
from torch import tensor, Tensor
from torch.distributions import MultivariateNormal

from .example import Example
from misc import *
from gfm import *


# 0. change to p_inf

# random generate QP / SCOP
# For each we train and evaluate ...

# domain feasibility: tolerance

# low-dim -> complex
# 2. complex 2d example: compare cube/ball
# 3. compare opt ip & bdr ip

# scale data range

# time - related difference

# interior point selection

# initial distribution


class Polytope2D(Example):

    def __init__(self, args):
        super().__init__(args, 2)
        self.projector = None
        self.reflect_fn = None
        self.init_domain()

    def init_domain(self):
        self.domain = LinearConstraint(
            tensor([
                [2., 1.],
                [-2., 1.],
                [1., -1.],
                [-1., -1.],
            ], device=self.device),
            tensor([2., 2., 1., 1.], device=self.device),
        )
        self.gauge_map = GaugeMap(self.domain, torch.zeros(2, device=self.device))
        if self.method == "gauge_mirror":
            self.prior_dist = MultivariateNormal(
                torch.zeros(2, device=self.device),
                torch.eye(2, device=self.device),
            )
        elif self.method.startswith("gauge"):
            self.prior_dist = HyperBallUniform(2)
        else:
            self.prior_dist = TruncatedDistribution(
                HyperBoxUniform(torch.full([2], -1, device=self.device),
                                tensor([2, 3], device=self.device)),
                self.domain
            )
        if self.verbose: print("Domain initialized")

    def data_distribution(self):
        return TruncatedDistribution(
            SumDistribution(
                MultivariateNormal(tensor([0, -.5], device=self.device),
                                   tensor([[.3, 0], [0, .3]], device=self.device)),
                MultivariateNormal(tensor([0, 1.], device=self.device),
                                   tensor([[.2, 0], [0, .6]], device=self.device)),
            ),
            self.domain
        )

    def gen0(self) -> float:
        z_0 = self.prior_dist.sample(torch.Size([self.n_gen]))
        ts = torch.linspace(0, 1, self.n_step, device=self.device)
        start = time.time()
        match self.method:
            case "vanilla":
                x_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
            case "reflect":
                if self.reflect_fn is None:
                    self.reflect_fn = PolytopeReflector(self.domain.At, self.domain.b)
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.reflect_fn)[-1]
            case "project":
                # raise NotImplementedError("Projected method not implemented yet.")
                if self.projector is None:
                    self.projector = PolytopeProjector(self.domain.At, self.domain.b)
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.projector)[-1]
            case "gauge_vanilla.yaml":
                z_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_reflect":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=ball_reflect)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_project.yaml":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=ball_project)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_mirror":
                y_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
                z_1 = unit_ball_dual_map(y_1)
                x_1 = self.gauge_map.from_disk(z_1)
            case _:
                raise RuntimeError(f"Unsupported method: {self.method}")
        end = time.time()
        self.gen_x_1 = x_1
        return end - start
