import logging
import math
from pickle import FALSE

from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor
import pyomo.environ as pyo

import gfm
import examples
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class StarDomain(gfm.ConstrainedSet):

    def __init__(self, alpha: float, n_tips: int):
        """
        A star-shaped domain defined by the equation:

        .. math::

            |x| \leq 1 + \\alpha * \sin(n_{tips} \cdot \\arctan2(y, x))

        :param alpha: Scaling factor, the smaller, the flatter the tips.
        :param n_tips: Number of tips of the star.
        """
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
        return self.gamma(vs) / torch.linalg.vector_norm(vs, dim=-1)


class Star(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        return StarDomain(self.cfg.example.alpha, self.cfg.example.n_tips), \
            torch.zeros(2, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        dim = 2
        device = self.device

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

    def _reflect_rf(self):
        alpha = float(self.cfg.example.alpha)
        n_tips = float(self.cfg.example.n_tips)
        star_fn = lambda x, y: \
            torch.sqrt(x * x + y * y) - alpha * torch.sin(n_tips * torch.atan2(y, x))

        @torch.enable_grad()
        @torch.inference_mode(False)
        def reflect_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = os + vs
            fs = self.get_domain().check_feasibility_v(xs)
            ifs = fs.logical_not()
            if torch.any(ifs):
                # If not all points are feasible, reflect them
                ts = self.get_domain().eval_intersection_v(
                    os[ifs], vs[ifs],
                    tol=1e-5, thresh=1e6,
                    device=vs.device
                )
                ns = torch.zeros(ts.shape[0], 2, device=vs.device)
                for i, j in zip(torch.argwhere(ifs), range(ts.shape[0])):
                    xy = os[i] + ts[j] * vs[i]
                    x = torch.tensor(xy[0, 0].item(), device="cpu", requires_grad=True)
                    y = torch.tensor(xy[0, 1].item(), device="cpu", requires_grad=True)
                    # z = star_fn(x, y)
                    z = torch.sqrt(x * x + y * y) - alpha * torch.sin(n_tips * torch.atan2(y, x))
                    z.backward()
                    ns[j, 0] = x.grad.item()
                    ns[j, 1] = y.grad.item()
                ns = ns / torch.linalg.vector_norm(ns, dim=-1, keepdim=True)
                # projection onto the normal vector is given by w=v^Tnn/n^Tn, where n^Tn=1 here.
                ws = torch.sum(vs[ifs] * ns, dim=-1, keepdim=True) * ns
                xs[ifs] -= 2 * ws
            return xs

        return reflect_fn

    def _project_rf(self):
        alpha = float(self.cfg.example.alpha)
        n_tips = float(self.cfg.example.n_tips)

        solver = pyo.SolverFactory('ipopt')
        if not solver.available(): raise RuntimeError("Ipopt not available.")

        def solve_projection(xy0: Tensor) -> Tensor:
            x0 = xy0[0].item()
            y0 = xy0[1].item()
            r0 = math.sqrt(x0 * x0 + y0 * y0)

            model = pyo.ConcreteModel(name="Star Convex Projection")

            model.x = pyo.Var(initialize=(0.99 - alpha) * x0 / r0)
            model.y = pyo.Var(initialize=(0.99 - alpha) * y0 / r0)

            model.obj = pyo.Objective(rule=lambda m: (m.x - x0) ** 2 + (m.y - y0) ** 2, sense=pyo.minimize)
            model.constraint = pyo.Constraint(rule=lambda m: (
                    pyo.sqrt(m.x ** 2 + m.y ** 2) <= 1.0 +
                    alpha * pyo.sin(2.0 * n_tips *
                                    pyo.atan(m.y / (m.x + pyo.sqrt(m.x ** 2 + m.y ** 2))))
            ))

            res = solver.solve(model)
            if res.solver.status == pyo.SolverStatus.ok:
                if res.solver.termination_condition == pyo.TerminationCondition.optimal or \
                        res.solver.termination_condition == pyo.TerminationCondition.feasible:
                    logger.info("Projection successful.")
                else:
                    logger.warning("Solver did not find an optimal solution, using original point.")
                opt_x = pyo.value(model.x)
                opt_y = pyo.value(model.y)
            else:
                logger.error("Solver failed to solve the problem, using original point.")
                opt_x = x0
                opt_y = y0
            return torch.tensor([opt_x, opt_y], device=xy0.device, dtype=xy0.dtype)

        def project_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = os + vs
            fs = self.get_domain().check_feasibility_v(xs)
            for i in torch.argwhere(~fs):
                xs[i] = solve_projection(xs[i].flatten())
            return xs

        return project_fn
