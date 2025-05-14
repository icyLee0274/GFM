import logging
from itertools import product
from math import sqrt
import os

from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
import cvxpy as cp

import gfm
import examples
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class Compound2D(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    @torch.no_grad()
    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        device = self.device
        A = Tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]).to(device=device, dtype=torch.float32)
        b = Tensor([2, 1.5, 0, 0]).to(device)
        linear = gfm.LinearConstraint(A, b)  # A x <= b

        # | x + (.3, .2) | <= 2.5
        ball = gfm.BallConstraint(Tensor([-.3, -.2]).to(device), 2.5)

        a1 = 2.5
        a2 = 1
        v1 = [2, 1]
        c = Tensor([.5, .45]).to(device)
        D = torch.diag(Tensor([1 / (a1 * a1), 1 / (a2 * a2)]).to(device))
        R = Tensor([[v1[0], -v1[1]], [v1[1], v1[0]]]).to(device) / sqrt(sum(v1))
        Q = R.matmul(D).matmul(R.t())
        p = -2 * c.matmul(Q)
        d = c.matmul(Q).matmul(c) - 1
        ellipsoid = gfm.QuadraticConstraint(Q, p, d.item())
        domain = gfm.Intersection(linear, ball, ellipsoid)

        if self.cfg.example.interior_point is None:
            A_np = A.cpu().numpy()
            b_np = b.cpu().numpy()
            c0_np = np.array([.3, .2])
            Q_np = Q.cpu().numpy()
            p_np = p.cpu().numpy()

            a = cp.Variable()
            x = cp.Variable(2)
            objective = cp.Minimize(a)
            constraints = [
                A_np @ x - b_np <= a,
                cp.norm(x + c0_np) - 2.5 <= a,
                cp.QuadForm(x, Q_np) + p_np.T @ x - d.item() <= a,
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if problem.status == cp.OPTIMAL:
                ip = x.value
                logger.info(f"Solved interior point: {ip}")
                np.save("interior_point.npy", ip)
                logger.info(f"Interior point saved to {os.path.abspath('interior_point.npy')}")
            else:
                raise RuntimeError("Failed to solve the interior point.")
        elif self.cfg.example.interior_point is str:
            ip = np.load(self.cfg.example.interior_point)
            logger.info(f"Interior point loaded: {ip}")
        else:
            ip = np.array(self.cfg.example.interior_point)
        ip = torch.from_numpy(ip).to(device=device, dtype=torch.float32)

        return domain, ip

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        data_dist = gfm.TruncatedDistribution(
            gfm.SumDistribution(
                MultivariateNormal(
                    torch.zeros(2).to(self.device),
                    .08 * Tensor([[4, 0], [0, 1]]).to(self.device),
                ),
                MultivariateNormal(
                    Tensor([1, 1.2]).to(self.device),
                    .15 * Tensor([[1, 2], [2, 6]]).to(self.device),
                ),
                MultivariateNormal(
                    Tensor([2, .6]).to(self.device),
                    .17 * Tensor([[.4, -.1], [-.1, .7]]).to(self.device),
                ),
                weights=Tensor([.4, .3, .4]),
                device=self.device,
            ),
            self.get_domain(),
            device=self.device,
        )
        return data_dist.sample([n]).to(self.device)  # shape: (n, 2)
