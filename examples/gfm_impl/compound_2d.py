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
                cp.QuadForm(x, Q_np) + p_np.T @ x + d.item() <= a,
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
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

    def _reflect_rf(self):
        device = self.device
        A = Tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]).to(device=device, dtype=torch.float32)
        b = Tensor([2, 1.5, 0, 0]).to(device)
        linear = gfm.LinearConstraint(A, b)  # A x <= b
        lr = gfm.PolytopeReflector(A.T, b)

        # | x + (.3, .2) | <= 2.5
        ball = gfm.BallConstraint(Tensor([-.3, -.2]).to(device), 2.5)
        br = gfm.QcReflector(
            torch.eye(2, device=device).expand(1, -1, -1),
            torch.tensor([[.6, .4]], device=device),
            torch.tensor([6.12], device=device),
        )

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
        er = gfm.QcReflector(Q.expand(1, -1, -1), p.expand(1, -1), d.expand(1))

        def reflect_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = os + vs
            fs = self.get_domain().check_feasibility_v(xs, vs.device)
            for i in torch.nonzero(fs):
                # We need to determine which condition is first violated.
                # This is done by evaluating the intersection of the constraints.
                # For violated constraints, the intersection will be less than 1,
                # and the closer to 0, the earlier the constraint is violated.
                il = linear.eval_intersection(os[i].flatten(), vs[i].flatten())
                ib = ball.eval_intersection(os[i].flatten(), vs[i].flatten())
                ie = ellipsoid.eval_intersection(os[i].flatten(), vs[i].flatten())
                if il <= ib and il <= ie:
                    xs[i] = lr(os[i].view(1, -1), vs[i].view(1, -1))
                elif ib <= il and ib <= ie:
                    xs[i] = br(os[i].view(1, -1), vs[i].view(1, -1))
                else:
                    xs[i] = er(os[i].view(1, -1), vs[i].view(1, -1))
            return xs

        return reflect_fn

    def _project_rf(self):
        device = self.device
        A_np = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ])
        b_np = np.array([2, 1.5, 0, 0])

        # | x + (.3, .2) | <= 2.5

        a1 = 2.5
        a2 = 1
        v1 = [2, 1]
        c = Tensor([.5, .45]).to(device)
        D = torch.diag(Tensor([1 / (a1 * a1), 1 / (a2 * a2)]).to(device))
        R = Tensor([[v1[0], -v1[1]], [v1[1], v1[0]]]).to(device) / sqrt(sum(v1))
        Q = R.matmul(D).matmul(R.t())
        p = -2 * c.matmul(Q)
        d = c.matmul(Q).matmul(c) - 1

        c0_np = np.array([.3, .2])
        Q_np = Q.cpu().numpy()
        p_np = p.cpu().numpy()

        x = cp.Variable(2)

        constraints = [
            A_np @ x <= b_np,
            cp.norm(x + c0_np) <= 2.5,
            cp.QuadForm(x, Q_np) + p_np.T @ x + d.item() <= 0.0,
        ]

        domain = self.get_domain()

        def project_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = os + vs
            fs = domain.check_feasibility_v(xs, vs.device)
            for i in torch.nonzero(~fs):
                x0 = xs[i].cpu().numpy()
                problem = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)), constraints)
                problem.solve(solver=cp.MOSEK)
                if problem.status == cp.OPTIMAL:
                    xs[i] = torch.from_numpy(x.value).to(xs)
                    # logger.info("Optimal projection solved.")
                else:
                    xs[i] = os[i]
                    logger.error(f"Optimal projection not found: {problem.status}")
            return xs

        return project_fn
