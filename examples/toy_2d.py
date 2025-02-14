import numpy as np
import torch, time, os
from torch import tensor, Tensor
from math import sqrt
import cvxpy as cp
from torch.distributions import MultivariateNormal

from .example import Example
from misc import *
from CvxIneq import *


class Toy2D(Example):

    def __init__(self, args):
        super().__init__(args, 2)
        self.ip_str = args.interior_point
        self.data = args.data if args.data is not None else os.path.join(self.output, "interior_point.npy")
        self.init_domain()
        if self.method == "gauge_reflect":
            self.rf = cube_reflect if self.norm == "Linf" else ball_reflect
        else:
            self.rf = cube_project if self.norm == "Linf" else ball_project

    def init_domain(self):
        A = Tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]).to(device=self.device, dtype=torch.float32)
        b = Tensor([2, 1.5, 0, 0]).to(self.device)
        linear = LinearConstraint(A, b)  # A x <= b

        # | x + (.3, .2) | <= 2.5
        ball = BallConstraint(Tensor([-.3, -.2]).to(self.device), 2.5)

        a1, a2 = 2.5, 1
        v1 = [2, 1]
        c = Tensor([.5, .45]).to(self.device)
        D = torch.diag(Tensor([1 / (a1 * a1), 1 / (a2 * a2)]).to(self.device))
        R = Tensor([[v1[0], -v1[1]], [v1[1], v1[0]]]).to(self.device) / sqrt(sum(v1))
        Q = R.matmul(D).matmul(R.t())
        p = -2 * c.matmul(Q)
        d = c.matmul(Q).matmul(c) - 1
        ellipsoid = QuadraticConstraint(Q, p, d.item())
        con = Intersection(linear, ball, ellipsoid)
        self.domain = con

        if self.ip_str is None:
            A_np = A.detach().numpy()
            b_np = b.detach().numpy()
            c0_np = np.array([.3, .2])
            Q_np = Q.detach().numpy()
            p_np = p.detach().numpy()

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
                np.save(self.data, ip)
                if self.verbose: print(f"Interior point solved: {ip}. Saved to {self.data}.")
            else:
                raise RuntimeError("Failed to solve the interior point.")
        elif self.ip_str == "load":
            ip = np.load(self.data)
            if self.verbose: print(f"Interior point loaded: {ip}.")
        else:
            ip = np.array([float(x) for x in self.ip_str.split(",")])

        ip = torch.from_numpy(ip).to(device=self.device, dtype=torch.float32)

        self.gauge_map = GaugeMap(self.domain, ip, "cube" if self.norm == "Linf" else "ball")

        self.init_prior(0, 2)

        if self.verbose: print("Domain initialized.")

    def data_distribution(self):
        with torch.no_grad():
            return TruncatedDistribution(
                SumDistribution(
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
                    weights=Tensor([.4, .3, .4])
                ),
                self.domain
            )

    def gen0(self) -> float:
        z_0 = self.prior_dist.sample(torch.Size([self.n_gen]))
        ts = torch.linspace(0, 1, self.n_step)
        start = time.time()
        match self.method:
            case "vanilla":
                x_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
            case "reflect":
                raise NotImplementedError("Reflected method is not implemented.")
            case "project":
                raise NotImplementedError("Projected method is not implemented.")
            case "gauge_vanilla":
                z_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_reflect":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_project":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_mirror":
                y_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
                if self.norm == "L2":
                    z_1 = unit_ball_dual_map(y_1)
                else:
                    z_1 = unit_cube_dual_map(y_1)
                x_1 = self.gauge_map.from_disk(z_1)
            case _:
                raise RuntimeError(f"Unsupported method: {self.method}")
        end = time.time()
        self.gen_x_1 = x_1
        return end - start
