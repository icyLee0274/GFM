import torch, time, os
from torch import tensor, Tensor
import cvxpy as cp

from .example import Example
from misc import *
from gfm import *


class Polytope(Example):

    def __init__(self, args, dim: int = 2):
        super().__init__(args, dim)
        self.data = args.data
        self.ip_str = args.interior_point
        self.ip_data = os.path.join(self.output, "interior_point.npy")
        if self.data is None: raise RuntimeError("Domain data must be provided!")
        self.init_domain()
        if self.method == "reflect":
            self.rf = PolytopeReflector(self.domain.At, self.domain.b)
        elif self.method == "project":
            self.rf = PolytopeProjector(self.domain.At, self.domain.b)
        elif self.method == "gauge_reflect":
            self.rf = cube_reflect if self.norm == "Linf" else ball_reflect
        elif self.method == "gauge_project.yaml":
            self.rf = cube_project if self.norm == "Linf" else ball_project
        else:
            self.rf = None

    def init_domain(self):
        npz = np.load(self.data)
        G_np = np.vstack([npz["G"], np.eye(self.dim), -np.eye(self.dim)])
        h_np = np.concatenate([npz["h"], np.full(self.dim, 4.), np.full(self.dim, 4.)])
        G = torch.from_numpy(G_np).to(device=self.device, dtype=torch.float32)
        h = torch.from_numpy(h_np).to(device=self.device, dtype=torch.float32)
        ip = torch.from_numpy(npz["ip"]).to(device=self.device, dtype=torch.float32)

        # if self.ip_str is None:
        #     ## Solve for "best" interior point
        #     a = cp.Variable()
        #     x = cp.Variable(self.dim)
        #     objective = cp.Minimize(a)
        #     constraints = [G_np @ x - h_np <= a]
        #     problem = cp.Problem(objective, constraints)
        #     problem.solve()
        #     if problem.status == cp.OPTIMAL:
        #         ip = torch.from_numpy(x.value).to(device=self.device, dtype=torch.float32)
        #         if self.verbose: print("Solved interior point for gauge map.")
        #         np.save(self.ip_data, ip)
        #     else:
        #         raise RuntimeError("Failed to solve the interior point.")
        # else:
        #     ip = np.load(self.ip_data)
        #     if self.verbose: print(f"Interior point loaded: {ip}.")

        self.domain = LinearConstraint(G, h)
        self.gauge_map = GaugeMap(self.domain, ip, dest="cube" if self.norm == "Linf" else "ball")
        self.init_prior(-1, 2)

        if self.verbose:
            print("Domain initialized.")

    def data_distribution(self):
        from itertools import product
        with torch.no_grad():
            avgs = [
                tensor(p, device=self.device, dtype=torch.float32)
                for p in product(*([[-3, 3]] * self.dim))
            ]
            for i in range(self.dim - 1):
                avgs.append(-2 + 4 * torch.rand(self.dim, device=self.device, dtype=torch.float32))
            cov = .4 * torch.eye(self.dim, device=self.device)
            dists = []
            for avg in avgs:
                dists.append(torch.distributions.MultivariateNormal(avg, cov))
            data_dist = TruncatedDistribution(SumDistribution(*dists), self.domain)
            return data_dist

    def gen0(self) -> float:
        z_0 = self.prior_dist.sample(torch.Size([self.n_gen]))
        ts = torch.linspace(0, 1, self.n_step)
        start = time.time()
        match self.method:
            case "vanilla":
                x_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
            case "reflect":
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
            case "project":
                x_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
            case "gauge_vanilla.yaml":
                z_1 = odeint_reflect(self.velocity, z_0, ts)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_reflect":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
                x_1 = self.gauge_map.from_disk(z_1)
            case "gauge_project.yaml":
                z_1 = odeint_reflect(self.velocity, z_0, ts, reflect_fn=self.rf)[-1]
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
