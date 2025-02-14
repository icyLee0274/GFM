import torch, time, os
from torch import tensor, Tensor
import cvxpy as cp

from CvxIneq.projections import EllipsoidProjector
from .example import Example
from misc import *
from CvxIneq import *


class QC(Example):

    def __init__(self, args, dim: int = 2):
        super().__init__(args, dim)
        self.hs = None
        self.Gs = None
        self.Hs = None
        self.data = args.data
        if self.data is None: raise RuntimeError("Domain data must be provided!")
        self.init_domain()
        if self.method == "reflect":
            self.rf = QcReflector(self.Hs, self.Gs, self.hs)
        elif self.method == "project":
            self.rf = EllipsoidProjector(self.Hs, self.Gs, self.hs)
        elif self.method == "gauge_reflect":
            self.rf = cube_reflect if self.norm == "Linf" else ball_reflect
        elif self.method == "gauge_project":
            self.rf = cube_project if self.norm == "Linf" else ball_project
        else:
            self.rf = None

    def init_domain(self):
        npz = np.load(self.data)
        Hs_np = npz["Hs"]
        Gs_np = npz["Gs"]
        hs_np = npz["fs"]

        Hs = torch.from_numpy(Hs_np).to(device=self.device, dtype=torch.float32)
        Gs = torch.from_numpy(Gs_np).to(device=self.device, dtype=torch.float32)
        hs = torch.from_numpy(hs_np).to(device=self.device, dtype=torch.float32)
        ip = torch.from_numpy(npz["ip"]).to(device=self.device, dtype=torch.float32)

        self.Hs = Hs
        self.Gs = Gs
        self.hs = -hs

        self.domain = Intersection(*[
            QuadraticConstraint(Hs[i], Gs[i], -hs[i].item()) for i in range(Hs.shape[0])
        ])
        self.gauge_map = GaugeMap(self.domain, ip, dest="cube" if self.norm == "Linf" else "ball")
        self.init_prior(-1, 2)

        if self.verbose:
            print("Domain initialized.")

    def data_distribution(self):
        from itertools import product
        with torch.no_grad():
            avgs = [
                tensor(p, device=self.device, dtype=torch.float32)
                for p in product(*([[-1, 1]] * self.dim))
            ]
            for i in range(self.dim - 1):
                avgs.append(-.5 + 1 * torch.rand(self.dim, device=self.device, dtype=torch.float32))
            cov = .06 * torch.eye(self.dim, device=self.device)
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
                z_1 = unit_ball_dual_map(y_1)
                x_1 = self.gauge_map.from_disk(z_1)
            case _:
                raise RuntimeError(f"Unsupported method: {self.method}")
        end = time.time()
        self.gen_x_1 = x_1
        return end - start
