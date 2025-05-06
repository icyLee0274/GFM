from itertools import product

from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor

import gfm
import examples
from gfm import ConstrainedSet


class Quadratic(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    @torch.no_grad()
    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        dim = self.cfg.example.dimension
        device = self.cfg.accelerator

        npz = np.load(self.cfg.example.domain_file)
        Hs = torch.from_numpy(npz["Hs"]).to(device=device, dtype=torch.float32)
        Gs = torch.from_numpy(npz["Gs"]).to(device=device, dtype=torch.float32)
        hs = torch.from_numpy(npz["fs"]).to(device=device, dtype=torch.float32)
        ip = torch.from_numpy(npz["ip"]).to(device=device, dtype=torch.float32)

        self.Hs = Hs
        self.Gs = Gs
        self.hs = -hs

        domain = gfm.Intersection(*[
            gfm.QuadraticConstraint(Hs[i], Gs[i], -hs[i].item()) for i in range(Hs.shape[0])
        ])
        return domain, ip

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        dim = self.cfg.example.dimension
        device = self.cfg.accelerator
        avgs = [
            tensor(p, device=device, dtype=torch.float32)
            for p in product(*([[-1, 1]] * dim))
        ]
        for i in range(self.dim - 1):
            avgs.append(-.5 + 1 * torch.rand(dim, device=device, dtype=torch.float32))
        cov = .06 * torch.eye(dim, device=device)
        dists = []
        for avg in avgs:
            dists.append(torch.distributions.MultivariateNormal(avg, cov))
        data_dist = gfm.TruncatedDistribution(gfm.SumDistribution(*dists), self.get_domain())
        data = data_dist.sample([n])
        return data

    def _reflect_rf(self):
        return gfm.QcReflector(self.Hs, self.Gs, self.hs)

    def _project_rf(self):
        return gfm.EllipsoidProjector(self.Hs, self.Gs, self.hs)
