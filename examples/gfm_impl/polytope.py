from itertools import product

from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor

import gfm
import examples


class Polytope(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _init_domain(self) -> tuple[gfm.ConstrainedSet, Tensor]:
        dim = self.cfg.example.dimension
        device = self.device

        npz = np.load(self.cfg.example.domain_file)
        G_np = np.vstack([npz["G"], np.eye(dim), -np.eye(dim)])
        h_np = np.concatenate([npz["h"], np.full(dim, 4.), np.full(dim, 4.)])
        G = torch.from_numpy(G_np).to(device=device, dtype=torch.float32)
        h = torch.from_numpy(h_np).to(device=device, dtype=torch.float32)
        ip = torch.from_numpy(npz["ip"]).to(device=device, dtype=torch.float32)

        return gfm.LinearConstraint(G, h), ip

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        dim = self.cfg.example.dimension
        device = self.device

        avgs = [
            torch.tensor(p, device=device, dtype=torch.float32)
            for p in product(*([[-3, 3]] * dim))
        ]
        for i in range(dim - 1):
            avgs.append(-2 + 4 * torch.rand(dim, device=device, dtype=torch.float32))
        cov = .4 * torch.eye(dim, device=device)
        dists = []
        for avg in avgs:
            dists.append(torch.distributions.MultivariateNormal(avg, cov))
        data_dist = gfm.TruncatedDistribution(gfm.SumDistribution(*dists), self.get_domain())
        data = data_dist.sample(torch.Size([n]))
        return data

    def _reflect_rf(self):
        domain: gfm.LinearConstraint = self.get_domain()
        return gfm.PolytopeReflector(domain.At, domain.b)

    def _project_rf(self):
        domain: gfm.LinearConstraint = self.get_domain()
        return gfm.PolytopeProjector(domain.At, domain.b)
