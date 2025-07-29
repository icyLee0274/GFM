import logging
import os
from itertools import product
from math import sqrt

from cvxpy import NonPos
from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
import cvxpy as cp
import scipy.interpolate as interpolate
import geoopt

import gfm
import examples
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class CatDist(torch.distributions.Distribution):

    def __init__(self, *dist: torch.distributions.Distribution):
        super().__init__()
        self.dist = dist

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        ss = []
        for d in self.dist:
            ss.append(d.rsample(sample_shape))
        return torch.cat(ss, dim=-1)


class Robotic(examples.GfmExampleBase):

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        domain = gfm.Intersection(
            gfm.SemiDefiniteConstraint(tensor([
                [[0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.]],
                [[1., 0.], [0., 0.]],
                [[0., 1.], [1., 0.]],
                [[0., 0.], [0., 1.]],
            ], dtype=torch.float32, device=self.device)),
            gfm.LinearConstraint(
                tensor([
                    [0., 0., 1., 0., 1.],
                    [1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [-1., 0., 0., 0., 0.],
                    [0., -1., 0., 0., 0.],
                ],
                    dtype=torch.float32, device=self.device),
                # tensor([5.99, 10., 10., 10., 10.], dtype=torch.float32, device=self.device),
                tensor([5.11, 7.71, 9.10, 6.82, 8.76], dtype=torch.float32, device=self.device),
            )
        )
        ip = torch.tensor([0., 0., 2., .3, 1.5], dtype=torch.float32, device=self.device)
        return domain, ip

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:
        spd = torch.load(self.cfg.example.spd_file, map_location=self.device, weights_only=True)
        poss = torch.load(self.cfg.example.pos_file, map_location=self.device, weights_only=True)

        poss_np = poss.cpu().numpy()

        xy = torch.empty([15, 10000, 2], dtype=torch.float32, device=self.device)
        ts = np.linspace(0, 1, 200)
        ss = np.linspace(0, 1, 10000)
        for i in range(15):
            interp = interpolate.make_interp_spline(ts, poss_np[:, [2 * i, 2 * i + 1]])
            xy[i] = torch.from_numpy(interp(ss)).to(xy)
        xy = xy.view(150000, -1)

        data = torch.cat([xy, spd], dim=1).to(self.device, dtype=torch.float32)

        return data

    # def get_prior(self) -> torch.distributions.Distribution:
    #     dist = getattr(self, "_prior", None)
    #     dim = 3
    #     scale = self.cfg.method.scale
    #     if dist is None:
    #         if self.cfg.method.transform == "L2":
    #             spd_dist = gfm.HyperBallUniform(dim, loc=torch.zeros(dim, device=self.device), scale=scale)
    #             pos_dist = MultivariateNormal(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
    #             dist = CatDist(spd_dist, pos_dist)
    #         elif self.cfg.method.transform == "L_inf":
    #             spd_dist = gfm.box_uniform(torch.zeros(dim), torch.full([dim], scale))
    #             pos_dist = MultivariateNormal(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
    #             dist = CatDist(spd_dist, pos_dist)
    #         elif self.cfg.method.name == "vanilla" or self.cfg.method.name == "ddpm":
    #             dist = MultivariateNormal(torch.zeros(5, device=self.device), torch.eye(5, device=self.device))
    #         else:
    #             raise NotImplementedError(f"Prior distribution not implemented for this method: {self.cfg.method.name}")
    #         setattr(self, "_prior", dist)
    #     return dist

    @torch.no_grad()
    def test_step(self, *args, **kwargs):
        x_1 = self.sample(self.cfg.test.n_gen, self.cfg.test.n_steps)
        data = self.get_data()[:80000, :]  # use less data to avoid oom
        mmd = gfm.maximum_mean_discrepancy(x_1, data)
        fea = self.get_domain().check_feasibility_v(x_1).sum() * 1.0
        # self.log("kl", kl)
        self.log("mmd", mmd)
        self.log("feasible", fea)

        if self.cfg.test.get("save_gen", False):
            i = 0
            while os.path.exists(f"gen_samples_{i}.pt"):
                i += 1
            f_name = os.path.abspath(f"gen_samples_{i}.pt")
            torch.save(x_1, f_name)
            logger.info(f"Saved gen samples to {f_name}")

        return {
            "loss": 0,
            # "kl": kl,
            "mmd": mmd,
            "feasible": fea,
        }
