import logging
from itertools import product
from math import sqrt
import os
import time

import numpy.random
from cvxpy import NonPos
from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
import cvxpy as cp

import gfm
import examples
from examples.combinatorial_sampling import s_vec, s_vec_inv
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


def generate_domain(cfg: DictConfig) -> dict[str, np.ndarray]:
    dim = cfg.example.mat_dim
    A = numpy.random.randn(dim, dim)
    b = numpy.array(cfg.example.beta)
    B = numpy.eye(dim)
    c = numpy.array(cfg.example.charlie)
    C = numpy.eye(dim)

    bound = -cfg.example.bound
    Ds = numpy.zeros([cfg.example.dimension * 2, dim, dim])
    k = 0
    for i in range(dim):
        for j in range(i, dim):
            Ds[k, i, j] = -1
            Ds[k, j, i] = -1
            k += 1
    for i in range(dim):
        for j in range(i, dim):
            Ds[k, i, j] = 1
            Ds[k, j, i] = 1
            k += 1
    cs = numpy.full(Ds.shape[0], bound)

    f_z = os.path.abspath(cfg.example.domain_file)
    numpy.savez(f_z, A=A, b=b, B=B, c=c, C=C, Ds=Ds, cs=cs)
    logger.info(f"Domain file saved to {f_z}.")

    return {
        "A": A,
        "b": b,
        "B": B,
        "c": c,
        "C": C,
        "Ds": Ds,
        "cs": cs,
    }


class Combinatorial(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        domain_file = os.path.abspath(self.cfg.example.domain_file)

        if os.path.exists(domain_file):
            npz = np.load(self.cfg.example.domain_file)
            logger.info(f"Domain file loaded from {domain_file}.")
        else:
            npz = generate_domain(self.cfg)

        Ds = torch.from_numpy(npz["Ds"]).to(self.device)
        cs = torch.from_numpy(npz["cs"]).to(self.device)

        dim = self.cfg.example.mat_dim

        Fs = torch.zeros(self.cfg.example.dimension + 1, dim, dim, device=self.device)
        k = 1
        for i in range(0, dim):
            for j in range(i, dim):
                Fs[k, i, j] = 1
                Fs[k, j, i] = 1
                k += 1
        psd = gfm.SemiDefiniteConstraint(Fs)

        # indices = torch.triu_indices(dim, dim, 0)
        _Ds = torch.zeros(Ds.shape[0], self.cfg.example.dimension, device=self.device)
        for i in range(Ds.shape[0]): _Ds[i] = s_vec(-2 * Ds[i] + torch.diag(Ds[i].diagonal()))
        linear = gfm.LinearConstraint(_Ds, -cs)

        domain = gfm.Intersection(psd, linear)
        ip = torch.eye(self.cfg.example.mat_dim, device=self.device)
        return domain, s_vec(ip)

    @torch.no_grad()
    def _init_data(self, n: int) -> Tensor:

        domain_file = os.path.abspath(self.cfg.example.domain_file)

        if os.path.exists(domain_file):
            npz = np.load(self.cfg.example.domain_file)
        else:
            npz = generate_domain(self.cfg)

        A = torch.from_numpy(npz["A"]).to(self.device)
        b = torch.from_numpy(npz["b"]).to(self.device).item()
        B = torch.from_numpy(npz["B"]).to(self.device)
        c = torch.from_numpy(npz["c"]).to(self.device).item()
        C = torch.from_numpy(npz["C"]).to(self.device)
        Ds = torch.from_numpy(npz["Ds"]).to(self.device)
        cs = torch.from_numpy(npz["cs"]).to(self.device)

        sampler = examples.CombinatorialSampler(A, b, B, c, C, Ds, cs)

        logger.info("Starting sampling...")
        start = time.time()
        _data = sampler.sample(
            torch.eye(self.cfg.example.mat_dim, device=self.device),
            n,
            burn_in=self.cfg.example.burn_in,
            thin=self.cfg.example.thin,
            metro_scale=self.cfg.example.metro_scale,
            metro_burn=self.cfg.example.metro_burn,
        )
        sample_time = time.time() - start

        logger.info(f"Hit-and-run sampling time: {sample_time} seconds.")

        data = torch.zeros(n, self.cfg.example.dimension, device=self.device)
        for i in range(n):
            data[i] = s_vec(_data[i])

        return data
