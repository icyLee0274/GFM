import itertools
import os
import time
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
import torch
import numpy as np
import cvxpy as cp
import pandas as pd

import gfm, gfm

logger = logging.getLogger(__name__)


class Collector:

    def __init__(self):
        self.kv = {
            "target": [],
            "n_cons": [],
            "dim": [],
            "seed": [],
            "ip_time": [],
            "gauge_time": [],
            "batch_size": [],
        }

    def collect(self, target, n_cons, dim, seed, batch_size, ip_time, gauge_time, **kwargs):
        self.kv["target"].append(target)
        self.kv["n_cons"].append(n_cons)
        self.kv["dim"].append(dim)
        self.kv["seed"].append(seed)
        self.kv["ip_time"].append(ip_time)
        self.kv["gauge_time"].append(gauge_time)
        self.kv["batch_size"].append(batch_size)
        length = len(self.kv["target"]) - 1
        for k, v in kwargs.items():
            self.pad(k, length)
            self.kv[k].append(v)

    def save(self, file="gauge_effi.csv"):
        length = len(self.kv["target"])
        for k in self.kv.keys():
            self.pad(k, length)
        pd.DataFrame(self.kv).to_csv(file, index=False)

    def pad(self, key, length):
        if key not in self.kv:
            self.kv[key] = []
        while len(self.kv[key]) < length:
            self.kv[key].append(None)


@hydra.main(version_base=None, config_path="../configs", config_name="gauge_effi")
def main(cfg: DictConfig):
    collector = Collector()
    logger.debug(os.getcwd())

    for target in cfg.target:
        match target:
            case "linear":
                logger.info("Testing linear constraints:\n\t%s", OmegaConf.to_container(cfg.linear))
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_linear(cfg.linear, collector)
            case "quadratic":
                logger.info("Testing quadratic constraints:\n\t%s", OmegaConf.to_container(cfg.quadratic))
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_qc(cfg.quadratic, collector)
            case "soc":
                logger.info("Testing soc constraints:\n\t%s", OmegaConf.to_container(cfg.soc))
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_soc(cfg.soc, collector)
            case "lmi":
                logger.info("Testing lmi constraints:\n\t%s", OmegaConf.to_container(cfg.lmi))
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_lmi(cfg.lmi, collector)
            case "poly":
                logger.info("Testing polytope constraints:\n\t%s", OmegaConf.to_container(cfg.poly))
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_poly(cfg.poly, collector)
            case _:
                raise NotImplementedError(f"Target '{cfg.target}' not implemented.")

    fn = f"{cfg.target}_effi.csv"
    collector.save(fn)
    logger.debug(f"Results saved to {fn}")


def solve_ip(a, x, constraints):
    objective = cp.Minimize(a)
    problem = cp.Problem(objective, constraints)
    start = time.time()
    problem.solve()
    end = time.time()
    return end - start, x.value


def test_gauge(domain: gfm.ConstrainedSet, bs: int, ip: np.ndarray, d: int,
               thresh: float = 1e8, tol: float = 1e-6) -> float:
    vs = torch.randn((bs, d)).to(torch.float32)
    os = torch.from_numpy(ip).to(torch.float32).expand(bs, -1)
    start = time.time()
    domain.eval_intersection_v(os, vs, thresh=thresh, tol=tol)
    end = time.time()
    gauge_time = end - start
    return gauge_time


def test_linear(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        G, h, _ = gfm.make_polytope(cfg.seed, d, cfg.n_cons, cfg.test_size, (-cfg.box, cfg.box))
        G = np.vstack([G, np.eye(d), -np.eye(d)])
        h = np.concatenate([h, np.full(2 * d, cfg.box)])

        # Solving interior point
        a = cp.Variable()
        x = cp.Variable(d)
        constraints = [G @ x - h <= a]
        ip_time, ip = solve_ip(a, x, constraints)

        # Evaluating gauge mapping
        domain = gfm.LinearConstraint(torch.from_numpy(G).to(torch.float32), torch.from_numpy(h).to(torch.float32))
        for bs in cfg.batch_size:
            gauge_time = test_gauge(domain, bs, ip, d)

            collector.collect("Linear", cfg.n_cons, d, cfg.seed, bs, ip_time, gauge_time)
            logger.info(f"Linear test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


def test_qc(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        Q, p, b = gfm.make_qc(d, cfg.n_cons, cfg.seed)

        # Solving interior point
        a = cp.Variable()
        x = cp.Variable(d)
        constraints = [cp.QuadForm(x, cp.psd_wrap(Q[i])) + p[i].T @ x - b[i] <= a for i in range(Q.shape[0])]
        ip_time, ip = solve_ip(a, x, constraints)

        # Evaluating gauge mapping
        Q = torch.from_numpy(Q).to(torch.float32)
        p = torch.from_numpy(p).to(torch.float32)
        domain = gfm.Intersection(*[
            gfm.QuadraticConstraint(Q[i], p[i], b[i].item()) for i in range(Q.shape[0])
        ])
        for bs in cfg.batch_size:
            gauge_time = test_gauge(domain, bs, ip, d)

            collector.collect("Quadratic", cfg.n_cons, d, cfg.seed, bs, ip_time, gauge_time)
            logger.info(f"Quadratic test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


def test_soc(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        A, b, c, s = gfm.make_soc(d, cfg.n_cons, cfg.seed)

        # Solving interior point
        a = cp.Variable()
        x = cp.Variable(d)
        constraints = [cp.norm(A[i] @ x + b[i], 2) - c[i].T @ x - s[i] <= a for i in range(A.shape[0])]
        ip_time, ip = solve_ip(a, x, constraints)

        # Evaluating gauge mapping
        A = torch.from_numpy(A).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        c = torch.from_numpy(c).to(torch.float32)

        domain = gfm.Intersection(*[
            gfm.ConeConstraint(A[i], b[i], c[i], s[i]) for i in range(A.shape[0])
        ])
        for bs in cfg.batch_size:
            gauge_time = test_gauge(domain, bs, ip, d)

            collector.collect("SOC", cfg.n_cons, d, cfg.seed, bs, ip_time, gauge_time)
            logger.info(f"SOC test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


def test_lmi(cfg: DictConfig, collector: Collector):
    def make_lmi(num_var, num_ineq, num_mat_dim, seed):
        rng = np.random.default_rng(seed)
        Fss = rng.normal(size=(num_ineq, num_var + 1, num_mat_dim, num_mat_dim)).astype(np.float32)
        FssT = np.transpose(Fss, (0, 1, 3, 2))
        res = Fss + FssT
        for i in range(num_ineq):
            res[i, 0, :, :] = Fss[i, 0, :, :] @ FssT[i, 0, :, :]
        return res

    for d in cfg.dims:
        Fss = make_lmi(d, cfg.n_cons, cfg.mat_dim, cfg.seed)

        # Solving interior point
        a = cp.Variable()
        x = cp.Variable(d)
        constraints = [
            Fss[k, 0] + sum(x[i] * Fss[k, i + 1] for i in range(d)) << 0
            for k in range(cfg.n_cons)
        ]
        ip_time, ip = solve_ip(a, x, constraints)
        if ip is None: ip = np.zeros(d)

        logger.info(f"Interior time {ip_time}.")

        # ip_time = 0.
        # ip = np.zeros(d)

        # Evaluating gauge mapping
        Fss = torch.from_numpy(Fss).to(torch.float32)
        domain = gfm.Intersection(*[
            gfm.SemiDefiniteConstraint(Fss[i]) for i in range(cfg.n_cons)
        ])
        for bs in cfg.batch_size:
            gauge_time = test_gauge(domain, bs, ip, d)

            collector.collect("LMI", cfg.n_cons, d, cfg.seed, bs, ip_time, gauge_time)
            logger.info(f"LMI test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


import sympy as sp


def test_poly(cfg: DictConfig, collector: Collector):
    rng = torch.Generator().manual_seed(cfg.seed)

    for dim, degree, bs in itertools.product(cfg.dims, cfg.degrees, cfg.batch_size):
        sym = sp.symbols(f"x:{dim}")
        n_mono = len(gfm.generate_monomials(sym, degree, constant=True))

        Qs = torch.zeros(cfg.n_cons, n_mono, n_mono, dtype=torch.float32)
        rhs = torch.rand(cfg.n_cons, dtype=torch.float32, generator=rng)

        for i in range(cfg.n_cons):
            ev = torch.rand(n_mono, dtype=torch.float32, generator=rng) * (cfg.eig_max - cfg.eig_min) + cfg.eig_min
            D = torch.diag(ev)

            Z = torch.randn(n_mono, n_mono, dtype=torch.float32, generator=rng)
            Q, _ = torch.linalg.qr(Z, mode="complete")

            Qs[i] = Q @ D @ Q.T
        Qs[:, -1, -1] = 0.0  # ensure that \vec{0} is an interior point

        domain = gfm.SosPolynomialConstraints(Qs, degree, rhs)

        ip = np.zeros(dim, dtype=np.float32)
        ip_time = 0.0

        gauge_time = test_gauge(domain, bs, ip, dim, thresh=cfg.thresh, tol=cfg.tol)

        collector.collect(
            "Polynomial", cfg.n_cons, dim, cfg.seed, bs,
            ip_time, gauge_time,
            degree=degree,
        )
        logger.info("Polynomial test of dimension %d and degree %d finished in %.2f seconds.",
                    dim, degree, gauge_time)


if __name__ == "__main__":
    main()
