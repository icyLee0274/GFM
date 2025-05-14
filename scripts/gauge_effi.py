import os
import time
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
import torch
import numpy as np
import cvxpy as cp
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gfm, gfm

logger = logging.getLogger(__name__)


class Collector:

    def __init__(self):
        self.target = []
        self.n_cons = []
        self.dim = []
        self.seed = []
        self.ip_time = []
        self.gauge_time = []
        self.bs = []

    def collect(self, target, n_cons, dim, seed, batch_size, ip_time, gauge_time):
        self.target.append(target)
        self.n_cons.append(n_cons)
        self.dim.append(dim)
        self.seed.append(seed)
        self.ip_time.append(ip_time)
        self.gauge_time.append(gauge_time)
        self.bs.append(batch_size)

    def save(self, file="gauge_effi.csv"):
        pd.DataFrame({
            "target": self.target,
            "n_cons": self.n_cons,
            "dim": self.dim,
            "seed": self.seed,
            "batch_size": self.bs,
            "ip_time": self.ip_time,
            "gauge_time": self.gauge_time,
        }).to_csv(file, index=False)


@hydra.main(version_base=None, config_path="../configs", config_name="gauge_effi")
def main(cfg: DictConfig):
    collector = Collector()
    logger.debug(os.getcwd())

    for target in cfg.target:
        match target:
            case "linear":
                logger.info(f"Testing linear constraints:\n  {OmegaConf.to_yaml(cfg.linear)}")
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_linear(cfg.linear, collector)
            case "quadratic":
                logger.info(f"Testing quadratic constraints:\n  {OmegaConf.to_yaml(cfg.quadratic)}")
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_qc(cfg.quadratic, collector)
            case "soc":
                logger.info(f"Testing soc constraints:\n  {OmegaConf.to_yaml(cfg.soc)}")
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_soc(cfg.soc, collector)
            case "lmi":
                logger.info(f"Testing lmi constraints:\n  {OmegaConf.to_yaml(cfg.lmi)}")
                for i in range(cfg.repeat):
                    logger.debug(f"Repeat {i}")
                    test_lmi(cfg.lmi, collector)
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


def test_gauge(domain: gfm.ConstrainedSet, bs: int, ip: np.ndarray, d: int) -> float:
    xs = torch.randn((bs, d)).to(torch.float32)
    vs = xs / torch.linalg.norm(xs, dim=-1, keepdim=True)
    os = torch.from_numpy(ip).to(torch.float32).expand(bs, -1)
    start = time.time()
    dist = domain.eval_intersection_v(os, vs)
    end = time.time()
    gauge_time = end - start
    return gauge_time


def test_linear(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        for n_cons in cfg.n_cons:
            G, h, _ = gfm.make_polytope(cfg.seed, d, n_cons, cfg.test_size, (-cfg.box, cfg.box))
            G = np.vstack([G, np.eye(d), -np.eye(d)])
            h = np.concatenate([h, np.full(2 * d, cfg.box)])

            # Solving interior point
            a = cp.Variable()
            x = cp.Variable(d)
            constraints = [G @ x - h <= a, x>=-cfg.box, x<=cfg.box]
            ip_time, ip = solve_ip(a, x, constraints)

            # Evaluating gauge mapping
            domain = gfm.LinearConstraint(torch.from_numpy(G).to(torch.float32), torch.from_numpy(h).to(torch.float32))
            for bs in cfg.batch_size:
                gauge_time = test_gauge(domain, bs, ip, d)

                collector.collect("Linear", n_cons, d, cfg.seed, bs, ip_time, gauge_time)
                logger.info(f"Linear test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


def test_qc(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        for n_cons in cfg.n_cons:
            Q, p, b = gfm.make_qc(d, n_cons, cfg.seed)

            # Solving interior point
            a = cp.Variable()
            x = cp.Variable(d)
            constraints = [cp.QuadForm(x, cp.psd_wrap(Q[i])) + p[i].T @ x - b[i] <= a for i in range(Q.shape[0])]
            constraints.extend([x>=-cfg.box, x<=cfg.box])
            ip_time, ip = solve_ip(a, x, constraints)

            # Evaluating gauge mapping
            Q = torch.from_numpy(Q).to(torch.float32)
            p = torch.from_numpy(p).to(torch.float32)
            domain = gfm.Intersection(*[
                gfm.QuadraticConstraint(Q[i], p[i], b[i].item()) for i in range(Q.shape[0])
            ])
            for bs in cfg.batch_size:
                gauge_time = test_gauge(domain, bs, ip, d)

                collector.collect("Quadratic", n_cons, d, cfg.seed, bs, ip_time, gauge_time)
                logger.info(f"Quadratic test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


def test_soc(cfg: DictConfig, collector: Collector):
    for d in cfg.dims:
        for n_cons in cfg.n_cons:
            A, b, c, s = gfm.make_soc(d, cfg.mat_dim, n_cons, cfg.seed)

            # Solving interior point
            a = cp.Variable()
            x = cp.Variable(d)
            constraints = [cp.norm(A[i] @ x + b[i], 2) - c[i].T @ x - s[i] <= a for i in range(A.shape[0])]
            constraints.extend([x>=-cfg.box, x<=cfg.box])
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

                collector.collect("SOC", n_cons, d, cfg.seed, bs, ip_time, gauge_time)
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
        for n_cons in cfg.n_cons:
            Fss = make_lmi(d, n_cons, cfg.mat_dim, cfg.seed)

            # Solving interior point
            a = cp.Variable()
            x = cp.Variable(d)
            constraints = [
                Fss[k, 0] + sum(x[i] * Fss[k, i + 1] for i in range(d)) << 0
                for k in range(n_cons)
            ]
            constraints.extend([x>=-cfg.box, x<=cfg.box])
            ip_time, ip = solve_ip(a, x, constraints)
            if ip is None: ip = np.zeros(d)

            logger.info(f"Interior time {ip_time}.")

            # ip_time = 0.
            # ip = np.zeros(d)

            # Evaluating gauge mapping
            Fss = torch.from_numpy(Fss).to(torch.float32)
            domain = gfm.Intersection(*[
                gfm.SemiDefiniteConstraint(Fss[i]) for i in range(n_cons)
            ])
            for bs in cfg.batch_size:
                gauge_time = test_gauge(domain, bs, ip, d)

                collector.collect("LMI", n_cons, d, cfg.seed, bs, ip_time, gauge_time)
                logger.info(f"LMI test of dimension {d} finished in {ip_time:.2f} and {gauge_time:.2f} seconds.")


if __name__ == "__main__":
    main()
