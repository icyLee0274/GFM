import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal

from torchdiffeq import odeint

from torchdiffeqReflect import odeint as odeint_reflect

from math import sqrt
import os
import time

from misc import *
from CvxIneq import (
    unit_ball_mirror_map as mirror_map,
    unit_ball_dual_map as dual_map,
    ball_reflect,
    ball_reflect_velocity,
    ball_projection, GaugeMap
)

from argparse import ArgumentParser


class ToyExample:

    def __init__(self, device, n_sample, n_epoch, batch_size, n_gen, n_step, hidden, output):
        self.device = device
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_gen = n_gen
        self.n_step = n_step
        self.hidden = hidden
        self.dir = output
        self.constraint = None
        self.data_distribution = None
        self.samples = None
        self.x0 = None
        self.gauge_map = None
        self.gauge_samples = None
        self.reflect_prior = None
        self.reflect_model = None
        self.reflect_gauge_gen = None
        self.reflect_gen = None
        self.project_gauge_gen = None
        self.project_gen = None
        self.vanilla_gauge_gen = None
        self.vanilla_gen = None
        self.mirror_samples = None
        self.mirror_model = None
        self.mirror_dual_gen = None
        self.mirror_gauge_gen = None
        self.mirror_gen = None

    def toy_constraints(self) -> ConstrainedSet:
        A = Tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]).to(self.device)
        b = Tensor([2, 1.5, 0, 0]).to(self.device)
        linear = LinearConstraint(A, b)
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
        self.constraint = con
        return con

    def constrained_set_boundary(self, n: int) -> Tensor:
        if self.constraint is None or self.x0 is None:
            raise ValueError("constraint and x0 must be defined")
        vs = torch.vstack([
            torch.cos(torch.linspace(0, 2 * torch.pi, n)),
            torch.sin(torch.linspace(0, 2 * torch.pi, n)),
        ]).t()
        ts = self.constraint.eval_intersection_v(self.x0.expand(n, -1), vs)
        return vs * ts.view(-1, 1) + self.x0.expand(n, -1)

    def init_data_distribution(self) -> torch.distributions.Distribution:
        if self.constraint is None:
            raise ValueError('constraint must be defined')
        dd = TruncatedDistribution(
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
            self.constraint
        )
        self.data_distribution = dd
        return dd

    def init_samples(self) -> Tensor:
        self.samples = self.data_distribution.sample((self.n_sample,))
        return self.samples

    def save_samples(self) -> None:
        if self.samples is None:
            raise ValueError('samples must be defined')
        torch.save(self.samples, os.path.join(self.dir, f"samples{self.n_sample}.pt"))

    def set_gauge_interior_point(self, p: Tensor) -> None:
        self.x0 = p

    def init_gauge_map(self) -> None:
        if self.constraint is None or self.x0 is None:
            raise ValueError('constraint and x0 must be defined')
        self.gauge_map = GaugeMap(self.constraint, self.x0)

    def init_gauge_samples(self) -> Tensor:
        if self.samples is None or self.gauge_map is None:
            raise ValueError('samples and gauge_map must be defined')
        self.gauge_samples = self.gauge_map.to_disk(self.samples)
        return self.gauge_samples

    def save_gauge_samples(self) -> None:
        if self.gauge_samples is None:
            raise ValueError('gauge_samples must be defined')
        torch.save(self.gauge_samples, os.path.join(self.dir, f"gauge_samples{self.n_sample}.pt"))

    def init_mirror_samples(self) -> Tensor:
        if self.gauge_samples is None:
            raise ValueError('gauge_samples must be defined')
        self.mirror_samples = mirror_map(self.gauge_samples)
        return self.mirror_samples

    def save_mirror_samples(self) -> None:
        if self.mirror_samples is None:
            raise ValueError('mirror_samples must be defined')
        torch.save(self.mirror_samples, os.path.join(self.dir, f"mirror_samples{self.n_sample}.pt"))

    def train_reflecting_model(self) -> None:
        if self.gauge_samples is None:
            raise ValueError('gauge_samples must be defined')
        reflect_velocity = FlowVelocity(h=self.hidden).to(self.device)
        reflect_opt = torch.optim.Adam(reflect_velocity.parameters(), lr=1e-3)
        reflect_loss = nn.MSELoss()
        reflect_dl = DataLoader(TensorDataset(self.gauge_samples), batch_size=self.batch_size, shuffle=True)
        reflect_prior = TruncatedDistribution(
            HyperBoxUniform(Tensor([-1, -1]), Tensor([2, 2])),
            BallConstraint(torch.zeros(2))
        )
        for _ in range(self.n_epoch):
            x_1 = next(iter(reflect_dl))[0]
            x_0 = reflect_prior.sample(torch.Size((len(x_1),)))
            t = torch.rand(len(x_1), 1)
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0
            reflect_opt.zero_grad()
            reflect_loss(reflect_velocity(t, x_t), dx_t).backward()
            reflect_opt.step()
        self.reflect_model = reflect_velocity
        self.reflect_prior = reflect_prior

    def save_reflecting_model(self) -> None:
        if self.reflect_model is None:
            raise ValueError('reflect_model must be defined')
        torch.save(self.reflect_model, os.path.join(self.dir, f"reflect_model.pt"))

    def gen_vanilla(self) -> float:
        if self.reflect_model is None:
            raise ValueError('reflect_model must be defined')
        z_0 = self.reflect_prior.sample(torch.Size((self.n_gen,)))
        start = time.time()
        z_t = odeint(self.reflect_model, z_0, torch.linspace(0, 1, self.n_step))
        z_1 = z_t[-1]
        x_1 = self.gauge_map.from_disk(z_1)
        end = time.time()
        torch.save(z_t, os.path.join(self.dir, f"vanilla_gauge_gen{self.n_gen}.pt"))
        torch.save(x_1, os.path.join(self.dir, f"vanilla_gen{self.n_step}.pt"))
        self.vanilla_gauge_gen = z_t
        self.vanilla_gen = x_1
        return end - start

    def gen_reflecting(self) -> float:
        if self.reflect_model is None:
            raise ValueError('reflect_model must be defined')
        z_0 = self.reflect_prior.sample(torch.Size((self.n_gen,)))
        start = time.time()
        z_t = odeint_reflect(
            self.reflect_model,
            z_0, torch.linspace(0, 1, self.n_step),
            reflect_fn=ball_reflect,
            reflect_velocity_fn=ball_reflect_velocity,
        )
        z_1 = z_t[-1]
        x_1 = self.gauge_map.from_disk(z_1)
        end = time.time()
        torch.save(z_t, os.path.join(self.dir, f"reflect_gauge_gen{self.n_gen}.pt"))
        torch.save(x_1, os.path.join(self.dir, f"reflect_gen{self.n_gen}.pt"))
        self.reflect_gauge_gen = z_t
        self.reflect_gen = x_1
        return end - start

    def gen_projecting(self) -> float:
        if self.reflect_model is None:
            raise ValueError('reflect_model must be defined')
        z_0 = self.reflect_prior.sample(torch.Size((self.n_gen,)))
        start = time.time()
        z_t = odeint_reflect(
            self.reflect_model,
            z_0, torch.linspace(0, 1, self.n_step),
            reflect_fn=ball_projection,
            reflect_velocity_fn=ball_reflect_velocity,
        )
        z_1 = z_t[-1]
        x_1 = self.gauge_map.from_disk(z_1)
        end = time.time()
        torch.save(z_t, os.path.join(self.dir, f"project_gauge_gen{self.n_gen}.pt"))
        torch.save(x_1, os.path.join(self.dir, f"project_gen{self.n_gen}.pt"))
        self.project_gauge_gen = z_t
        self.project_gen = x_1
        return end - start

    def train_mirror_model(self) -> None:
        if self.mirror_samples is None:
            raise ValueError('mirror_samples must be defined')
        mirror_velocity = FlowVelocity().to(self.device)
        mirror_opt = torch.optim.Adam(mirror_velocity.parameters(), lr=1e-3)
        mirror_loss = nn.MSELoss()
        mirror_dl = DataLoader(TensorDataset(self.mirror_samples), batch_size=self.batch_size, shuffle=True)
        for _ in range(self.n_epoch):
            y_1 = next(iter(mirror_dl))[0]
            y_0 = torch.randn_like(y_1)
            t = torch.rand(len(y_1), 1)
            y_t = (1 - t) * y_0 + t * y_1
            dy_t = y_1 - y_0
            mirror_opt.zero_grad()
            mirror_loss(mirror_velocity(t, y_t), dy_t).backward()
            mirror_opt.step()
        self.mirror_model = mirror_velocity

    def save_mirror_model(self) -> None:
        if self.mirror_model is None:
            raise ValueError('mirror_model must be defined')
        torch.save(self.mirror_model, os.path.join(self.dir, f"mirror_model.pt"))

    def gen_mirror(self) -> float:
        if self.mirror_model is None:
            raise ValueError('mirror_model must be defined')
        y_0 = torch.randn(self.n_gen, 2)
        start = time.time()
        y_t = odeint(self.mirror_model, y_0, torch.linspace(0, 1, self.n_step))
        z_1 = dual_map(y_t[-1])
        x_1 = self.gauge_map.from_disk(z_1)
        end = time.time()
        torch.save(y_t, os.path.join(self.dir, f"mirror_dual_gen{self.n_gen}.pt"))
        torch.save(z_1, os.path.join(self.dir, f"mirror_gauge_gen{self.n_gen}.pt"))
        torch.save(x_1, os.path.join(self.dir, f"mirror_gen{self.n_gen}.pt"))
        self.mirror_dual_gen = y_t
        self.mirror_gauge_gen = z_1
        self.mirror_gen = x_1
        return end - start


def main():
    #
    parser = ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--n_epoch", type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--n_gen", type=int, default=1000)
    parser.add_argument("--n_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", type=str, default="./toy-example-out")
    parser.add_argument("--repeat", type=int, default=1)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")
        torch.set_default_device(device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    example = ToyExample(
        device,
        args.n_sample,
        args.n_epoch,
        args.batch_size,
        args.n_gen,
        args.n_step,
        args.hidden,
        args.output,
    )

    run_example(example)

    plot_example(example)

    eval_kl_div(example)

    if args.repeat > 1:
        test_time_kl(example, args.repeat)


def run_example(ex: ToyExample) -> None:
    print("Initializing...")
    ex.toy_constraints()
    ex.init_data_distribution()

    print("Generating samples...")
    ex.init_samples()
    print("Samples generated.")
    ex.save_samples()
    print("Samples saved.")

    print("Set Gauge map interior point to [1.0, 0.5]")
    ex.set_gauge_interior_point(Tensor([1., .5]).to(ex.device))
    ex.init_gauge_map()
    print("Evaluating Gauge map...")
    ex.init_gauge_samples()
    print("Gauge map evaluated.")
    ex.save_gauge_samples()
    print("Gauge mapped samples saved.")

    print("Training reflecting model...")
    ex.train_reflecting_model()
    print("Reflecting model trained.")
    ex.save_reflecting_model()
    print("Reflecting model saved.")
    t = ex.gen_reflecting()
    print(f"Reflecting model generation done. ({t:.3f}s)")
    t = ex.gen_projecting()
    print(f"Projecting model generation done. ({t:.3f}s)")
    t = ex.gen_vanilla()
    print(f"Vanilla model generation done. ({t:.3f}s)")

    print("Evaluating mirror map...")
    ex.init_mirror_samples()
    print("Mirror model evaluated.")
    ex.save_mirror_samples()
    print("Mirrored samples saved.")
    print("Training mirror model...")
    ex.train_mirror_model()
    print("Mirror model trained.")
    ex.save_mirror_model()
    print("Mirror model saved.")
    t = ex.gen_mirror()
    print(f"Mirror model generation done. ({t:.3f}s)")


def plot_example(ex: ToyExample) -> None:
    import matplotlib.pyplot as plt

    def _plot(ax, xy: Tensor, bdr: Tensor, feasible: Tensor):
        from scipy.stats import gaussian_kde
        with torch.no_grad():
            kde = gaussian_kde(xy.numpy().T)
            ps = xy[feasible, :].numpy()
            qs = xy[~feasible, :].numpy()
            bs = bdr.numpy()
            ds = kde(ps.T)
            ax.scatter(ps[:, 0], ps[:, 1], c=ds, cmap="cividis", s=.5)
            ax.scatter(qs[:, 0], qs[:, 1], color="red", s=.5)
            ax.plot(bs[:, 0], bs[:, 1], linewidth=.2, c='black')
            ax.set_aspect(aspect='equal', adjustable='box')

    fig, axes = plt.subplots(
        nrows=2, ncols=5,
        sharex="row", sharey="row",
        figsize=(6, 3),
    )

    circle = torch.vstack([
        torch.cos(torch.linspace(0, 2 * torch.pi, 1000)),
        torch.sin(torch.linspace(0, 2 * torch.pi, 1000)),
    ]).t()
    _plot(axes[0, 0], ex.gauge_samples, circle, torch.ones(ex.n_sample, dtype=torch.bool))
    _plot(axes[0, 1], ex.vanilla_gauge_gen[-1], circle, torch.linalg.vector_norm(ex.vanilla_gauge_gen[-1], dim=-1) <= 1)
    _plot(axes[0, 2], ex.reflect_gauge_gen[-1], circle, torch.linalg.vector_norm(ex.reflect_gauge_gen[-1], dim=-1) <= 1)
    _plot(axes[0, 3], ex.project_gauge_gen[-1], circle, torch.linalg.vector_norm(ex.project_gauge_gen[-1], dim=-1) <= 1)
    _plot(axes[0, 4], ex.mirror_gauge_gen, circle, torch.linalg.vector_norm(ex.mirror_gauge_gen, dim=-1) <= 1)

    dbr = ex.constrained_set_boundary(1000)
    _plot(axes[1, 0], ex.samples, dbr, torch.ones(ex.n_sample, dtype=torch.bool))
    _plot(axes[1, 1], ex.vanilla_gen, dbr, ex.constraint.check_feasibility_v(ex.vanilla_gen))
    _plot(axes[1, 2], ex.reflect_gen, dbr, ex.constraint.check_feasibility_v(ex.reflect_gen))
    _plot(axes[1, 3], ex.project_gen, dbr, ex.constraint.check_feasibility_v(ex.project_gen))
    _plot(axes[1, 4], ex.mirror_gen, dbr, ex.constraint.check_feasibility_v(ex.mirror_gen))

    axes[1, 0].set_xlabel("Ground Truth", fontsize=9)
    axes[1, 1].set_xlabel("Vanilla", fontsize=9)
    axes[1, 2].set_xlabel("Reflected", fontsize=9)
    axes[1, 3].set_xlabel("Projected", fontsize=9)
    axes[1, 4].set_xlabel("Mirrored", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(ex.dir, "samples.png"), dpi=300)
    print("Figure saved")


def test_time_kl(ex: ToyExample, n: int) -> None:
    import numpy as np
    import pandas as pd
    import ite

    print("Testing generation time and KL divergences...")
    times = np.zeros((n, 4))
    kls = np.zeros((n, 4))
    vio = np.zeros((n, 4))
    co = ite.cost.BDKL_KnnK()
    for i in range(n):
        times[i, 0] = ex.gen_vanilla()
        times[i, 1] = ex.gen_reflecting()
        times[i, 2] = ex.gen_projecting()
        times[i, 3] = ex.gen_mirror()
        with torch.no_grad():
            kls[i, 0] = co.estimation(ex.vanilla_gen.numpy(), ex.samples.numpy())
            kls[i, 1] = co.estimation(ex.reflect_gen.numpy(), ex.samples.numpy())
            kls[i, 2] = co.estimation(ex.project_gen.numpy(), ex.samples.numpy())
            kls[i, 3] = co.estimation(ex.mirror_gen.numpy(), ex.samples.numpy())
            vio[i, 0] = torch.sum(torch.linalg.vector_norm(ex.vanilla_gauge_gen[-1], dim=-1) <= 1, dtype=torch.int)
            vio[i, 1] = torch.sum(torch.linalg.vector_norm(ex.reflect_gauge_gen[-1], dim=-1) <= 1, dtype=torch.int)
            vio[i, 2] = torch.sum(torch.linalg.vector_norm(ex.project_gauge_gen[-1], dim=-1) <= 1, dtype=torch.int)
            vio[i, 3] = torch.sum(torch.linalg.vector_norm(ex.mirror_gauge_gen, dim=-1) <= 1, dtype=torch.int)
        print(f"Finished generation {i + 1}.")
    (pd
     .DataFrame(times, columns=["Vanilla", "Reflected", "Projected", "Mirror Map"])
     .to_csv(os.path.join(ex.dir, "inference_times.csv"))
     )
    (
        pd
        .DataFrame(kls, columns=["Vanilla", "Reflected", "Projected", "Mirror Map"])
        .to_csv(os.path.join(ex.dir, "kl_div.csv"))
    )
    (
        pd
        .DataFrame(vio, columns=["Vanilla", "Reflected", "Projected", "Mirror Map"])
        .to_csv(os.path.join(ex.dir, "violations.csv"))
    )


def eval_kl_div(ex: ToyExample) -> None:
    import ite

    co = ite.cost.BDKL_KnnK()
    with torch.no_grad():
        kl_vanilla = co.estimation(ex.vanilla_gen.numpy(), ex.samples.numpy())
        kl_reflected = co.estimation(ex.reflect_gen.numpy(), ex.samples.numpy())
        kl_projected = co.estimation(ex.project_gen.numpy(), ex.samples.numpy())
        kl_mirror = co.estimation(ex.mirror_gen.numpy(), ex.samples.numpy())
    with open(os.path.join(ex.dir, "kl_div.csv"), "w") as f:
        f.write("Vanilla, Reflected, Projected, Mirror Map\n")
        f.write(f"{kl_vanilla}, {kl_reflected}, {kl_projected}, {kl_mirror}\n")
    print("KL divergence evaluation done.")


if __name__ == "__main__":
    main()
