import torch
from pypower.case9 import case9
from sympy.strategies.core import switch
from torch import nn, tensor, Tensor
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
    ball_project, GaugeMap
)

from argparse import ArgumentParser


class SpdFlow(nn.Module):

    def __init__(self, dim: int = 3, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, traj: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        if traj.dim() == 0 or (traj.dim() == 1 and traj.shape[0] == 1):
            traj = traj.expand(len(x_t), 1)
        return self.net(torch.cat((t, traj, x_t), dim=-1))


class SpdExample:

    def __init__(self, device, n_epoch, batch_size, n_gen, n_step, hidden, output):
        self.device = device
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_gen = n_gen
        self.n_step = n_step
        self.hidden = hidden
        self.output = output
        self.constraint = Intersection(
            SemiDefiniteConstraint(tensor([
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 1], [1, 0]],
                [[0, 0], [0, 1]]
            ], dtype=torch.float32, device=self.device)),
            LinearConstraint(
                tensor([[1., 0., 1.]], device=self.device),
                tensor([5.99], device=self.device)
            )
        )
        self.x0 = tensor([2., .3, 1.5], device=self.device)
        self.spd_samples = torch.load("data/SPD-L.pt", map_location=self.device).to(dtype=torch.float32)
        self.spd_t = torch.linspace(0, 1, 10000, device=self.device).repeat(15)
        self.gauge_map = GaugeMap(self.constraint, self.x0)
        self.gauge_samples = self.gauge_map.to_disk(self.spd_samples, device=self.device)
        self.prior_dist = TruncatedDistribution(
            HyperBoxUniform(
                tensor([-1, -1, -1], device=self.device),
                tensor([2, 2, 2], device=self.device)
            ),
            BallConstraint(torch.zeros(3, device=self.device)),
            device=self.device
        )
        self.velocity = None

    def train(self):
        velocity = SpdFlow(dim=3, h=self.hidden).to(self.device)
        opt = torch.optim.Adam(velocity.parameters(), lr=1e-3)
        loss = nn.MSELoss()
        dl = DataLoader(TensorDataset(self.gauge_samples, self.spd_t), batch_size=self.batch_size, shuffle=True)
        for i in range(self.n_epoch):
            z_1, traj_t = next(iter(dl))
            z_0 = self.prior_dist.sample((self.batch_size,))
            t = torch.rand(self.batch_size, 1)
            z_t = (1 - t) * z_0 + t * z_1
            dz_t = z_1 - z_0
            opt.zero_grad()
            loss(velocity(t, traj_t.view(-1, 1), z_t), dz_t).backward()
            opt.step()
            if i % 500 == 0:
                print(f"Trained epoch {i}.")
        self.velocity = velocity

    def gen_vanilla(self) -> tuple[float, Tensor]:
        z_0 = self.prior_dist.sample((self.n_gen,))
        traj_t = torch.linspace(0, 1, self.n_gen, device=self.device).view(-1, 1)
        start = time.time()
        z_t = odeint(
            lambda t, z: self.velocity(t, traj_t, z),
            z_0,
            torch.linspace(0, 1, self.n_step, device=self.device)
        )
        x_1 = self.gauge_map.from_disk(z_t[-1])
        end = time.time()
        return end - start, x_1

    def gen_reflected(self) -> tuple[float, Tensor]:
        z_0 = self.prior_dist.sample((self.n_gen,))
        traj_t = torch.linspace(0, 1, self.n_gen, device=self.device).view(-1, 1)
        start = time.time()
        z_t = odeint_reflect(
            lambda t, z: self.velocity(t, traj_t, z),
            z_0,
            torch.linspace(0, 1, self.n_step, device=self.device),
            reflect_fn=ball_reflect,
            reflect_velocity_fn=ball_reflect_velocity,
        )
        x_1 = self.gauge_map.from_disk(z_t[-1])
        end = time.time()
        return end - start, x_1

    def gen_projected(self) -> tuple[float, Tensor]:
        z_0 = self.prior_dist.sample((self.n_gen,))
        traj_t = torch.linspace(0, 1, self.n_gen, device=self.device).view(-1, 1)
        start = time.time()
        z_t = odeint_reflect(
            lambda t, z: self.velocity(t, traj_t, z),
            z_0,
            torch.linspace(0, 1, self.n_step, device=self.device),
            reflect_fn=ball_project,
            reflect_velocity_fn=ball_reflect_velocity,
        )
        x_1 = self.gauge_map.from_disk(z_t[-1])
        end = time.time()
        return end - start, x_1


def main():
    parser = ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--n_gen", type=int, default=10)
    parser.add_argument("--n_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--output", type=str, default="./spd-out")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--method", type=str, default="vanilla", choices=["vanilla", "reflect", "project"])

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ex = SpdExample(
        device,
        args.n_epoch,
        args.batch_size,
        args.n_gen,
        args.n_step,
        args.hidden,
        args.output
    )

    print("Initialized.")

    if args.repeat <= 0:
        print("Training...")
        ex.train()
        print("Saving velocity...")
        torch.save(ex.velocity, os.path.join(args.output, "velocity.pt"))
    else:
        print("Loading velocity...")
        ex.velocity = torch.load(os.path.join(args.output, "velocity.pt"), map_location=device)

        print("Generating...")
        for i in range(args.repeat):
            match args.method:
                case "vanilla":
                    t, x = ex.gen_vanilla()
                case "reflect":
                    t, x = ex.gen_reflected()
                case "project":
                    t, x = ex.gen_projected()
            print(f"{i}: Generated in {t:.2f}s.")
            torch.save(x, os.path.join(args.output, f"{args.method}_gen{i}.pt"))


if __name__ == "__main__":
    main()
