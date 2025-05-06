from sys import prefix

import torch, os
from torch import nn, tensor, Tensor
from torch.utils.data import DataLoader, TensorDataset

from misc import *
from gfm import *

__all__ = ["Example"]

from misc.flow_velocity import FlowVelocityDeep3


class Example:

    def __init__(self, args, dim: int):
        self.example = args.example
        self.device = args.device
        self.dim = dim
        self.norm = args.norm
        self.gen_sample = args.gen_sample
        self.n_sample = args.n_sample
        self.n_epoch = args.n_epoch
        self.batch_size = args.batch_size
        self.n_gen = args.n_gen
        self.n_step = args.n_step
        self.deep = args.deep
        self.hidden = args.hidden
        self.repeat = args.repeat
        self.output = args.output
        self.method = args.method
        self.verbose = args.verbose
        self.append = args.append
        # fields for to be defined by implementations
        self.velocity = None
        self.domain: ConstrainedSet | None = None
        self.gauge_map: GaugeMap | None = None
        # fields for the training process that should be filled by `init_training`
        self.true_samples = None
        self.training_samples = None
        self.prior_dist = None
        # fields for generating
        self.gen_x_1 = None

    def init_domain(self):
        raise NotImplementedError

    def data_distribution(self):
        raise NotImplementedError

    def init_prior(self, loc: Tensor | float, scale: Tensor | float):
        if self.method == "gauge_mirror":
            self.prior_dist = torch.distributions.MultivariateNormal(
                torch.zeros(self.dim, device=self.device),
                torch.eye(self.dim, device=self.device),
            )
        elif self.method.startswith("gauge"):
            self.prior_dist = HyperBallUniform(self.dim)
        else:
            if loc is not Tensor: loc = torch.full([self.dim], loc, dtype=torch.float32, device=self.device)
            if scale is not Tensor: scale = torch.full([self.dim], scale, dtype=torch.float32, device=self.device)
            self.prior_dist = HyperBoxUniform(loc.to(self.device), scale.to(self.device))

    def init_training(self):
        if self.gen_sample:
            data_dist = self.data_distribution()
            self.true_samples = data_dist.sample(torch.Size([self.n_sample]))
            if self.verbose: print(f"Samples generated: {self.n_sample}.")
        else:
            self.true_samples = torch.load(os.path.join(self.output, f"true_samples.pt"))
            if self.verbose: print("Samples loaded.")
        if self.method == "gauge_mirror":
            if self.norm == "L2":
                self.training_samples = unit_ball_mirror_map(self.gauge_map.to_disk(self.true_samples))
            else:
                self.training_samples = unit_cube_mirror_map(self.gauge_map.to_disk(self.true_samples))
        elif self.method.startswith("gauge"):
            self.training_samples = self.gauge_map.to_disk(self.true_samples)
        else:
            self.training_samples = self.true_samples
        self.velocity = FlowVelocityDeep3(self.dim, self.hidden) if self.deep else FlowVelocity(self.dim, self.hidden)

    def load_model(self):
        match self.method:
            case "vanilla" | "reflect" | "project":
                prefix = "vanilla"
            case "gauge_vanilla.yaml" | "gauge_reflect" | "gauge_project.yaml":
                prefix = "gauge"
            case "gauge_mirror":
                prefix = "mirror"
        self.velocity = torch.load(os.path.join(self.output, f"{prefix}_velocity{'_deep' if self.deep else ''}.pt"),
                                   map_location=self.device)

    def save_model(self):

        match self.method:
            case "vanilla" | "reflect" | "project":
                prefix = "vanilla"
            case "gauge_vanilla.yaml" | "gauge_reflect" | "gauge_project.yaml":
                prefix = "gauge"
            case "gauge_mirror":
                prefix = "mirror"
        torch.save(self.velocity, os.path.join(self.output, f"{prefix}_velocity{'_deep' if self.deep else ''}.pt"))

    def train(self):
        self.init_training()
        opt = torch.optim.Adam(self.velocity.parameters(), lr=5e-3, weight_decay=1e-5)
        sche = torch.optim.lr_scheduler.StepLR(opt, gamma=0.99, step_size=100)
        loss = nn.MSELoss()
        # loss = nn.L1Loss
        dl = DataLoader(TensorDataset(self.training_samples),
                        batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.n_epoch):
            z_1 = next(iter(dl))[0]
            z_0 = self.prior_dist.sample([self.batch_size]).to(self.device)
            t = torch.rand(self.batch_size, 1).to(self.device)
            z_t = (1 - t) * z_0 + t * z_1
            dz_t = z_1 - z_0
            opt.zero_grad()
            loss(self.velocity(t, z_t), dz_t).backward()
            opt.step()
            sche.step()
            if self.verbose and epoch % 1000 == 0:
                print(f'Epoch: {epoch}')

    def gen0(self) -> float:
        raise NotImplementedError

    def generate(self):
        if not os.path.exists(os.path.join(self.output, f"{self.method_name()}_gen")):
            os.mkdir(os.path.join(self.output, f"{self.method_name()}_gen"))
        for i in range(self.repeat):
            t = self.gen0()
            torch.save(self.gen_x_1, os.path.join(self.output, f"{self.method_name()}_gen/{i}.pt"))
            if self.verbose:
                print(f"Generated {i} in {t:.2f}s.")

    def generate_test(self):
        import pandas as pd, ite

        if not os.path.exists(os.path.join(self.output, f"{self.method_name()}_gen")):
            os.mkdir(os.path.join(self.output, f"{self.method_name()}_gen"))

        co = ite.cost.BDKL_KnnK()
        stats = torch.zeros(self.repeat, 4)
        for i in range(self.repeat):
            t = self.gen0()
            torch.save(self.gen_x_1, os.path.join(self.output, f"{self.method_name()}_gen/{i}.pt"))
            if self.verbose: print(f"Generated {i} in {t:.2f}s.")
            with torch.no_grad():
                kl = co.estimation(self.gen_x_1, self.true_samples)
                mmd = (mmd_square(self.gen_x_1, self.true_samples)) ** 0.5
                fr = self.domain.check_feasibility_v(self.gen_x_1).sum()
            stats[i, 0] = t
            stats[i, 1] = kl
            stats[i, 2] = mmd
            stats[i, 3] = fr
        if self.append and os.path.exists(os.path.join(self.output, f"{self.method_name()}_stats.csv")):
            (pd
             .DataFrame(stats, columns=["Time", "KL", "MMD", "Feasibility"])
             .to_csv(os.path.join(self.output, f"{self.method_name()}_stats.csv"), index=False, mode="a", header=False))
        else:
            (pd
             .DataFrame(stats, columns=["Time", "KL", "MMD", "Feasibility"])
             .to_csv(os.path.join(self.output, f"{self.method_name()}_stats.csv"), index=False))

    def method_name(self):
        return f"{self.method}_deep" if self.deep else self.method
