import torch, os
from torch import nn, tensor, Tensor
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["Example"]

from CvxIneq import GaugeMap
from misc import ConstrainedSet
from misc.mmd import mmd_square


class Example:

    def __init__(self, args):
        self.example = args.example
        self.device = args.device
        self.gen_sample = args.gen_sample
        self.n_sample = args.n_sample
        self.n_epoch = args.n_epoch
        self.batch_size = args.batch_size
        self.n_gen = args.n_gen
        self.n_step = args.n_step
        self.hidden = args.hidden
        self.repeat = args.repeat
        self.output = args.output
        self.method = args.method
        self.verbose = args.verbose
        # fields for to be defined by implementations
        self.velocity = None
        self.domain: ConstrainedSet | None = None
        self.gauge_map: GaugeMap | None = None
        # fields for training, should be filled by `init_training`
        self.true_samples = None
        self.training_samples = None
        self.prior_dist = None
        # fields for generating
        self.gen_x_1 = None

    def init_domain(self):
        raise NotImplementedError

    def init_training(self):
        raise NotImplementedError

    def train(self):
        self.init_training()
        opt = torch.optim.Adam(self.velocity.parameters(), lr=1e-3)
        loss = nn.MSELoss()
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
            if self.verbose and epoch % 1000 == 0:
                print(f'Epoch: {epoch}')

    def gen0(self) -> float:
        raise NotImplementedError

    def generate(self):
        for i in range(self.repeat):
            t = self.gen0()
            torch.save(self.gen_x_1, os.path.join(self.output, f"{self.method}_gen{i}.pt"))
            if self.verbose:
                print(f"Generated {i} in {t:.2f}s.")

    def generate_test(self):
        import pandas as pd, ite

        co = ite.cost.BDKL_KnnK()
        stats = torch.zeros(self.repeat, 3)
        for i in range(self.repeat):
            t = self.gen0()
            torch.save(self.gen_x_1, os.path.join(self.output, f"{self.method}_gen{i}.pt"))
            if self.verbose: print(f"Generated {i} in {t:.2f}s.")
            with torch.no_grad():
                kl = co.estimation(self.gen_x_1, self.true_samples)
                mmd = mmd_square(self.gen_x_1, self.true_samples)
            stats[i, 0] = t
            stats[i, 1] = kl
            stats[i, 2] = mmd
        (pd
         .DataFrame(stats, columns=["Time", "KL", "MMD"])
         .to_csv(os.path.join(self.output, f"{self.method}_stats.csv"), index=False))
