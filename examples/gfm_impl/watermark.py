import logging
import os

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor
import pyomo.environ as pyo
from torch.distributions import MultivariateNormal

import gfm
import examples
from examples.gfm_impl.image_dataset import ImageFolderDataset
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class Watermark(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _init_data(self, n: int) -> Tensor:
        raise RuntimeError(
            "Load data through dataset instead of raw data. This example overrides the dataloader methods."
        )

    def train_dataloader(self):
        dataset = ImageFolderDataset(
            path=self.cfg.example.data_path,
            resolution=None,
            use_pyspng=self.cfg.example.use_pyspng,
            max_size=self.cfg.example.max_size,
            use_labels=False,
            xflip=False,
            random_seed=self.cfg.example.dataset_seed,
            cache=self.cfg.example.dataset_cache,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.get("dataloader_workers", 1),
        )

    def test_step(self, *args, **kwargs) -> STEP_OUTPUT:
        x_1 = self.sample(self.cfg.test.n_gen, self.cfg.test.n_steps)
        fea = self.get_domain().check_feasibility_v(x_1).sum() * 1.0

        self.log("feasible", fea)

        if self.cfg.test.get("save_gen", False):
            i = 0
            while os.path.exists(f"gen_samples_{i}.pt"):
                i += 1
            f_name = os.path.abspath(f"gen_samples_{i}.pt")
            torch.save(x_1, f_name)
            logger.info("Saved gen samples to %s", f_name)

        return {"loss": 0, "feasible": fea}

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        f_basis = self.cfg.example.basis_path
        if f_basis is None or not os.path.isfile(f_basis):
            raise RuntimeError(f"Basis file {f_basis} does not exist.")
        basis = torch.load(f_basis, map_location=self.device, weights_only=True)
        logger.info("Loaded basis from %s", f_basis)

        upper, lower = {
            "ffhq": (1.05, -1.05),
            "afhqv2": (0.9, -0.9),
        }.get(self.cfg.example.dataset_name, (None, None))
        if upper is None: raise RuntimeError(f"Invalid dataset name {self.cfg.example.dataset_name}.")

        # The basis is a tensor of shape (d, K)
        # The constraints are given by: lower <= x `basis` <= upper
        dim = basis.shape[0]
        cons = basis.shape[1]
        At = torch.cat([basis, -basis], dim=1)
        b = torch.cat([torch.full(cons, upper, device=self.device),
                       torch.full(cons, -lower, device=self.device)], dim=0)
        self.register_buffer('basis', At)
        self.register_buffer('bounds', b)
        domain = gfm.LinearConstraint(b=b, At=At, box_lower=-1.0, box_upper=1.0)

        return domain, torch.zeros(basis.shape[0], device=self.device)

    def _reflect_rf(self):
        return gfm.PolytopeReflector(self.basis, self.bounds, box_lower=-1.0, box_upper=1.0)

    def _project_rf(self):
        return gfm.PolytopeProjector(self.basis, self.bounds)

    def vectorize(self, xs: Tensor) -> Tensor:
        return xs.reshape(xs.shape[0], -1)

    def training_step(self, batch, batch_idx):
        x_1 = self.vectorize(batch[0])
        z_1 = self.transform(x_1)
        return super().training_step([z_1], batch_idx)

    def get_prior(self) -> torch.distributions.Distribution:
        dist = getattr(self, "_prior", None)
        if dist is None:
            dim = self.cfg.example.dimension
            match self.cfg.method.name:
                case "vanilla" | "reflect" | "project":
                    dist = gfm.box_uniform(torch.zeros(dim, device=self.device), torch.ones(dim, device=self.device))
                case "ddpm":
                    dist = MultivariateNormal(torch.zeros(dim, device=self.device), torch.eye(dim, device=self.device))
                case "gauge_reflect" | "gauge_project":
                    if self.cfg.method.transform == "L_inf":
                        dist = gfm.box_uniform(torch.zeros(dim, device=self.device),
                                               torch.ones(dim, device=self.device))
                    elif self.cfg.method.transform == "L_2":
                        dist = gfm.HyperBallUniform(dim, torch.zeros(dim, device=self.device), 1)
                case _:
                    raise NotImplementedError(f"Method {self.cfg.method.name} not implemented for prior distribution.")
            setattr(self, "_prior", dist)
        return dist
