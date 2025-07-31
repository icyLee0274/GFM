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
        self._init_domain()

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
            num_workers=self.cfg.train.get("dataloader_workers", 0),
            generator=torch.Generator(device=self.device),
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

        # upper, lower = {
        #     "ffhq": (1.05, -1.05),
        # "afhqv2": (0.9, -0.9),
        # }.get(self.cfg.example.dataset_name, (None, None))
        # if upper is None: raise RuntimeError(f"Invalid dataset name {self.cfg.example.dataset_name}.")
        upper = self.cfg.example.upper
        lower = self.cfg.example.lower

        # The basis is a tensor of shape (d, K)
        # The constraints are given by: lower <= x `basis` <= upper
        dim = self.cfg.example.dimension
        cons = basis.shape[1]
        n_pixels = self.cfg.example.n_pixels
        basis[n_pixels:, :] = 0

        if basis.shape[0] < dim:
            ex_basis = torch.zeros(dim, cons, device=self.device)
            ex_basis[:basis.shape[0], :] = basis
            basis = ex_basis

        At = torch.cat([basis, -basis], dim=1)
        b = torch.cat([torch.full([cons], upper, device=self.device),
                       torch.full([cons], -lower, device=self.device)], dim=0)
        self.register_buffer('basis', torch.cat([basis[:n_pixels, :], -basis[:n_pixels, :]], dim=1))
        self.register_buffer('bounds', b)
        domain = gfm.LinearConstraint(b=b, At=At, box_lower=-1.0, box_upper=1.0)

        return domain, torch.zeros(basis.shape[0], device=self.device)

    def generate_basis(self):
        dim = self.cfg.example.n_pixels
        Z = torch.randn(dim, dim, device=self.device)
        Q, _ = torch.linalg.qr(Z, mode="complete")
        return Q

    def _reflect_rf(self):

        rf0 = gfm.PolytopeReflector(self.basis, self.bounds, box_lower=-1.0, box_upper=1.0)
        rf1 = gfm.cube_reflect
        n_pixels = self.cfg.example.n_pixels

        def reflect_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = torch.zeros_like(vs)
            xs[:, :n_pixels] = rf0(os[:, :n_pixels], vs[:, :n_pixels])
            xs[:, n_pixels:] = rf1(os[:, n_pixels:], vs[:, n_pixels:])
            return xs

        return reflect_fn

    def _project_rf(self):
        pf0 = gfm.PolytopeProjector(self.basis, self.bounds, box_lower=-1.0, box_upper=1.0)
        pf1 = gfm.cube_project
        n_pixels = self.cfg.example.n_pixels

        def project_fn(os: Tensor, vs: Tensor) -> Tensor:
            xs = torch.zeros_like(vs)
            xs[:, :n_pixels] = pf0(os[:, :n_pixels], vs[:, :n_pixels])
            xs[:, n_pixels:] = pf1(os[:, n_pixels:], vs[:, n_pixels:])
            return xs

        return project_fn

    def _vectorize(self, xs: Tensor) -> Tensor:
        return xs.reshape(xs.shape[0], -1)

    def training_step(self, batch, batch_idx):
        x_1 = self._vectorize(batch[0])
        z_1 = self.transform(x_1)
        if self.cfg.method.name.startswith("gauge"):
            z_1[z_1 > 1.0] = 1.0
            z_1[z_1 < -1.0] = -1.0
        return super().training_step([z_1], batch_idx)

    def get_prior(self) -> torch.distributions.Distribution:
        dist = getattr(self, "_prior", None)
        if dist is None:
            dim = self.cfg.example.dimension
            match self.cfg.method.name:
                case "vanilla":
                    dist = gfm.box_uniform(torch.zeros(dim, device=self.device), torch.ones(dim, device=self.device))
                case "reflect" | "project":
                    dist = gfm.TruncatedDistribution(
                        gfm.box_uniform(torch.zeros(dim, device=self.device), torch.ones(dim, device=self.device)),
                        self.get_domain(),
                        self.device
                    )
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

    def _init_transformation(self):
        transform = getattr(self, "_transform", None)
        inverse_transform = getattr(self, "_inverse_transform", None)
        if transform is None:
            match self.cfg.method.transform:
                case "L2":
                    raise NotImplementedError("L2 transformation is not used for images.")
                case "L_inf":
                    n_pixels = self.cfg.example.n_pixels
                    domain = gfm.LinearConstraint(b=self.bounds, At=self.basis, box_lower=-1.0, box_upper=1.0)
                    gauge_map = gfm.GaugeMap(domain, torch.zeros(n_pixels, device=self.device), "cube")

                    def phi(x: Tensor) -> Tensor:
                        y = x.clone()
                        y[:, :n_pixels] = gauge_map.to_disk(x[:, :n_pixels])
                        return y

                    def phi_inv(z: Tensor) -> Tensor:
                        y = z.clone()
                        y[:, :n_pixels] = gauge_map.from_disk(z[:, :n_pixels])
                        return y

                    transform = phi
                    inverse_transform = phi_inv
                case None:
                    transform = lambda x: x
                    inverse_transform = lambda x: x
            setattr(self, "_transform", transform)
            setattr(self, "_inverse_transform", inverse_transform)
