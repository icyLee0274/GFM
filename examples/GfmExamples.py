import logging
import os
from time import time

import geoopt
import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch import nn, tensor, Tensor
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

import gfm
from gfm import (
    GaugeMap,
    unit_cube_mirror_map, unit_cube_dual_map,
    unit_ball_mirror_map, unit_ball_dual_map,
    odeint_reflect,
    cube_reflect, ball_reflect,
    cube_project, ball_project,
    HyperBallUniform, box_uniform,
    maximum_mean_discrepancy, ConstrainedSet,
    TruncatedDistribution,
)

logger = logging.getLogger(__name__)


class GfmExampleBase(lightning.LightningModule):
    """
    Base class for implementing GFM examples.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # TODO: load velocity field dynamics
        vf_impl = getattr(gfm, self.cfg.velocity.implementation)
        vf_args = OmegaConf.to_container(self.cfg.velocity, resolve=True)
        vf_args.pop("implementation")
        if vf_args.get("n_in", None) is None: vf_args["n_in"] = self.cfg.example.dimension + 1
        if vf_args.get("n_out", None) is None: vf_args["n_out"] = self.cfg.example.dimension
        if self.cfg.method.get("override_n_in", None) is not None:
            vf_args["n_in"] = eval(self.cfg.method.override_n_in)
        if self.cfg.method.get("override_n_out", None) is not None:
            vf_args["n_out"] = eval(self.cfg.method.override_n_out)
        self.velocity = vf_impl(**vf_args)
        # self.velocity = Mlp(
        #     cfg.example.dimension + 1,
        #     cfg.example.dimension,
        #     cfg.velocity.width,
        #     cfg.velocity.depth,
        #     cfg.velocity.activation
        # ) if cfg.velocity.get("implementation", "MlpVelocityField") == "MlpVelocityField" else (
        #     ResNet(
        #         cfg.example.dimension + 1,
        #         cfg.example.dimension,
        #         cfg.velocity.width,
        #         cfg.velocity.layers,
        #     ))
        self.save_hyperparameters()

        #### The following buffers are for DDPM only ####
        # Precompute forward process constants
        if self.cfg.method.name == "ddpm":
            beta = torch.linspace(1e-4, 0.02, self.cfg.method.horizon)
            alpha = 1. - beta
            self.register_buffer('beta', beta)
            self.register_buffer('alpha', alpha)
            self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))
            self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
            self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - self.alpha_bar))

    def get_loss(self):
        loss = getattr(self, "_loss", None)
        if loss is None:
            loss = getattr(nn, self.cfg.train.loss)()
            setattr(self, "_loss", loss)
        return loss

    def get_prior(self) -> torch.distributions.Distribution:
        dist = getattr(self, "_prior", None)
        dim = self.cfg.example.dimension
        scale = self.cfg.method.scale
        if dist is None:
            match self.cfg.method.transform:
                case "L2":
                    dist = HyperBallUniform(dim, loc=torch.zeros(dim, device=self.device), scale=scale)
                case "L_inf":
                    dist = box_uniform(torch.zeros(dim),
                                       torch.full([dim], scale))
                case "mirror_2" | "mirror_inf":
                    dist = MultivariateNormal(
                        torch.zeros(dim),
                        scale * torch.eye(dim)
                    )
                case None:
                    match self.cfg.method.name:
                        case "vanilla" | "ddpm":
                            dist = MultivariateNormal(
                                torch.zeros(dim),
                                scale * torch.eye(dim)
                            )
                        case "reflect" | "project" | "metropolis":
                            dist = TruncatedDistribution(
                                box_uniform(
                                    self.get_interior_point() if self.cfg.example.prior_center is None
                                    else tensor(self.cfg.example.prior_center, device=self.device),
                                    tensor(self.cfg.example.prior_scale, device=self.device),
                                ),
                                self.get_domain(),
                                self.device,
                            )
            setattr(self, "_prior", dist)
        return dist

    def get_domain(self) -> gfm.ConstrainedSet:
        domain = getattr(self, "_domain", None)
        if domain is None:
            domain, ip = self._init_domain()
            setattr(self, "_domain", domain)
            setattr(self, "_ip", ip)
        return domain

    def get_interior_point(self) -> Tensor:
        ip = getattr(self, "_ip", None)
        if ip is None:
            domain, ip = self._init_domain()
            setattr(self, "_domain", domain)
            setattr(self, "_ip", ip)
        return ip

    def get_data(self) -> Tensor:
        """
        Returns the true data samples.

        If the data file specified by {out_prefix}/{example.samples.file} does not exist,
        data samples are generated by the `_init_data` method.
        :return: Data samples, Tensor of N * dim.
        """
        samples = getattr(self, "_data", None)
        if samples is None:
            data_file = os.path.join(self.cfg.out_prefix, self.cfg.example.data_file)
            if os.path.exists(data_file) and os.path.isfile(data_file):
                samples = torch.load(data_file, map_location=self.device)
            else:
                samples = self._init_data(self.cfg.example.n_samples)
                torch.save(samples, data_file)
            setattr(self, "_data", samples)
        return samples

    def get_manifold(self) -> geoopt.Manifold | None:
        """
        Returns the manifold on which the data lies.

        :return: `None` if Euclidean space, otherwise the manifold.
        """
        return None

    def transform(self, xs: Tensor) -> Tensor:
        transform = getattr(self, "_transform", None)
        if transform is None:
            self._init_transformation()
            transform = getattr(self, "_transform", None)
        return transform(xs)

    def inverse_transform(self, zs: Tensor) -> Tensor:
        inverse_transform = getattr(self, "_inverse_transform", None)
        if inverse_transform is None:
            self._init_transformation()
            inverse_transform = getattr(self, "_inverse_transform", None)
        return inverse_transform(zs)

    def get_reflect_fn(self):
        rf = getattr(self, "_reflect_fn", None)
        if rf is None:
            match self.cfg.method.name:
                case "vanilla" | "gauge_vanilla" | "gauge_mirror" | "ddpm":
                    rf = None
                case "reflect":
                    rf = self._reflect_rf()
                case "project":
                    rf = self._project_rf()
                case "metropolis":
                    rf = gfm.MetropolisSampler(self.get_domain())
                case "gauge_reflect":
                    rf = cube_reflect if self.cfg.method.transform == "L_inf" else ball_reflect
                case "gauge_project":
                    rf = cube_project if self.cfg.method.transform == "L_inf" else ball_project
                case _:
                    raise NotImplementedError
            setattr(self, "_reflect_fn", rf)
        return rf

    def _reflect_rf(self):
        raise NotImplementedError

    def _project_rf(self):
        raise NotImplementedError

    def _init_domain(self) -> tuple[ConstrainedSet, Tensor]:
        raise NotImplementedError

    def _init_data(self, n: int) -> Tensor:
        raise NotImplementedError

    def _init_transformation(self):
        transform = getattr(self, "_transform", None)
        inverse_transform = getattr(self, "_inverse_transform", None)
        manifold = self.get_manifold()
        if transform is None:
            match self.cfg.method.transform:
                # TODO: scale the gauge map
                case "L2":
                    gauge_map = GaugeMap(self.get_domain(), self.get_interior_point(), "ball", manifold)
                    transform = lambda x: gauge_map.to_disk(x)
                    inverse_transform = lambda x: gauge_map.from_disk(x)
                case "L_inf":
                    gauge_map = GaugeMap(self.get_domain(), self.get_interior_point(), "cube", manifold)
                    transform = lambda x: gauge_map.to_disk(x)
                    inverse_transform = lambda x: gauge_map.from_disk(x)
                case "mirror_2":
                    gauge_map = GaugeMap(self.get_domain(), self.get_interior_point(), "ball", manifold)
                    transform = lambda x: unit_ball_mirror_map(gauge_map.to_disk(x))
                    inverse_transform = lambda x: gauge_map.from_disk(unit_ball_dual_map(x))
                case "mirror_inf":
                    gauge_map = GaugeMap(self.get_domain(), self.get_interior_point(), "cube", manifold)
                    transform = lambda x: unit_cube_mirror_map(gauge_map.to_disk(x))
                    inverse_transform = lambda x: gauge_map.from_disk(unit_cube_dual_map(x))
                case None:
                    transform = lambda x: x
                    inverse_transform = lambda x: x
            setattr(self, "_transform", transform)
            setattr(self, "_inverse_transform", inverse_transform)

    def configure_optimizers(self):
        opti = getattr(torch.optim, self.cfg.train.optimizer)(
            self.parameters(),
            **OmegaConf.to_container(self.cfg.train.optimizer_args)
        )
        sche = torch.optim.lr_scheduler.StepLR(
            opti,
            **OmegaConf.to_container(self.cfg.train.scheduler_args)
        )
        return {
            "optimizer": opti,
            "lr_scheduler": {
                "scheduler": sche,
                "interval": "step",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        data = self.get_data()
        training_data = self.transform(data)
        return DataLoader(
            TensorDataset(training_data),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            # https://stackoverflow.com/questions/68621210/runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu
            generator=torch.Generator(device=self.device),
        )

    def test_dataloader(self):
        return [
            DataLoader(
                TensorDataset(tensor([self.cfg.test.n_gen])),
                batch_size=1, shuffle=False
            )
            for _ in range(self.cfg.test.repeats)
        ]

    def on_train_start(self) -> None:
        self.velocity.to(self.device)

    def on_test_start(self) -> None:
        self.velocity.to(self.device)

    def training_step(self, batch, batch_idx):
        if self.cfg.method.name == "ddpm":
            return self.ddpm_training_step(batch)
        z_1 = batch[0]
        z_0 = self.get_prior().sample([len(z_1)]).to(z_1)
        t = torch.rand(len(z_1), 1).to(z_1)
        z_t = (1 - t) * z_0 + t * z_1
        dz_t = z_1 - z_0
        loss = self.get_loss()(self.velocity(t, z_t), dz_t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def sample(self, n_samples: int, n_steps: int) -> Tensor:
        start = time()
        z_0 = self.get_prior().sample([n_samples]).to(self.device)
        prior_time = time() - start
        t = torch.linspace(0, 1, n_steps).to(self.device)
        start = time()
        z_1 = (self.integrate_ddpm(z_0) if self.cfg.method.name == "ddpm"
               else odeint_reflect(self.velocity, z_0, t, self.get_reflect_fn())[-1])
        integral_time = time() - start
        start = time()
        x_1 = self.inverse_transform(z_1)
        transform_time = time() - start
        self.log("prior_time", prior_time)
        self.log("integral_time", integral_time)
        self.log("transform_time", transform_time)
        return x_1

    @torch.no_grad()
    def test_step(self, *args, **kwargs) -> STEP_OUTPUT:
        # co = ite.cost.BDKL_KnnK()
        x_1 = self.sample(self.cfg.test.n_gen, self.cfg.test.n_steps)
        data = self.get_data()
        # kl = co.estimation(x_1.cpu().numpy(), data.cpu().numpy())
        mmd = maximum_mean_discrepancy(x_1, data)
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

    ###### THE FOLLOWING METHODS ARE FOR DDPM ONLY ######

    def q_sample(self, t: Tensor, x_0: Tensor, noise: Tensor) -> Tensor:
        """
        Sample from the forward process at time t.
        :param t: Time step, shape (N, 1)
        :param x_0: Current sample, shape (N, dim)
        :param noise: Noise, shape (N, dim)
        :return: Sample at time t, shape (N, dim)
        """
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1).to(x_0)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).to(x_0)
        return sqrt_ab * x_0 + sqrt_1mab * noise

    def p_sample(self, t: Tensor, x_t: Tensor) -> Tensor:
        """
        Sample from the reverse process at time t.
        :param t: Time step, shape (N, 1)
        :param x_t: Current sample, shape (N, dim)
        :return: Sample at time t-1, shape (N, dim)
        """
        pred_noise = self.velocity(t * 1.0 / self.cfg.method.horizon, x_t)
        beta_t = self.beta[t].unsqueeze(-1).to(x_t)
        alpha_t = self.alpha[t].unsqueeze(-1).to(x_t)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1).to(x_t)

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise)
        if t[0] == 0:
            return mean
        noise = torch.randn_like(x_t)
        x_t1 = mean + torch.sqrt(beta_t) * noise
        return x_t1

    def ddpm_training_step(self, batch):
        x0 = batch[0]
        t = torch.randint(0, self.cfg.method.horizon, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(t, x0, noise)
        pred_noise = self.velocity(t * 1.0 / self.cfg.method.horizon, xt)
        loss = self.get_loss()(pred_noise, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def integrate_ddpm(self, x0: Tensor) -> Tensor:
        x = x0
        for t in reversed(range(self.cfg.method.horizon)):
            t_batch = torch.full((x0.shape[0],), t, dtype=torch.long, device=x0.device)
            x = self.p_sample(t_batch, x)
        return x

    ###### Training and sampling methods for mean flow ######

    def meanflow_training_step(self, batch):
        """
        Training step for mean flow.
        :param batch: Batch of data samples.
        :return: Loss value.
        """
        # Remarks:
        # Notations in "Mean Flows for One-step Generative Modeling" is inconsistent with the code here,
        # which is consistent with the original FLow-Matching paper
        x_1 = batch[0]
        x_0 = self.get_prior().sample([len(x_1)]).to(x_1)
        t_s = torch.rand(x_1.size(0), 2).to(x_1)
        t, s = torch.aminmax(t_s, dim=1, keepdim=True)  # keep dim so that t and s are (N, 1)
        x_t = (1 - t) * x_0 + t * x_1
        v = x_1 - x_0

        u, dudt = torch.func.jvp(self.velocity, (x_t, t, s), (v, torch.ones_like(t), torch.zeros_like(s)))

        u_tgt = v + (s - t) * dudt

        loss = self.get_loss()(u, u_tgt.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def meanflow_integrate(self, x0: Tensor, n_steps: int = 1) -> Tensor:
        """
        Integrate the mean flow velocity field.
        :param x0: Initial sample, shape (N, dim).
        :param n_steps: Number of integration steps.
        :return: Integrated sample, shape (N, dim).
        """
        n = x0.shape[0]
        t = torch.linspace(0, 1, n_steps + 1).to(x0.device).expand(n, -1).T.unsqueeze(-1)  # shape: (n_steps+1, N, 1)
        x_t = x0
        for i in range(n_steps - 1):
            # t=t[i], s=t[i+1]
            u = self.velocity(x_t, t[i], t[i + 1])
            # for each step, x_{t+1} = x_t + (s-t) * u_\theta(x_t, t, s)
            x_t += u * (t[i + 1] - t[i])
        return x_t
