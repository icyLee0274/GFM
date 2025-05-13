import time
import logging
import os
import sys

from omegaconf import OmegaConf, DictConfig
import hydra
import torch
import lightning
import numpy as np
import cvxpy as cp
import pandas as pd

sys.path.append('../')
import examples.gfm_impl as impls

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="gfm_examples", version_base=None)
def main(cfg: DictConfig):
    if cfg.accelerator == "gpu":
        if torch.cuda.is_available():
            if cfg.devices is not str and hasattr(cfg.devices, "__getitem__"):
                torch.set_default_device(f"cuda:{cfg.devices[0]}")
            else:
                torch.set_default_device("cuda")
        elif torch.backends.mps.is_available():
            torch.set_default_device("mps")
        else:
            raise RuntimeError("Cannot determine GPU device.")
    else:
        torch.set_default_device(cfg.accelerator)

    torch.set_default_dtype(torch.float32)

    match cfg.action:
        case "train":
            on_train(cfg)
        case "test":
            on_test(cfg)


def on_train(cfg: DictConfig):
    impl = getattr(impls, cfg.example.implementation, None)
    if impl is None: raise NotImplementedError(
        f"Implementation {cfg.example.implementation} for example {cfg.example.name} not found.")
    model = impl(cfg)
    logger.info("Model initialized.")

    ckpt_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.out_prefix, "checkpoints"),
        filename=f"{cfg.method.name}-{{epoch:02d}}",
        every_n_epochs=cfg.train.checkpoint,
        save_last=False,
        save_top_k=-1,
    )
    trainer = lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.train.epoch,
        # logger=logger,
        callbacks=[ckpt_callback],
        log_every_n_steps=10,
    )

    logger.info("Starting training.")
    trainer.fit(model)

    f = os.path.abspath(os.path.join(cfg.out_prefix, f"{cfg.method.name}-final.ckpt"))
    trainer.save_checkpoint(f)
    logger.info("Model saved to %s", f)


def on_test(cfg: DictConfig):
    impl = getattr(impls, cfg.example.implementation, None)
    if impl is None: raise NotImplementedError(
        f"Implementation {cfg.example.implementation} for example {cfg.example.name} not found.")
    model = impl.load_from_checkpoint(
        cfg.test.checkpoint if cfg.test.get("checkpoint", None) is not None else
        os.path.join(cfg.out_prefix, f"{cfg.method.name}-final.ckpt"),
        cfg=cfg,
    )
    logger.info("Model initialized.")

    trainer = lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        log_every_n_steps=1,
    )

    res = trainer.test(model)

    if len(res) == 1:
        kl = [r["kl"] for r in res]
        mmd = [r["mmd"] for r in res]
        fea = [r["feasible"] for r in res]
        p_time = [r["prior_time"] for r in res]
        t_time = [r["transform_time"] for r in res]
        i_time = [r["integral_time"] for r in res]
    else:
        kl = [r[f"kl/dataloader_idx_{i}"] for i, r in enumerate(res)]
        mmd = [r[f"mmd/dataloader_idx_{i}"] for i, r in enumerate(res)]
        fea = [r[f"feasible/dataloader_idx_{i}"] for i, r in enumerate(res)]
        p_time = [r[f"prior_time/dataloader_idx_{i}"] for i, r in enumerate(res)]
        t_time = [r[f"transform_time/dataloader_idx_{i}"] for i, r in enumerate(res)]
        i_time = [r[f"integral_time/dataloader_idx_{i}"] for i, r in enumerate(res)]

    pd.DataFrame({
        "kl": kl,
        "mmd": mmd,
        "feasible": fea,
        "prior_time": p_time,
        "transformation_time": t_time,
        "integral_time": i_time,
        "method": cfg.method.name,
    }).to_csv(os.path.join(cfg.out_prefix, f"{cfg.method.name}.csv"), index=False)


if __name__ == "__main__":
    main()
