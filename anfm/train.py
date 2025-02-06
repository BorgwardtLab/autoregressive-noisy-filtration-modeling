# Import necessary because graph_tool will cause deadlocks
import graph_tool.all as _  # isort: skip

import os
import pickle
import signal
import sys
import time
from functools import partial

import hydra
import numpy as np
import omegaconf
import torch
from lightning.fabric import Fabric
from loguru import logger
from torch.utils.data import DataLoader

import wandb
from anfm import CODE_DIR
from anfm.data.base.collate import filtration_collate_fn
from anfm.data.base.dataset import AbstractFiltrationDataset
from anfm.logging import (
    AggregatedCallback,
    TimeLogger,
    TrainLossLogger,
    ValidationLogger,
    log_with_rank,
)
from anfm.models.edge_predictor import cross_entropy
from anfm.utils import (
    ensure_reproducibility,
    get_last_ckpt_path,
    get_meta_info,
    save_meta_info,
)

interrupted = False


def interrupt_handler(signum, frame):
    global interrupted
    interrupted = True
    logger.info("Received interrupt signal...")


def train(cfg):
    torch.set_float32_matmul_precision("medium")

    # warnings.showwarning = custom_warning_filter

    # Register the signal handler
    signal.signal(signal.SIGUSR1, interrupt_handler)

    if not torch.cuda.is_available():
        logger.warning("Found no GPU")

    fabric = Fabric(
        accelerator=cfg.device,
        devices=cfg.num_devices,
        precision=cfg.precision,
        strategy=cfg.strategy,
    )
    fabric.launch()
    fabric.barrier()
    start_time = time.time()

    log_info = partial(log_with_rank, fabric.global_rank)
    log_info(f"Using {fabric.strategy} strategy with {fabric.world_size} devices")

    if fabric.is_global_zero:
        log_info(f"Starting training with config: {cfg}")

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    has_timelimit = cfg.timelimit is not None
    run_dir_set = any("hydra.run.dir" in arg for arg in sys.argv)
    if has_timelimit and not run_dir_set:
        raise ValueError("The run has a timelimit but no fixed output directory")
    if has_timelimit:
        log_info(
            f"Run has timelimit of {cfg.timelimit} hours and is restartable with same command"
        )

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    meta_info = get_meta_info(outdir)
    checkpoint_dir = os.path.join(outdir, "checkpoints")
    fabric.barrier()

    if fabric.is_global_zero:
        if meta_info["runid"] is None:
            log_info("No previous wandb run found")

        wandb.login()
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=cfg_dict,
            mode="disabled" if not cfg.wandb.enabled else None,
            dir=outdir,
            id=meta_info["runid"],
            resume="allow",
            entity=cfg.wandb.entity,
        )
        log_info(f"Already had {meta_info['restarts']} restarts")
        meta_info["restarts"] += 1
        meta_info["runid"] = run.id
        save_meta_info(outdir, meta_info)
        run.log_code(str(CODE_DIR))
        ensure_reproducibility(cfg, outdir=outdir, restart_idx=meta_info["restarts"])
        os.makedirs(checkpoint_dir, exist_ok=True)

    fabric.barrier()

    torch.manual_seed(
        cfg.seed + 42 * meta_info["restarts"] - 1 + 1000 * fabric.global_rank
    )
    np.random.seed(
        cfg.seed + 42 * meta_info["restarts"] - 1 + 1000 * fabric.global_rank
    )

    log_info("Setting up dataset...")

    filtration_kwargs = omegaconf.OmegaConf.to_container(
        cfg.filtration, resolve=True, throw_on_missing=True
    )
    dataset_cfg = omegaconf.OmegaConf.to_container(
        cfg.dataset, resolve=True, throw_on_missing=True
    )
    if "val_overrides" in dataset_cfg:
        val_overrides = dataset_cfg["val_overrides"]
        del dataset_cfg["val_overrides"]
    else:
        val_overrides = {}
    assert "transform" not in dataset_cfg

    t0 = time.time()
    dataset: AbstractFiltrationDataset = hydra.utils.instantiate(
        dataset_cfg,
        split="train",
        filtration_kwargs=filtration_kwargs,
        _convert_="partial",
    )
    dataset_cfg.update(val_overrides)
    val_dataset: AbstractFiltrationDataset = hydra.utils.instantiate(
        dataset_cfg,
        split="val",
        filtration_kwargs=filtration_kwargs,
        hold_out=dataset.graph_set,
        _convert_="partial",
    )
    test_dataset: AbstractFiltrationDataset = hydra.utils.instantiate(
        dataset_cfg,
        split="test",
        filtration_kwargs=filtration_kwargs,
        hold_out=dataset.graph_set + val_dataset.graph_set,
        _convert_="partial",
    )

    if fabric.is_global_zero:
        with open(os.path.join(outdir, "train_set.pkl"), "wb") as f:
            pickle.dump(dataset.graph_set.nx_graphs, f)
        with open(os.path.join(outdir, "val_set.pkl"), "wb") as f:
            pickle.dump(val_dataset.graph_set.nx_graphs, f)
        with open(os.path.join(outdir, "test_set.pkl"), "wb") as f:
            pickle.dump(test_dataset.graph_set.nx_graphs, f)

    collate_fn = partial(
        filtration_collate_fn,
        node_ordering_noise=cfg.dataloading.node_ordering_noise,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.hyper_parameters.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=cfg.dataloading.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.hyper_parameters.val_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=True,
        num_workers=cfg.dataloading.num_workers,
    )
    t1 = time.time()
    log_info(f"Dataset preparation took {(t1 - t0) / 60:.2f} minutes")

    dataloader = fabric.setup_dataloaders(dataloader)
    data_iter = iter(dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    log_info("Setting up model...")
    model = hydra.utils.instantiate(
        cfg.model,
        num_edge_types=2,  # Edge or no edge
        max_nodes=dataset.max_nodes,
        _recursive_=False,
    )
    model.train()
    log_info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.hyper_parameters.learning_rate)

    if cfg.compile_model:
        model.forward = torch.compile(model.forward)

    model, optim = fabric.setup(model, optim)
    model.mark_forward_method("sample")

    decay_iterations = cfg.num_steps - cfg.hyper_parameters.warmup_steps
    if cfg.hyper_parameters.decay_mode == "exponential":
        gamma = cfg.hyper_parameters.last_lr_factor ** (1 / decay_iterations)
        decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    elif cfg.hyper_parameters.decay_mode == "linear":
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optim,
            1.0,
            cfg.hyper_parameters.last_lr_factor,
            total_iters=decay_iterations,
        )
    elif cfg.hyper_parameters.decay_mode == "constant":
        decay_scheduler = torch.optim.lr_scheduler.ConstantLR(optim, 1)
    else:
        raise NotImplementedError
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        [
            torch.optim.lr_scheduler.LinearLR(
                optim, 0.001, 1.0, total_iters=cfg.hyper_parameters.warmup_steps
            ),
            decay_scheduler,
        ],
        milestones=[cfg.hyper_parameters.warmup_steps],
    )

    state = {"model": model, "optim": optim, "step": 0, "lr_scheduler": lr_scheduler}
    last_ckpt_path = get_last_ckpt_path(checkpoint_dir)
    if last_ckpt_path is not None:
        fabric.load(last_ckpt_path, state)
    step_idx = state["step"]

    callback = AggregatedCallback(
        [
            TrainLossLogger(
                log_interval=cfg.wandb.log_interval, fabric=fabric, state=state, cfg=cfg
            ),
            ValidationLogger(
                val_interval=cfg.wandb.val_interval,
                sample_interval=cfg.wandb.sample_interval,
                val_dataloader=val_dataloader,
                val_dataset=val_dataset,
                train_dataset=dataset,
                fabric=fabric,
                state=state,
                cfg=cfg,
            ),
            TimeLogger(cfg.wandb.log_time_interval, fabric=fabric, cfg=cfg),
        ]
    )

    epoch_idx = 0

    grad_acc = cfg.hyper_parameters.gradient_accumulation

    optim.zero_grad()
    backward_count = 0
    timelimit_reached = False
    while step_idx < cfg.num_steps:
        try:
            filtrations, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            filtrations, targets = next(data_iter)
            epoch_idx += 1

        is_accumulating = backward_count % grad_acc != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            mixture_logits, logits, mask = model(filtrations)
            bs = mask.size(1)
            loss = cross_entropy(mixture_logits, logits, mask, targets, **cfg.loss)
            fabric.backward(loss)  # , model=model)
            loss = loss.item()

        backward_count += 1

        if not is_accumulating:
            if step_idx % 250 == 0:
                # Log gradient magnitude
                flat_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
                flat_grad = np.abs(flat_grad.detach().cpu().numpy())
                grad_magnitude = np.linalg.norm(flat_grad)
                quantile_levels = np.array([0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
                quantiles = np.quantile(np.abs(flat_grad), quantile_levels).tolist()
                max_value = np.max(np.abs(flat_grad))
                log_info(
                    f"Gradient magnitude {grad_magnitude:.2f}, max value {max_value}, quantiles at {quantile_levels.tolist()}: {quantiles}"
                )
            if (
                hasattr(cfg.hyper_parameters, "clip_val")
                and cfg.hyper_parameters.clip_val is not None
            ):
                fabric.clip_gradients(
                    model, optim, clip_val=cfg.hyper_parameters.clip_grad
                )
                assert not hasattr(cfg.hyper_parameters, "clip_norm")
                grad_norm = None
            elif (
                hasattr(cfg.hyper_parameters, "clip_norm")
                and cfg.hyper_parameters.clip_norm is not None
            ):
                grad_norm = fabric.clip_gradients(
                    model, optim, max_norm=cfg.hyper_parameters.clip_norm
                ).item()
                assert not hasattr(cfg.hyper_parameters, "clip_val")
            else:
                grad_norm = None
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
            step_idx += 1
            callback.call(
                "on_batch_end",
                step_idx=step_idx,
                loss=loss,
                grad_norm=grad_norm,
                epoch_idx=epoch_idx,
                bs=bs,
            )

        if step_idx % 100 == 0:
            timelimit_reached = interrupted or (
                has_timelimit and (time.time() - start_time) / 3600 >= cfg.timelimit
            )
            timelimit_reached = bool(fabric.broadcast(torch.tensor(timelimit_reached)))

        if step_idx % cfg.checkpoint_steps == 0 or timelimit_reached:
            state["step"] = step_idx
            fabric.save(
                os.path.join(checkpoint_dir, f"step_{step_idx}.ckpt"),
                state,
            )
        if timelimit_reached:
            break

    if step_idx >= cfg.num_steps:
        state["step"] = step_idx
        fabric.save(
            os.path.join(checkpoint_dir, "final.ckpt"),
            state,
        )

    dataset.close()
    val_dataset.close()
    test_dataset.close()

    if step_idx < cfg.num_steps and timelimit_reached:
        log_info("Interrupting due to timeout, signaling for restart...")
        return 124 if fabric.is_global_zero else 0
    return 0


@hydra.main(config_path="./configs/teacher_forcing", version_base=None)
def main(cfg):
    logger.info(f"Starting training with config: {cfg}")
    exit_code = train(cfg)
    logger.info(f"Exiting with code {exit_code}...")
    time.sleep(2)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
