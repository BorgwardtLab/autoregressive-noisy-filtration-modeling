import inspect
import time

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse, to_networkx

import wandb
from anfm.data.visualization import plot_filtration, plot_gridspec_graphs
from anfm.models.edge_predictor import cross_entropy

matplotlib.use("agg")


def log_with_rank(rank, message):
    logger.opt(colors=True).info(f"<c>[rank={rank}]</c> {message}")


class AggregatedCallback:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def call(self, method_name, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                fn = getattr(callback, method_name)
                argnames = inspect.signature(fn).parameters.keys()
                fn_kwargs = {name: kwargs[name] for name in argnames}
                fn(**fn_kwargs)
            else:
                continue


class TrainLossLogger:
    def __init__(self, log_interval, fabric, state, cfg):
        self.log_interval = log_interval
        self.state = state
        self.fabric = fabric
        self.cfg = cfg

    def on_batch_end(self, step_idx, epoch_idx, loss, grad_norm, bs):
        if step_idx % self.log_interval == 0 or step_idx == 1:
            if self.fabric.is_global_zero:
                logging_info = {
                    "train_loss": loss,
                    "time/epoch": epoch_idx,
                    "lr": self.state["optim"].param_groups[0]["lr"],
                }
                if grad_norm is not None:
                    logging_info["grad_norm"] = grad_norm
                wandb.log(
                    logging_info,
                    step=step_idx,
                )
            log_with_rank(
                self.fabric.global_rank,
                f"Step {step_idx}/{self.cfg.num_steps} batch size {bs}: loss={loss}",
            )


class ValidationLogger:
    def __init__(
        self,
        val_interval,
        sample_interval,
        val_dataloader,
        val_dataset,
        train_dataset,
        fabric,
        state,
        cfg,
    ):
        self.val_interval = val_interval
        self.sample_interval = sample_interval
        self.val_dataloader = val_dataloader
        self.val_data_iter = iter(val_dataloader)
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.fabric = fabric
        self.state = state
        self.cfg = cfg

    def on_batch_end(self, step_idx):
        model = self.state["model"]
        if step_idx % self.val_interval == 0 or step_idx == 1:
            model.eval()

            try:
                filtrations, targets = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                filtrations, targets = next(self.val_data_iter)

            torch.cuda.empty_cache()
            logger.info(
                f"{self.val_dataset.max_nodes}, {len(filtrations)}, {targets.shape}, {len(self.val_dataset)}"
            )
            with torch.inference_mode(), torch.no_grad():
                mixture_logits, logits, mask = model(
                    filtrations, max_num_nodes=self.val_dataset.max_nodes
                )
                loss = cross_entropy(mixture_logits, logits, mask, targets, n_samples=4)
                loss = self.fabric.all_reduce(loss, reduce_op="mean")
                logger.info(f"Validation, loss={loss.item()}")

            if self.fabric.is_global_zero:
                wandb.log({"val_loss": loss.item()}, step=step_idx)
            model.train()

        if step_idx % self.sample_interval == 0 or step_idx == 1:
            model.eval()

            graph_samples = []
            num_nodes = [
                self.val_dataset.sample_num_nodes()
                for _ in range(self.cfg.wandb.num_samples)
            ]
            logger.info(f"Sampling graphs with node numbers {num_nodes}")
            with torch.inference_mode(), torch.no_grad():
                filtration_samples, graph_samples = model.sample(
                    num_nodes, device=self.cfg.device
                )
            logger.info("Finished sampling, computing metrics")
            metrics = self.val_dataset.evaluate_graphs(
                [to_networkx(g, to_undirected=True) for g in graph_samples],
                val_graphs=self.val_dataset.nx_graphs,
                train_graph_set=self.train_dataset.graph_set,
            )
            metrics = self.fabric.all_reduce(metrics, reduce_op="mean")
            logger.info("Logging sampling metrics")
            if self.fabric.is_global_zero:
                wandb.log(metrics, step=step_idx)

                figures = {}
                true_filtration, pred_targets = self.train_dataset[
                    np.random.randint(0, len(self.train_dataset))
                ]
                edge_index, _ = dense_to_sparse(pred_targets[-1, ..., 1])
                fig_filtration = plot_filtration(
                    Batch.from_data_list(true_filtration), Data(edge_index=edge_index)
                )
                fig_graph, fig_density = plot_gridspec_graphs(
                    [true_filtration],
                    [
                        Data(
                            edge_index=edge_index,
                            num_nodes=true_filtration[0].num_nodes,
                        )
                    ],
                    num_columns=1,
                )
                figures["truth_final"] = wandb.Image(fig_graph)
                figures["truth_filtration"] = wandb.Image(fig_filtration)
                figures["truth_density"] = wandb.Image(fig_density)
                wandb.log(figures, step=step_idx)
                plt.close(fig_density)
                plt.close(fig_graph)
                plt.close(fig_filtration)

                figures = {}

                # We only plot one full filtration from the samples
                fig_sampled_filtration = plot_filtration(
                    Batch.from_data_list(filtration_samples[0]),
                    graph_samples[0],
                )
                # We plot 9 final samples
                fig_sampled_graph, fig_density = plot_gridspec_graphs(
                    filtration_samples[:9], graph_samples[:9], num_columns=3
                )

                figures["sample_final"] = wandb.Image(fig_sampled_graph)
                figures["sample_filtration"] = wandb.Image(fig_sampled_filtration)
                figures["sample_density"] = wandb.Image(fig_density)

                wandb.log(figures, step=step_idx)
                plt.close(fig_sampled_graph)
                plt.close(fig_filtration)
                plt.close(fig_density)
            model.train()

        model.train()


class TimeLogger:
    def __init__(self, log_time_interval, fabric, cfg):
        self.log_time_interval = log_time_interval
        self.last_log_step = None
        self.last_log_time = None
        self.cfg = cfg
        self.fabric = fabric

    def on_batch_end(self, step_idx):
        if not self.fabric.is_global_zero:
            return
        if step_idx % self.log_time_interval == 0 or step_idx == 1:
            if self.last_log_step is not None:
                eta = (self.cfg.num_steps - step_idx) / (
                    (step_idx - self.last_log_step) / (time.time() - self.last_log_time)
                )
                eta /= 3600
                wandb.log({"time/ETA[h]": eta}, step=step_idx)
                logger.info(f"ETA {eta:.2f} hours")
            self.last_log_step = step_idx
            self.last_log_time = time.time()
