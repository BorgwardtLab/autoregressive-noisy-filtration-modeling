import graph_tool.all as gt  # isort: skip
import os
import sys
import time
from itertools import chain, islice
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
import torch_geometric.transforms as T
from einops import rearrange
from lightning import Fabric
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx, to_dense_adj, to_networkx
from tqdm import tqdm

import anfm
import wandb
from anfm.data.base.collate import filtration_collate_fn
from anfm.data.base.features import KNodeCycles, global_laplacian
from anfm.data.visualization import plot_gridspec_graphs
from anfm.models.edge_predictor import (
    cross_entropy,
    entropy_lower_bound,
    entropy_upper_bound,
)
from anfm.utils import (
    ensure_reproducibility,
    get_last_ckpt_path,
    get_meta_info,
    save_meta_info,
)


class FiltrationToGraphDataset(Dataset):
    def __init__(self, filtration_dataset, featurizer, subset_size=None):
        super().__init__()
        self.graphs = []
        for g in tqdm(filtration_dataset.nx_graphs):
            graph = from_networkx(g)
            if hasattr(graph, "label"):
                delattr(graph, "label")
            if hasattr(graph, "weight"):
                delattr(graph, "weight")
            self.graphs.append(graph)
        if subset_size is not None:
            self.graphs = self.graphs[:subset_size]
            self.nx_graphs = filtration_dataset.nx_graphs[:subset_size]
        else:
            self.nx_graphs = filtration_dataset.nx_graphs
        self.graphs = featurizer.add_features(self.graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class RewardWhitener(nn.Module):
    def __init__(self, momentum=0.025, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.register_buffer("has_trained", torch.tensor(False))
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("var", torch.tensor(1.0))

    def forward(self, rewards):
        assert rewards.ndim == 1
        var, mean = torch.var_mean(rewards)
        if self.has_trained:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * mean
            self.var = (1 - self.momentum) * self.var + self.momentum * var
        else:
            self.mean = mean
            self.var = var
            self.has_trained = torch.tensor(True, device=self.has_trained.device)
        return (rewards - self.mean) / torch.sqrt(self.var + self.epsilon)


class NodeFeaturizer:
    def __init__(
        self, random_walk_pe_dim=20, num_laplacian_eigenvecs=4, cycle_counts=False
    ):
        self.feature_dims = {}
        if random_walk_pe_dim > 0:
            self.random_walk_transform = T.AddRandomWalkPE(
                walk_length=random_walk_pe_dim, attr_name="random_walk_pe"
            )
            self.feature_dims["random_walk_pe"] = random_walk_pe_dim
        else:
            self.random_walk_transform = None
        self.num_laplacian_eigenvecs = num_laplacian_eigenvecs
        if self.num_laplacian_eigenvecs > 0:
            self.feature_dims["eig_vecs"] = self.num_laplacian_eigenvecs
            self.feature_dims["eig_vals"] = self.num_laplacian_eigenvecs
        if cycle_counts:
            self.cycle_transform = KNodeCycles()
            self.feature_dims["local_cycle_counts"] = 3
            self.feature_dims["global_cycle_counts"] = 4
        else:
            self.cycle_transform = None

    def add_features(self, graph_list):
        assert isinstance(graph_list, list)
        max_num_nodes = max([g.num_nodes for g in graph_list])
        batch = Batch.from_data_list(graph_list)
        assert batch.num_graphs == len(graph_list)

        if self.num_laplacian_eigenvecs > 0:
            eig_vecs, eig_vals = global_laplacian(
                batch.edge_index,
                max_num_nodes=max_num_nodes,
                k=self.num_laplacian_eigenvecs,
                batch=batch.batch,
                fast=batch.num_graphs % 16 == 0,
                return_dense=True,
            )
            assert eig_vecs.shape == (
                batch.num_graphs,
                max_num_nodes,
                self.num_laplacian_eigenvecs,
            )
            assert eig_vals.shape == (
                batch.num_graphs,
                max_num_nodes,
                self.num_laplacian_eigenvecs,
            )
        else:
            eig_vecs, eig_vals = None, None

        if self.cycle_transform is not None:
            dense_adj = to_dense_adj(batch.edge_index, batch.batch)
            local_counts, global_counts = self.cycle_transform.k_cycles(dense_adj)
            assert local_counts.ndim == 3 and local_counts.shape[:2] == (
                batch.num_graphs,
                max_num_nodes,
            )
            assert global_counts.ndim == 2 and global_counts.size(0) == batch.num_graphs
        else:
            local_counts, global_counts = None, None

        for i, g in enumerate(graph_list):
            if self.random_walk_transform is not None:
                graph_list[i] = self.random_walk_transform(g)
            else:
                graph_list[i].random_walk_pe = None
            if eig_vecs is not None:
                graph_list[i].eig_vecs = eig_vecs[i][: g.num_nodes]
                graph_list[i].eig_vals = eig_vals[i][: g.num_nodes]
            else:
                graph_list[i].eig_vecs = torch.zeros(g.num_nodes, 0)
                graph_list[i].eig_vals = torch.zeros(g.num_nodes, 0)
            if local_counts is not None:
                graph_list[i].local_cycle_counts = local_counts[i][: g.num_nodes]
                graph_list[i].global_cycle_counts = (
                    global_counts[i].unsqueeze(0).expand(max_num_nodes, -1)
                )
            else:
                graph_list[i].local_cycle_counts = torch.zeros(g.num_nodes, 0)
                graph_list[i].global_cycle_counts = torch.zeros(g.num_nodes, 0)

        return graph_list


def get_gradient_magnitudes(model, quantile_levels=(0.01, 0.1, 0.5, 0.9, 0.99, 0.999)):
    flat_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    flat_grad = np.abs(flat_grad.detach().cpu().numpy())
    grad_magnitude = np.linalg.norm(flat_grad)
    quantile_levels = np.array(quantile_levels)
    quantiles = np.quantile(np.abs(flat_grad), quantile_levels).tolist()
    max_value = np.max(np.abs(flat_grad))
    return {
        "grad_magnitude": grad_magnitude,
        "max_value": max_value,
        "quantiles": quantiles,
    }


def labeled_graph_collate(samples):
    data_list, label_list = (*zip(*samples),)
    return Batch.from_data_list(data_list), torch.Tensor(label_list)


@torch.inference_mode()
def evaluate_discriminator(fabric, discriminator, positive_batch, negative_batch):
    is_training = discriminator.training
    discriminator.eval()
    positive_outputs = discriminator(positive_batch)
    negative_outputs = discriminator(negative_batch)
    discriminator.train(is_training)

    metrics = {}
    labels = np.concatenate(
        [np.ones(len(positive_outputs)), np.zeros(len(negative_outputs))]
    )
    outputs = np.concatenate(
        [
            positive_outputs.squeeze(-1).cpu().detach().numpy(),
            negative_outputs.squeeze(-1).cpu().detach().numpy(),
        ]
    )
    metrics["AUROC"] = roc_auc_score(labels, outputs)
    metrics["CE"] = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(outputs), torch.from_numpy(labels), reduction="mean"
    ).item()
    metrics["accuracy"] = (
        ((positive_outputs > 0).sum() + (negative_outputs < 0).sum())
        / (len(positive_outputs) + len(negative_outputs))
    ).item()
    metrics["TNR"] = (negative_outputs < 0).sum().item() / len(negative_outputs)
    metrics["TPR"] = (positive_outputs > 0).sum().item() / len(positive_outputs)
    metrics = fabric.all_reduce(metrics, reduce_op="mean")
    return metrics


def run_discriminator_training(
    fabric,
    discriminator,
    optim,
    positive_examples,
    negative_examples,
    batch_size,
    num_steps,
    clamp_range=None,
    log_prefix="",
):
    """Given positive and negative examples, train the discriminator for num_steps steps.

    Args:
        discriminator (nn.Module): The discriminator model.
        positive_examples (torch.utils.data.Dataset): Samples from true data distribution.
        negative_examples (torch.utils.data.Dataset): Samples from generator.
        batch_size (int): The batch size.
        num_steps (int): The number of steps to train the discriminator.
    """
    # Combine positive and negative examples into a single dataset with labels
    discriminator.train()
    positive_labels = torch.ones(len(positive_examples))
    negative_labels = torch.zeros(len(negative_examples))
    assert len(positive_labels) == len(negative_labels)
    positive_examples = torch.utils.data.StackDataset(
        positive_examples, positive_labels
    )

    negative_examples = torch.utils.data.StackDataset(
        [
            Data(
                edge_index=g.edge_index,
                random_walk_pe=g.random_walk_pe,
                eig_vecs=g.eig_vecs,
                eig_vals=g.eig_vals,
                local_cycle_counts=g.local_cycle_counts,
                global_cycle_counts=g.global_cycle_counts,
                num_nodes=g.num_nodes,
            ).cpu()
            for g in negative_examples
        ],
        negative_labels,
    )
    combined_dataset = torch.utils.data.ConcatDataset(
        [positive_examples, negative_examples]
    )
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=labeled_graph_collate,
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    data_iter = iter(dataloader)

    for step_idx in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        inputs, labels = batch
        optim.zero_grad()
        outputs = discriminator(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs[:, 0], labels
        )
        logger.opt(colors=True).info(
            f"{log_prefix} Substep {step_idx}, loss: {loss.item()}"
        )
        fabric.backward(loss)
        optim.step()
        if clamp_range is not None:
            for p in discriminator.parameters():
                p.data.clamp_(-clamp_range, clamp_range)


def train_and_validate_discriminator(
    fabric,
    generator,
    discriminator,
    discriminator_optim,
    filtration_train_set,
    graph_train_set,
    graph_val_set,
    num_train_samples,
    num_val_samples,
    discriminator_batch_size,
    num_discriminator_steps,
    featurizer,
    device,
    validate=True,
    clamp_range=None,
    generation_batchsize=512,
    log_prefix="",
):
    # Create positive and negative samples for discriminator training
    positive_train_examples = torch.utils.data.Subset(
        graph_train_set,
        np.random.choice(
            len(graph_train_set),
            size=num_train_samples,
            replace=False,
        ),
    )
    if validate:
        positive_val_examples = torch.utils.data.Subset(
            graph_val_set,
            np.random.choice(
                len(graph_val_set),
                size=num_val_samples,
                replace=False,
            ),
        )
    else:
        positive_val_examples = None
    logger.opt(colors=True).info(
        f"{log_prefix} Sampling {num_train_samples}+{num_val_samples} graphs from generative model..."
    )
    t0 = time.time()
    _, negative_train_examples = generate_samples(
        generator,
        filtration_train_set,
        num_train_samples,
        device,
        batch_size=generation_batchsize,
    )
    num_sampled = len(negative_train_examples)
    negative_train_examples = featurizer.add_features(negative_train_examples)
    if validate:
        _, negative_val_examples = generate_samples(
            generator,
            filtration_train_set,
            num_val_samples,
            device,
            batch_size=generation_batchsize,
        )
        num_sampled += len(negative_val_examples)
        negative_val_examples = featurizer.add_features(negative_val_examples)
    else:
        negative_val_examples = None

    logger.opt(colors=True).info(
        f"{log_prefix} Sampling {num_sampled} graphs took {(time.time() - t0) / 60:.2f} minutes"
    )

    metrics = {}
    negative_train_batch = Batch.from_data_list(
        negative_train_examples[: min(num_val_samples, num_train_samples)]
    )
    positive_train_batch = Batch.from_data_list(
        [
            positive_train_examples[i]
            for i in range(min(num_val_samples, num_train_samples))
        ]
    ).to(device)
    train_metrics = evaluate_discriminator(
        fabric, discriminator, positive_train_batch, negative_train_batch
    )
    logger.opt(colors=True).info(
        f"{log_prefix} Discriminator metrics before training on train set: {train_metrics}"
    )
    metrics.update(
        {f"discriminator/train_before/{k}": v for k, v in train_metrics.items()}
    )

    if validate:
        negative_val_batch = Batch.from_data_list(negative_val_examples)
        positive_val_batch = Batch.from_data_list(
            [positive_val_examples[i] for i in range(num_val_samples)]
        ).to(device)
        val_metrics = evaluate_discriminator(
            fabric, discriminator, positive_val_batch, negative_val_batch
        )
        logger.opt(colors=True).info(
            f"{log_prefix} Discriminator metrics before training on val set: {val_metrics}"
        )
        metrics.update(
            {f"discriminator/val_before/{k}": v for k, v in val_metrics.items()}
        )

        logger.opt(colors=True).info(
            f"{log_prefix} Computing metrics by comparing {len(negative_val_examples)} generated graphs to {len(graph_val_set.nx_graphs)} ground-truth graphs..."
        )
        sampling_metrics = filtration_train_set.evaluate_graphs(
            [to_networkx(g, to_undirected=True) for g in negative_val_examples],
            val_graphs=graph_val_set.nx_graphs,
            train_graph_set=filtration_train_set.graph_set,
        )
        sampling_metrics = fabric.all_reduce(sampling_metrics, reduce_op="mean")
        metrics.update({f"generator/{k}": v for k, v in sampling_metrics.items()})
        logger.opt(colors=True).info(
            f"{log_prefix} <r>Generator</r> metrics: {metrics}"
        )

    logger.opt(colors=True).info(
        f"{log_prefix} Training discriminator for {num_discriminator_steps} steps..."
    )
    if num_discriminator_steps > 0:
        run_discriminator_training(
            fabric,
            discriminator,
            discriminator_optim,
            positive_train_examples,
            negative_train_examples,
            discriminator_batch_size,
            num_discriminator_steps,
            log_prefix=log_prefix,
            clamp_range=clamp_range,
        )
    logger.opt(colors=True).info(f"{log_prefix} Discriminator training step complete.")

    train_metrics = evaluate_discriminator(
        fabric, discriminator, positive_train_batch, negative_train_batch
    )
    metrics.update(
        {f"discriminator/train_after/{k}": v for k, v in train_metrics.items()}
    )
    logger.opt(colors=True).info(
        f"{log_prefix} Discriminator metrics on train set after training: {train_metrics}"
    )

    if validate:
        val_metrics = evaluate_discriminator(
            fabric, discriminator, positive_val_batch, negative_val_batch
        )
        metrics.update(
            {f"discriminator/val_after/{k}": v for k, v in val_metrics.items()}
        )
        logger.opt(colors=True).info(
            f"{log_prefix} Discriminator metrics on val set after training: {val_metrics}"
        )

    return metrics


def train_value_model(
    fabric,
    value_model,
    value_model_optim,
    filtrations,
    rewards,
    num_steps,
    batch_size,
    num_accumulation_steps,
    grad_clip_value=None,
    log_prefix="",
):
    value_model.train()
    assert len(rewards) == len(filtrations)
    filtration_size = len(filtrations[0])

    quantile_levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    reward_quantiles = np.quantile(rewards.cpu().numpy(), quantile_levels).tolist()
    logger.opt(colors=True).info(
        f"{log_prefix} Whitened reward quantiles at levels {quantile_levels}: {reward_quantiles}"
    )
    logger.opt(colors=True).info(
        f"{log_prefix} Whitened reward min, mean, max: {rewards.min().item()}, {rewards.mean().item()}, {rewards.max().item()}"
    )

    average_loss = 0
    current_index = 0
    for substep_idx in range(num_steps):
        value_model_optim.zero_grad()
        substep_avg_loss = 0
        for acc_step in range(num_accumulation_steps):
            with fabric.no_backward_sync(
                value_model, enabled=(acc_step != num_accumulation_steps - 1)
            ):
                if current_index + batch_size > len(filtrations):
                    current_index = 0
                filtration_batch = Batch.from_data_list(
                    list(
                        chain(
                            *zip(
                                *filtrations[current_index : current_index + batch_size]
                            )
                        )
                    )
                )
                reward_batch = rewards[current_index : current_index + batch_size]
                current_index += batch_size
                assert len(reward_batch) == batch_size
                reward_to_go = reward_batch.unsqueeze(0).expand(filtration_size, -1)
                predicted_rewards = value_model(filtration_batch)
                assert predicted_rewards.shape == (filtration_batch.num_graphs, 1)
                predicted_rewards = rearrange(
                    predicted_rewards.squeeze(-1), "(t b) -> t b", b=batch_size
                )
                assert predicted_rewards.shape == reward_to_go.shape
                if torch.isnan(predicted_rewards).any():
                    logger.opt(colors=True).error(
                        f"{log_prefix} NaN in predicted rewards of value model, masking those values"
                    )
                    predicted_rewards = torch.where(
                        torch.isnan(predicted_rewards),
                        torch.zeros_like(predicted_rewards),
                        predicted_rewards,
                    )
                    reward_to_go = torch.where(
                        torch.isnan(predicted_rewards),
                        torch.zeros_like(predicted_rewards),
                        reward_to_go,
                    )
                if torch.isnan(reward_to_go).any():
                    logger.opt(colors=True).error(
                        f"{log_prefix} NaN in reward to go, masking those values"
                    )
                    predicted_rewards = torch.where(
                        torch.isnan(reward_to_go),
                        torch.zeros_like(reward_to_go),
                        predicted_rewards,
                    )
                    reward_to_go = torch.where(
                        torch.isnan(reward_to_go),
                        torch.zeros_like(reward_to_go),
                        reward_to_go,
                    )
                loss = torch.nn.functional.mse_loss(predicted_rewards, reward_to_go)
                substep_avg_loss += loss.item()
                fabric.backward(loss)
        substep_avg_loss /= num_accumulation_steps
        average_loss += substep_avg_loss
        logger.opt(colors=True).info(
            f"{log_prefix} Value substep {substep_idx} loss: {substep_avg_loss}"
        )
        if grad_clip_value is not None:
            quantile_levels = (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
            grad_info = get_gradient_magnitudes(
                value_model, quantile_levels=quantile_levels
            )
            logger.opt(colors=True).info(
                f"{log_prefix} Value model gradient magnitude {grad_info['grad_magnitude']:.2f}, max value {grad_info['max_value']}, quantiles at {quantile_levels}: {grad_info['quantiles']}"
            )
            fabric.clip_gradients(
                value_model,
                value_model_optim,
                clip_val=grad_clip_value,
                error_if_nonfinite=True,
            )

        value_model_optim.step()
    return {"value_model/average_loss": average_loss / (num_steps)}


def train_policy_ppo(
    fabric: Fabric,
    generator,
    discriminator,
    value_model,
    reward_whitener,
    generator_optim,
    value_model_optim,
    train_filtration_set,
    num_steps,
    num_epochs,
    generator_batch_size,
    generator_grad_accumulation,
    value_model_num_steps,
    value_model_batch_size,
    value_model_grad_accumulation,
    featurizer,
    device,
    epsilon=0.2,
    lower_reward_clip=None,
    generation_batchsize=512,
    generator_grad_clip_value=None,
    value_model_grad_clip_value=None,
    log_prefix="",
):
    """Use PPO algorithm to optimize the generator."""
    discriminator.eval()

    metrics = {}
    all_average_rewards = []
    all_average_entropy_lb = []
    all_average_entropy_ub = []

    for substep_idx in range(num_steps):
        num_samples = generator_batch_size * generator_grad_accumulation
        logger.opt(colors=True).info(
            f"{log_prefix} Substep {substep_idx}, generating {num_samples} samples..."
        )
        all_filtrations, all_samples = generate_samples(
            generator,
            train_filtration_set,
            num_samples,
            device,
            batch_size=generation_batchsize,
        )
        if substep_idx == 0:
            fig_sampled_graph, fig_density = plot_gridspec_graphs(
                all_filtrations[:9], all_samples[:9], num_columns=3
            )
            metrics["generator/samples"] = wandb.Image(fig_sampled_graph)
            metrics["generator/densities"] = wandb.Image(fig_density)

        filtration_size = len(all_filtrations[0])
        all_targets = filtrations_to_targets(all_filtrations, all_samples)
        all_rewards = grade_samples(discriminator, all_samples, featurizer).clone()
        logger.opt(colors=True).info(
            f"{log_prefix} Raw reward min, mean, max: {all_rewards.min().item()}, {all_rewards.mean().item()}, {all_rewards.max().item()}"
        )
        if lower_reward_clip is not None:
            all_rewards = torch.clamp(all_rewards, min=lower_reward_clip)
        average_reward = all_rewards.mean().item()
        all_average_rewards.append(average_reward)
        all_rewards = reward_whitener(all_rewards)

        if value_model is not None:
            value_model.eval()
            logger.opt(colors=True).info(f"{log_prefix} Computing baselined rewards...")
            with torch.inference_mode():
                all_filtrations_batch = Batch.from_data_list(
                    list(chain(*zip(*all_filtrations)))
                )
                state_values = value_model(all_filtrations_batch)
                assert state_values.shape == (all_filtrations_batch.num_graphs, 1)
                state_values = state_values.squeeze(-1)
                assert all_rewards.shape == (num_samples,)
                all_baselined_rtg = all_rewards.unsqueeze(0).expand(
                    filtration_size, -1
                ) - rearrange(state_values, "(t b) -> t b", b=num_samples)
                assert all_baselined_rtg.shape == (filtration_size, num_samples)
            all_baselined_rtg = all_baselined_rtg.clone()
            assert not torch.isnan(all_baselined_rtg).any()
            value_model.train()
            logger.opt(colors=True).info(f"{log_prefix} Training value model...")
            value_model_metrics = train_value_model(
                fabric,
                value_model,
                value_model_optim,
                all_filtrations,
                all_rewards,
                value_model_num_steps,
                value_model_batch_size,
                value_model_grad_accumulation,
                grad_clip_value=value_model_grad_clip_value,
                log_prefix=log_prefix,
            )
            metrics.update(value_model_metrics)
        else:
            all_rewards = (all_rewards - torch.mean(all_rewards)) / torch.std(
                all_rewards
            )
            all_baselined_rtg = all_rewards.unsqueeze(0).expand(filtration_size, -1)

        # Important: We need to switch the generator to eval mode to remove stochasticity
        generator.eval()

        nll_begin = []
        for epoch in range(num_epochs):
            generator_optim.zero_grad()

            for j in range(generator_grad_accumulation):
                with fabric.no_backward_sync(
                    generator, enabled=(j != generator_grad_accumulation - 1)
                ):
                    filtrations = all_filtrations[
                        j * generator_batch_size : (j + 1) * generator_batch_size
                    ]
                    baselined_rtg = all_baselined_rtg[
                        :, j * generator_batch_size : (j + 1) * generator_batch_size
                    ]
                    targets = all_targets[
                        j * generator_batch_size : (j + 1) * generator_batch_size
                    ]  # Now, we have to pack everything into batches
                    assert len(filtrations) == len(targets)
                    assert len(filtrations[0]) == len(targets[0]), (
                        targets.shape,
                        all_targets.shape,
                    )
                    filtrations, targets = filtration_collate_fn(
                        batch=list(zip(filtrations, targets)),
                        node_ordering_noise=0,
                        compute_ordering=False,
                    )
                    mixture_logits, logits, mask = generator(
                        filtrations,
                        max_num_nodes=all_targets[0].size(-2),
                    )
                    nll = cross_entropy(
                        mixture_logits,
                        logits,
                        mask,
                        targets,
                        reduce_batch=False,
                        reduce_filtration=False,
                        hard_labels=True,
                    )
                    assert nll.shape == baselined_rtg.shape
                    if epoch == 0:
                        nll_begin.append(nll.detach())
                    log_ratio = nll_begin[j] - nll
                    ratio = torch.exp(log_ratio)
                    logger.opt(colors=True).info(
                        f"{log_prefix} Substep {substep_idx}, epoch {epoch}, ratio: {ratio.mean().item()}, baselined rtg: {baselined_rtg.mean().item()}"
                    )

                    pg_loss1 = -ratio * baselined_rtg
                    pg_loss2 = (
                        -torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * baselined_rtg
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    entropy_lb = entropy_lower_bound(mixture_logits, logits, mask)
                    entropy_ub = entropy_upper_bound(mixture_logits, logits, mask)
                    assert entropy_lb.ndim == 2 and entropy_ub.ndim == 2
                    all_average_entropy_lb.append(entropy_lb.mean().item())
                    all_average_entropy_ub.append(entropy_ub.mean().item())

                    logger.opt(colors=True).info(
                        f"{log_prefix} Substep {substep_idx} epoch {epoch}, loss: {pg_loss.item()}, ratio: {ratio.mean().item()}"
                    )
                    fabric.backward(pg_loss)

            if substep_idx == 0 and epoch == 0:
                # Log gradient magnitude
                quantile_levels = (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
                grad_info = get_gradient_magnitudes(
                    generator, quantile_levels=quantile_levels
                )
                logger.opt(colors=True).info(
                    f"{log_prefix} Gradient magnitude {grad_info['grad_magnitude']:.2f}, max value {grad_info['max_value']}, quantiles at {quantile_levels}: {grad_info['quantiles']}"
                )

            if generator_grad_clip_value is not None:
                fabric.clip_gradients(
                    generator,
                    generator_optim,
                    clip_val=generator_grad_clip_value,
                    error_if_nonfinite=True,
                )

            generator_optim.step()

    metrics["generator/average_reward"] = np.mean(all_average_rewards)
    metrics["generator/average_entropy_lb"] = np.mean(all_average_entropy_lb)
    metrics["generator/average_entropy_ub"] = np.mean(all_average_entropy_ub)

    return metrics


def filtrations_to_targets(filtrations, samples, max_num_nodes=None, num_edge_types=2):
    """Given a list of filtrations and samples, return a batch of target tensors."""
    max_num_nodes = (
        max(g.num_nodes for g in samples) if max_num_nodes is None else max_num_nodes
    )
    targets = [[] for _ in samples]
    for i, filtration in enumerate(filtrations):
        for subgraph in islice(filtration, 1, None):
            adj = (
                to_dense_adj(subgraph.edge_index, max_num_nodes=max_num_nodes)
                .squeeze(0)
                .long()
            )
            onehot_adj = torch.nn.functional.one_hot(adj, num_classes=num_edge_types)
            targets[i].append(onehot_adj)
        adj = (
            to_dense_adj(samples[i].edge_index, max_num_nodes=max_num_nodes)
            .squeeze(0)
            .long()
        )
        onehot_adj = torch.nn.functional.one_hot(adj, num_classes=num_edge_types)
        targets[i].append(onehot_adj)
    return [torch.stack(target, dim=0) for target in targets]


@torch.inference_mode()
def generate_samples(generator, train_dataset, num_samples, device, batch_size=512):
    is_training = generator.training

    # We sample the number of nodes, but sort it to reduce the amount of padding
    all_num_nodes = np.array(
        [train_dataset.sample_num_nodes() for _ in range(num_samples)]
    )
    sorting_permutation = np.argsort(all_num_nodes)
    inverse_permutation = np.argsort(sorting_permutation)
    all_num_nodes = all_num_nodes[sorting_permutation]

    all_filtrations, all_samples = [], []
    generator.eval()
    num_generated = 0
    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)

        num_nodes = all_num_nodes[num_generated : num_generated + bs].tolist()
        filtrations, samples = generator.sample(num_nodes, device=device)
        all_filtrations.extend(filtrations)
        all_samples.extend(samples)
        num_generated += bs

    all_samples = [all_samples[i] for i in inverse_permutation]
    all_filtrations = [all_filtrations[i] for i in inverse_permutation]
    generator.train(is_training)
    return all_filtrations, all_samples


@torch.inference_mode()
def grade_samples(discriminator, samples, featurizer, batch_size=128):
    samples = featurizer.add_features(samples)
    is_training = discriminator.training
    discriminator.eval()

    num_graded = 0
    all_results = []
    while num_graded < len(samples):
        bs = min(batch_size, len(samples) - num_graded)
        batch = Batch.from_data_list(samples[num_graded : num_graded + bs])
        result = discriminator(batch)
        result = torch.nn.functional.logsigmoid(result).squeeze(-1).detach()
        all_results.append(result)
        num_graded += bs
    discriminator.train(is_training)
    return torch.cat(all_results, dim=0)


def train(cfg):
    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(cfg.deterministic)
    start_time = time.time()
    has_timelimit = cfg.timelimit is not None
    run_dir_set = any("hydra.run.dir" in arg for arg in sys.argv)
    if has_timelimit and not run_dir_set:
        raise ValueError("The run has a timelimit but no fixed output directory")
    if has_timelimit:
        logger.info(
            f"Run has timelimit of {cfg.timelimit} hours and is restartable with same command"
        )

    logger.info(f"Starting GAN training with config: {cfg}")
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_dir = os.path.join(outdir, "checkpoints")
    generator_ckpt_dir = os.path.join(outdir, "checkpoints_generator")

    assert cfg.num_devices == 1, "Only single device training is supported for now"
    fabric = Fabric(
        accelerator=cfg.device,
        devices=cfg.num_devices,
    )
    fabric.launch()
    logger.info(f"World size {fabric.world_size}")

    meta_info = get_meta_info(outdir)

    if fabric.is_global_zero:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(generator_ckpt_dir, exist_ok=True)
        if meta_info["runid"] is None:
            logger.info("No previous wandb run found")
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=cfg_dict,
            mode="disabled" if not cfg.wandb.enabled else None,
            dir=outdir,
            id=meta_info["runid"],
            entity=cfg.wandb.entity,
            resume="allow",
        )
        logger.info(f"Already had {meta_info['restarts']} restarts")
        meta_info["restarts"] += 1
        meta_info["runid"] = run.id
        save_meta_info(outdir, meta_info)
        ensure_reproducibility(cfg, outdir=outdir, restart_idx=meta_info["restarts"])

    fabric.barrier()

    seed = cfg.seed + 100 * fabric.global_rank + 1000 * meta_info["restarts"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load pre-trained config
    pretrained_path = Path(cfg.pretrained_model_folder)
    pretrained_cfg = omegaconf.OmegaConf.load(pretrained_path / "resolved_config.yaml")
    logger.info(f"Loaded pretraining config {pretrained_cfg}")
    if fabric.is_global_zero:
        yaml_config = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
        with open(os.path.join(outdir, "resolved_config.yaml"), "w") as f:
            f.write(yaml_config)

    logger.info("Setting up dataset...")
    filtration_kwargs = omegaconf.OmegaConf.to_container(
        pretrained_cfg.filtration, resolve=True, throw_on_missing=True
    )
    dataset_cfg = omegaconf.OmegaConf.to_container(
        pretrained_cfg.dataset, resolve=True, throw_on_missing=True
    )
    if "val_overrides" in dataset_cfg:
        val_overrides = dataset_cfg["val_overrides"]
        del dataset_cfg["val_overrides"]
    else:
        val_overrides = {}

    t0 = time.time()
    filtration_train_set = hydra.utils.instantiate(
        dataset_cfg,
        split="train",
        filtration_kwargs=filtration_kwargs,
        _convert_="partial",
    )
    dataset_cfg.update(val_overrides)
    filtration_val_set = hydra.utils.instantiate(
        dataset_cfg,
        split="val",
        filtration_kwargs=filtration_kwargs,
        hold_out=filtration_train_set.graph_set,
        _convert_="partial",
    )
    logger.info(f"Loaded train and validation sets, converting now...")
    # The node_featurizers computes positional and structural features for the discriminator
    node_featurizer = NodeFeaturizer(**cfg.discriminator_features)
    graph_train_set = FiltrationToGraphDataset(
        filtration_train_set,
        featurizer=node_featurizer,
    )
    graph_val_set = FiltrationToGraphDataset(
        filtration_val_set, featurizer=node_featurizer, subset_size=cfg.val_ds_size
    )

    t1 = time.time()
    logger.info(f"Overall, dataset preparation took {(t1 - t0) / 60:.2f} minutes")

    # Setting up generator and discriminator
    generator = hydra.utils.instantiate(
        pretrained_cfg.model,
        num_edge_types=2,
        max_nodes=filtration_train_set.max_nodes,
        _recursive_=False,
    )
    discriminator = hydra.utils.instantiate(
        cfg.discriminator,
        feature_dims=node_featurizer.feature_dims,
        _recursive_=False,
    )
    if cfg.value_model is not None:
        value_model = hydra.utils.instantiate(
            cfg.value_model,
            output_dim=1,
            max_nodes=filtration_train_set.max_nodes,
            _recursive_=False,
        )
        logger.info(
            f"Initialized value model with {sum(p.numel() for p in value_model.parameters())} parameters"
        )
    else:
        value_model = None

    whitener = RewardWhitener(
        momentum=cfg_dict.get("reward_whitener", {}).get("momentum", 0.025)
    )

    pretrained_ckpt_path = (
        pretrained_path / "checkpoints" / cfg.pretrained_model_checkpoint
    )
    last_ckpt_path = get_last_ckpt_path(
        ckpt_dir
    )  # Checkpoint from GAN trining, possibly None

    logger.info(f"Loading generator from {pretrained_ckpt_path}")
    pretrained_state = fabric.load(pretrained_ckpt_path)
    generator.load_state_dict(pretrained_state["model"])

    generator_optim = torch.optim.Adam(
        generator.parameters(), lr=cfg.hyper_parameters.generator.lr
    )
    discriminator_optim = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.hyper_parameters.discriminator.lr
    )
    if value_model is not None:
        value_model_optim = torch.optim.Adam(
            value_model.parameters(), lr=cfg.hyper_parameters.value_model.lr
        )
    else:
        value_model_optim = None

    generator, generator_optim = fabric.setup(generator, generator_optim)
    generator.mark_forward_method("sample")
    for mod in generator.modules():
        if isinstance(mod, anfm.models.mixer_model.DataNormalization):
            mod.freeze_ema()
            logger.info(f"Freezing EMA for module {mod}")
    discriminator, discriminator_optim = fabric.setup(
        discriminator, discriminator_optim
    )
    if value_model is not None:
        value_model, value_model_optim = fabric.setup(value_model, value_model_optim)
    whitener = fabric.setup(whitener)

    state = {
        "generator": generator,
        "generator_optim": generator_optim,
        "discriminator": discriminator,
        "discriminator_optim": discriminator_optim,
        "value_model": value_model,
        "value_model_optim": value_model_optim,
        "whitener": whitener,
        "step": 0,
    }
    last_ckpt_path = get_last_ckpt_path(ckpt_dir)
    if last_ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {last_ckpt_path}")
        remaineder = fabric.load(last_ckpt_path, state, strict=True)
        assert not remaineder, f"Remaineder: {remaineder}"
        assert (
            generator is state["generator"]
            and generator_optim is state["generator_optim"]
        )
        assert (
            discriminator is state["discriminator"]
            and discriminator_optim is state["discriminator_optim"]
        )
        assert (
            value_model is state["value_model"]
            and value_model_optim is state["value_model_optim"]
        )
        assert whitener is state["whitener"]
    else:
        logger.info("Pretraining the discriminator...")
        pretrain_metrics = train_and_validate_discriminator(
            fabric=fabric,
            generator=generator,
            discriminator=discriminator,
            discriminator_optim=discriminator_optim,
            filtration_train_set=filtration_train_set,
            graph_train_set=graph_train_set,
            graph_val_set=graph_val_set,
            num_train_samples=cfg.hyper_parameters.discriminator.num_pretrain_samples,
            num_val_samples=cfg.hyper_parameters.num_val_samples,
            discriminator_batch_size=cfg.hyper_parameters.discriminator.batch_size,
            num_discriminator_steps=cfg.hyper_parameters.discriminator.num_pretrain_steps,
            featurizer=node_featurizer,
            device=cfg.device,
            validate=True,
            generation_batchsize=cfg.generation_batchsize,
            clamp_range=cfg.hyper_parameters.discriminator.clamp_range,
            log_prefix="<le>Discriminator Pretraining</le>:",
        )
        logger.info("Pretraining of discriminator complete...")
        wandb.log(pretrain_metrics, step=0)
        if cfg.value_model is not None:
            logger.info(
                f"Generating {cfg.hyper_parameters.value_model.num_pretrain_samples} samples for value model pretraining..."
            )
            filtrations, samples = generate_samples(
                generator,
                filtration_train_set,
                cfg.hyper_parameters.value_model.num_pretrain_samples,
                cfg.device,
                batch_size=cfg.generation_batchsize,
            )
            logger.info("Pretraining value model...")
            rewards = grade_samples(discriminator, samples, node_featurizer).clone()
            logger.opt(colors=True).info(
                f"Raw reward min, mean, max: {rewards.min().item()}, {rewards.mean().item()}, {rewards.max().item()}"
            )
            if cfg.hyper_parameters.lower_reward_clip is not None:
                rewards = torch.clamp(
                    rewards, min=cfg.hyper_parameters.lower_reward_clip
                )
            rewards = whitener(rewards)
            train_value_model(
                fabric=fabric,
                value_model=value_model,
                value_model_optim=value_model_optim,
                filtrations=filtrations,
                rewards=rewards,
                num_steps=cfg.hyper_parameters.value_model.num_pretrain_steps,
                batch_size=cfg.hyper_parameters.value_model.batch_size,
                num_accumulation_steps=cfg.hyper_parameters.value_model.grad_accumulation,
                grad_clip_value=cfg.hyper_parameters.value_model.grad_clip_value,
            )
        logger.info("Pretraining value model complete...")

    logger.info("Proceeding with main loop...")
    # Main traininig loop
    for step_idx in range(state["step"] + 1, cfg.hyper_parameters.num_iterations):
        # Train generator
        metrics = train_policy_ppo(
            fabric=fabric,
            generator=generator,
            discriminator=discriminator,
            value_model=value_model,
            reward_whitener=whitener,
            generator_optim=generator_optim,
            value_model_optim=value_model_optim,
            train_filtration_set=filtration_train_set,
            num_steps=cfg.hyper_parameters.generator.num_steps,
            generator_batch_size=cfg.hyper_parameters.generator.batch_size,
            generator_grad_accumulation=cfg.hyper_parameters.generator.grad_accumulation,
            value_model_num_steps=cfg.hyper_parameters.value_model.num_steps,
            value_model_batch_size=cfg.hyper_parameters.value_model.batch_size,
            value_model_grad_accumulation=cfg.hyper_parameters.value_model.grad_accumulation,
            num_epochs=cfg.hyper_parameters.generator.num_epochs,
            featurizer=node_featurizer,
            device=cfg.device,
            lower_reward_clip=cfg.hyper_parameters.lower_reward_clip,
            generation_batchsize=cfg.generation_batchsize,
            generator_grad_clip_value=cfg.hyper_parameters.generator.grad_clip_value,
            value_model_grad_clip_value=cfg.hyper_parameters.value_model.grad_clip_value,
            log_prefix=f"<le>Generator training step {step_idx}</le>:",
        )
        wandb.log(metrics, step=step_idx)

        # Train discriminator
        metrics = train_and_validate_discriminator(
            fabric=fabric,
            generator=generator,
            discriminator=discriminator,
            discriminator_optim=discriminator_optim,
            filtration_train_set=filtration_train_set,
            graph_train_set=graph_train_set,
            graph_val_set=graph_val_set,
            num_train_samples=cfg.hyper_parameters.discriminator.num_train_samples,
            num_val_samples=cfg.hyper_parameters.num_val_samples,
            discriminator_batch_size=cfg.hyper_parameters.discriminator.batch_size,
            num_discriminator_steps=cfg.hyper_parameters.discriminator.num_steps,
            featurizer=node_featurizer,
            device=cfg.device,
            validate=step_idx % cfg.hyper_parameters.discriminator.val_interval == 0,
            generation_batchsize=cfg.generation_batchsize,
            clamp_range=cfg.hyper_parameters.discriminator.clamp_range,
            log_prefix=f"<le>Discriminator training step {step_idx}</le>:",
        )
        wandb.log(metrics, step=step_idx)

        current_time = time.time()
        timelimit_reached = (
            has_timelimit and (current_time - start_time) / 3600 > cfg.timelimit
        )

        if (
            timelimit_reached
            or step_idx % cfg.save_interval == 0
            or step_idx == cfg.hyper_parameters.num_iterations - 1
        ):
            state["step"] = step_idx
            fabric.save(os.path.join(ckpt_dir, f"checkpoint_{step_idx}.pt"), state)
            fabric.save(
                os.path.join(generator_ckpt_dir, f"checkpoint_{step_idx}.pt"),
                {"model": generator},
            )

        if timelimit_reached:
            logger.info("Interrupting due to timeout, signaling for restart...")
            return 124 if fabric.is_global_zero else 0

    return 0


@hydra.main(config_path="./configs/gan", version_base=None)
def main(cfg):
    exit_code = train(cfg)
    logger.info(f"Exiting with code {exit_code}...")
    time.sleep(2)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
