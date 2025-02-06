import graph_tool.all as _  # isort: skip

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from lightning.fabric import Fabric
from loguru import logger
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from anfm.data.visualization import draw_nx_graph
from anfm.train_gan import NodeFeaturizer


def load_model_and_datasets(
    ckpt_path,
    is_gan,
):
    cfg_path = ckpt_path.parent.parent / "resolved_config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_path)
    logger.info(f"Retrieved config {cfg}")

    if is_gan:
        logger.info("Retrieving pretrained config")
        pretrained_path = cfg.pretrained_model_folder
        cfg_path = Path(pretrained_path) / "resolved_config.yaml"
        gan_cfg = cfg
        cfg = omegaconf.OmegaConf.load(cfg_path)
        logger.info(f"Retrieved pretrained config {cfg}")

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

    fabric = Fabric(accelerator=cfg.device, devices=1)
    logger.info(f"Loading train set with config {dataset_cfg}")
    train_set = instantiate(
        dataset_cfg, split="train", filtration_kwargs=filtration_kwargs, _convert_="partial",
    )
    logger.info("Loading val set")
    dataset_cfg.update(val_overrides)
    val_set = instantiate(
        dataset_cfg,
        split="val",
        filtration_kwargs=filtration_kwargs,
        hold_out=train_set.graph_set,
        _convert_="partial",
    )
    test_set = instantiate(
        dataset_cfg,
        split="test",
        filtration_kwargs=filtration_kwargs,
        hold_out=train_set.graph_set + val_set.graph_set,
        _convert_="partial",
    )
    logger.info(f"Max nodes: {train_set.max_nodes}")
    model = instantiate(
        cfg.model,
        num_edge_types=2,
        max_nodes=train_set.max_nodes,
        _recursive_=False,
    )

    model = fabric.setup(model)
    model.mark_forward_method("sample")

    if is_gan:
        node_featurizer = NodeFeaturizer(**gan_cfg.discriminator_features)
        discriminator = instantiate(
            gan_cfg.discriminator,
            feature_dims=node_featurizer.feature_dims,
            _recursive_=False,
        )
        state = {"generator": model, "discriminator": discriminator}
    else:
        state = {"model": model}
        discriminator = None
        node_featurizer = None

    logger.info(
        f"Instantiated model with {sum(p.numel() for p in model.parameters())} parameters"
    )
    fabric.load(ckpt_path, state)

    logger.info("Loaded checkpoint")
    return fabric, model, discriminator, node_featurizer, train_set, val_set, test_set


def generate_samples(model, num_samples, batch_size, train_set, device, pbar=False):
    model.FORWARD_TIME = 0
    model.EXTRA_FEATURE_TIME = 0
    model.FILTRATION_CONV_TIME = 0
    model.NX_CONV_TIME = 0
    nx_graphs = []
    with torch.inference_mode(), tqdm(total=num_samples, disable=not pbar) as pbar:
        while len(nx_graphs) < num_samples:
            bs = min(batch_size, num_samples - len(nx_graphs))
            num_nodes = [train_set.sample_num_nodes() for _ in range(bs)]
            _, graphs = model.sample(num_nodes, device=device, return_filtrations=False)
            t0 = time.time()
            nx_graphs.extend([to_networkx(g, to_undirected=True) for g in graphs])
            t1 = time.time()
            model.NX_CONV_TIME += t1 - t0
            pbar.update(bs)
    return nx_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1024,
        help="Number of model samples to generate",
    )
    parser.add_argument(
        "--batchsize", type=int, default=512, help="Batch size for generation"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Set this option when evaluating on the test set. By default, the validation set is used.",
    )
    parser.add_argument(
        "--gan",
        action="store_true",
        help="Set this option when evaluating a checkpoint from adversarial finetuning",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Directory to store evaluation results",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    (
        fabric,
        model,
        discriminator,
        node_featurizer,
        train_set,
        val_set,
        test_set,
    ) = load_model_and_datasets(
        Path(args.checkpoint),
        args.gan,
    )

    if args.test:
        eval_set = test_set
    else:
        eval_set = val_set

    logger.info("Loaded checkpoint, starting to sample")

    t0 = time.time()
    model.eval()
    nx_graphs = generate_samples(
        model, args.n_samples, args.batchsize, train_set, fabric.device
    )
    t1 = time.time()
    logger.info(f"Took {(t1 - t0)/60:.2f} minutes to sample {args.n_samples} graphs")
    assert len(nx_graphs) == args.n_samples

    logger.info(
        f"Calculating evaluation metrics for {len(nx_graphs)} model samples, {len(eval_set.nx_graphs)} ground truth graphs"
    )
    eval_result = eval_set.evaluate_graphs(
        nx_graphs, eval_set.nx_graphs, train_set.graph_set, comprehensive=True
    )
    eval_result["checkpoint"] = args.checkpoint
    eval_result["n_samples"] = args.n_samples
    eval_result["eval_set_size"] = len(eval_set)
    eval_result["eval_set_size_nx"] = len(eval_set.nx_graphs)
    eval_result["use_test"] = args.test
    eval_result["time"] = t1 - t0
    eval_result["forward_time"] = model.FORWARD_TIME
    eval_result["extra_feature_time"] = model.EXTRA_FEATURE_TIME
    eval_result["filtration_conv_time"] = model.FILTRATION_CONV_TIME
    eval_result["nx_conv_time"] = model.NX_CONV_TIME
    logger.info(f"Evaluation results: {eval_result}")

    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok=True)
        with open(os.path.join(args.dump_dir, "eval_result.json"), "w") as f:
            json.dump(eval_result, f)
        with open(os.path.join(args.dump_dir, "preview_samples.pkl"), "wb") as f:
            pickle.dump(nx_graphs[:256], f)
        with open(os.path.join(args.dump_dir, "all_samples.pkl"), "wb") as f:
            pickle.dump(nx_graphs, f)
        for i in range(16):
            fig, ax = plt.subplots()
            graph = nx_graphs[i]
            layout = nx.fruchterman_reingold_layout(graph, seed=42)
            draw_nx_graph(graph, layout, ax)
            plt.axis("off")
            fig.savefig(os.path.join(args.dump_dir, f"sample_{i}.pdf"))
