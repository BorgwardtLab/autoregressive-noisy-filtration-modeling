import os
from collections import deque

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    Compose,
)

import wandb
from molecule_labeling import DATA_DIR
from molecule_labeling.autoencoder import DiscreteLabelAutoencoder, LabelAutoencoder
from molecule_labeling.dataset import GuacamolDataset
from molecule_labeling.utils import FCDMetric, count_valid_molecules


@hydra.main(version_base=None, config_path="./configs", config_name="continuous_vae")
def train(cfg):
    device = cfg.device
    rwpe_dim = 8
    laplace_transform = AddLaplacianEigenvectorPE(3, "laplace")  # 3 is the default
    rwpe_transform = AddRandomWalkPE(rwpe_dim, "rwpe")  # 8 is the default
    transforms = Compose([laplace_transform, rwpe_transform])

    train_ds = GuacamolDataset(
        "train", DATA_DIR / "guacamol_vae_data", pre_transform=transforms
    )
    val_ds = GuacamolDataset(
        "val", DATA_DIR / "guacamol_vae_data", pre_transform=transforms
    )
    _ = GuacamolDataset(
        "test", DATA_DIR / "guacamol_vae_data", pre_transform=transforms
    )
    val_mols = [
        val_ds.graph_to_mol(val_ds[idx])
        for idx in np.random.choice(len(val_ds), 20_000, replace=False)
    ]
    fcd_metric = FCDMetric(val_mols)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=4)
    val_iter = iter(val_loader)

    if cfg.variant == "continuous":
        model = LabelAutoencoder(
            node_labels={
                "atom_labels": 12,
                "explicit_hydrogens": 4,
                "charges": 5,
                "radical_electrons": 3,
            },
            edge_labels={"bond_labels": 4},
            structural_features={"laplace": 3, "rwpe": rwpe_dim},
            decoder_layers=5,
            encoder_layers=5,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
    elif cfg.variant == "discrete":
        model = DiscreteLabelAutoencoder(
            node_labels={
                "atom_labels": 12,
                "explicit_hydrogens": 4,
                "charges": 5,
                "radical_electrons": 3,
            },
            edge_labels={"bond_labels": 4},
            structural_features={"laplace": 3, "rwpe": rwpe_dim},
            decoder_layers=5,
            encoder_layers=5,
            latent_dim=8,
            temperature=0.1,
            hard=True,
        )
    else:
        raise ValueError(f"Invalid variant: {cfg.variant}")

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    wandb.init(project="molecule-relabeling")

    sample_buffer = deque(maxlen=20000)
    step = 0
    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            batch = batch.to(device)
            results = model(batch)
            loss = model.evidence_lower_bound(batch, results)
            optim.zero_grad()
            loss.backward()
            optim.step()
            log_info = {"train/loss": loss.item()}
            if step % 10 == 0:
                logger.info(f"Epoch {epoch}, step {step}, loss={loss.item()}")

            if step % 250 == 0:
                model.eval()
                with torch.inference_mode():
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    val_batch = val_batch.to(device)
                    results = model(val_batch)
                    val_loss = model.evidence_lower_bound(val_batch, results)
                    assert val_batch.edge_index.shape[1] != val_batch.num_nodes
                    real_mols = val_ds.batch_to_mols(val_batch)
                    all_attr = results[
                        "node_logits"
                    ]  # Will take the argmax when deconding atoms
                    all_attr.update(
                        results["edge_logits"]
                    )  # Will take the argmax when decoding bonds
                    (
                        reconstructed_mols,
                        num_valid_reconstruction,
                        num_correct_reconstruction,
                    ) = count_valid_molecules(
                        val_batch, all_attr, reference_mols=real_mols
                    )

                    node_attr, edge_attr = model.sample(val_batch, variant="max")
                    all_attr = node_attr
                    all_attr.update(edge_attr)
                    sampled_mols, num_valid_sampled, _ = count_valid_molecules(
                        val_batch, all_attr
                    )
                    sample_buffer.extend(sampled_mols)
                    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
                    for i in range(4):
                        actual = Draw.MolToImage(real_mols[i])
                        reconstruction = Draw.MolToImage(reconstructed_mols[i])
                        sampled = Draw.MolToImage(sampled_mols[i])
                        ax[0, i].imshow(actual)
                        ax[0, i].axis("off")
                        ax[1, i].imshow(reconstruction)
                        ax[1, i].axis("off")
                        ax[2, i].imshow(sampled)
                        ax[2, i].axis("off")

                    log_info.update(
                        {
                            "val/loss": val_loss.item(),
                            "val/valid_sample": num_valid_sampled / len(sampled_mols),
                            "val/correct_reconstruction": num_correct_reconstruction
                            / len(reconstructed_mols),
                            "val/valid_reconstruction": num_valid_reconstruction
                            / len(reconstructed_mols),
                            "val/samples": wandb.Image(fig),
                        }
                    )
                    logger.info(
                        f"During reconstruction, {num_valid_reconstruction}/{len(reconstructed_mols)} were valid, {num_correct_reconstruction}/{len(reconstructed_mols)} were correct"
                    )
                    logger.info(
                        f"During sampling, {num_valid_sampled}/{len(sampled_mols)} were valid, {num_correct_reconstruction}/{len(sampled_mols)} were correct"
                    )
                    plt.close(fig)
                model.train()

            if step % 500 == 0 and step > 0:
                fcd = fcd_metric(sample_buffer)
                log_info.update({"val/fcd": fcd})
                logger.info(f"FCD: {fcd}")

            wandb.log(log_info, step=step)
            step += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(outdir, f"model_{epoch}.pth"))
            torch.save(
                model.decoder.state_dict(), os.path.join(outdir, f"decoder_{epoch}.pth")
            )

    torch.save(model.state_dict(), os.path.join(outdir, "model.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(outdir, "decoder.pth"))


if __name__ == "__main__":
    train()
