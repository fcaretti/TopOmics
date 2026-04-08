"""Train STAMP baseline (RNA only, spatial)."""
import json, os, sys, warnings
import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset

import torch


def main(snakemake):
    from sctm.stamp import STAMP

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]
    ds_cfg = cfg["datasets"][dataset]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"=== STAMP: {dataset} ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])

    # STAMP is RNA-only
    rna_mod = ds_cfg["modalities"][0]
    if hasattr(data, "mod"):
        adata_rna = data.mod[rna_mod].copy()
    else:
        adata_rna = data.copy()

    if "counts" in adata_rna.layers:
        adata_rna.X = adata_rna.layers["counts"].copy()

    if "spatial_connectivities" not in adata_rna.obsp and hasattr(data, "obsp"):
        if "spatial_connectivities" in data.obsp:
            adata_rna.obsp["spatial_connectivities"] = data.obsp["spatial_connectivities"]

    model = STAMP(
        adata_rna,
        n_topics=bl["n_latent"],
        n_layers=1,
        hidden_size=128,
        layer=None,
        dropout=0.1,
        enc_distribution="mvn",
        gene_likelihood="nb",
        mode="sign",
        verbose=True,
    )

    model.train(
        max_epochs=bl["max_epochs"],
        learning_rate=0.01,
        device=device,
        batch_size=256,
        early_stop=True,
        patience=20,
    )

    cell_topic = model.get_cell_by_topic(device=device)
    latent = cell_topic.values

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "stamp_params.pt"))
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)
    cell_topic.to_csv(os.path.join(out_dir, "cell_topic.csv"))

    feature_topic = model.get_feature_by_topic()
    feature_topic.to_csv(os.path.join(out_dir, "feature_topic.csv"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_topics": bl["n_latent"], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
