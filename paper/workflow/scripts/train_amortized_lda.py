"""Train scvi-tools AmortizedLDA baseline on a single modality."""

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    import scvi

    dataset = snakemake.params.dataset
    modality = snakemake.params.modality
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]

    print(f"=== AmortizedLDA: {dataset} / {modality} ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])

    # Extract single-modality AnnData
    if hasattr(data, "mod"):
        adata = data.mod[modality].copy()
    else:
        adata = data.copy()

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    n_topics = bl["n_latent"]
    scvi.model.AmortizedLDA.setup_anndata(adata, layer=None)
    model = scvi.model.AmortizedLDA(
        adata,
        n_topics=n_topics,
        n_hidden=bl["n_hidden"],
        cell_topic_prior=1 / n_topics,
    )
    model.train(
        max_epochs=bl["max_epochs"],
        train_size=0.8,
        validation_size=0.2,
        batch_size=128,
    )

    theta = model.get_latent_representation()
    latent = np.asarray(theta)

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_topics": n_topics, "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
