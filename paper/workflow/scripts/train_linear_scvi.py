"""Train LinearSCVI baseline for retina dataset."""
import json, os, sys, warnings
import numpy as np
import scanpy as sc
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    import scvi as scvi_pkg

    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    variant = snakemake.params.variant

    batch_correction = "no_batch" not in variant

    print(f"=== LinearSCVI retina (batch_correction={batch_correction}) ===")

    adata = load_dataset("retina", data_dir, cfg["datasets"])
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    adata_hvg.X = adata_hvg.layers["counts"]

    setup_kwargs = dict(layer=None)
    if batch_correction:
        setup_kwargs["batch_key"] = "batch"
    scvi_pkg.model.LinearSCVI.setup_anndata(adata_hvg, **setup_kwargs)

    model = scvi_pkg.model.LinearSCVI(adata_hvg, n_latent=30)
    model.train(
        max_epochs=100, batch_size=128,
        early_stopping=True, early_stopping_patience=10,
    )

    latent = model.get_latent_representation()

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": 30, "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
