"""Train scVI baseline. Handles both regular datasets and retina (batch/no-batch)."""
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
    bl = cfg["baseline_defaults"]

    # Determine if this is a retina variant
    variant = snakemake.params.get("variant", None)
    if variant and variant.startswith("scvi"):
        # Retina
        dataset = "retina"
        batch_correction = "no_batch" not in variant
        n_latent = 30
        n_hidden = 128
        n_layers = 2
        max_epochs = 100
    else:
        dataset = snakemake.params.dataset
        batch_correction = False
        n_latent = bl["n_latent"]
        n_hidden = bl["n_hidden"]
        n_layers = 1
        max_epochs = bl["max_epochs"]

    print(f"=== scVI: {dataset} (batch_correction={batch_correction}) ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])

    if dataset == "retina":
        adata = data[:, data.var["highly_variable"]].copy()
        adata.X = adata.layers["counts"]
    elif hasattr(data, "mod"):
        # For multimodal data, use the RNA modality
        ds_cfg = cfg["datasets"][dataset]
        rna_mod = ds_cfg["modalities"][0]
        adata = data.mod[rna_mod].copy()
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()
    else:
        adata = data.copy()
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()

    setup_kwargs = dict(layer=None)
    if batch_correction and "batch" in adata.obs.columns:
        setup_kwargs["categorical_covariate_keys"] = ["batch"]

    scvi_pkg.model.SCVI.setup_anndata(adata, **setup_kwargs)
    model = scvi_pkg.model.SCVI(
        adata, n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers, gene_likelihood="nb"
    )
    model.train(
        max_epochs=max_epochs, batch_size=128,
        early_stopping=True, early_stopping_patience=10,
    )

    latent = model.get_latent_representation()

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": n_latent, "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
