"""
TopOmics training for retina dataset (batch correction axis).
Called by Snakemake with params: cfg, out_dir, data_dir.
"""

import json
import os
import sys
import warnings

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore", message=".*was not registered in the param store.*")
warnings.filterwarnings("ignore", message=".*Found plate statements in guide but not model.*")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset

from topomics.models import MultimodalAmortizedLDA


def main(snakemake):
    cfg_name = snakemake.params.cfg
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config

    ds_cfg = cfg["datasets"]["retina"]
    ds_topomics = ds_cfg.get("topomics", {})
    retina_cfg = cfg["retina_configs"][cfg_name]

    n_topics = ds_topomics.get("n_topics", 30)
    n_hidden = ds_topomics.get("n_hidden", 128)
    max_epochs = ds_topomics.get("max_epochs", 100)
    batch_size = ds_topomics.get("batch_size", 128)

    encode_covariates = retina_cfg["encode_covariates"]
    batch_correction = retina_cfg["batch_correction"]

    print(f"=== TopOmics Retina: {cfg_name} ===")
    print(f"  encode_covariates={encode_covariates}, batch_correction={batch_correction}")

    # --- Load data ---
    adata = load_dataset("retina", data_dir, cfg["datasets"])

    # Use HVGs and raw counts
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    adata_hvg.X = adata_hvg.layers["counts"]

    # --- Setup ---
    setup_kwargs = dict(layer=None)
    if batch_correction:
        setup_kwargs["categorical_covariate_keys"] = ["batch"]
    MultimodalAmortizedLDA.setup_anndata(adata_hvg, **setup_kwargs)

    # --- Create model ---
    model = MultimodalAmortizedLDA(
        adata_hvg,
        n_topics=n_topics,
        n_inputs_modalities=[adata_hvg.n_vars],
        likelihoods=["gamma_poisson"],
        n_hidden=n_hidden,
        encode_covariates=encode_covariates if batch_correction else False,
    )

    # --- Train ---
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        validation_size=0.1,
        early_stopping=True,
        early_stopping_monitor="elbo_val",
        early_stopping_patience=10,
    )

    # --- Save ---
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)

    theta = model.get_latent_representation()
    latent_vals = theta.values if hasattr(theta, "values") else np.asarray(theta)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent_vals)

    metrics = {
        "perplexity": model.get_perplexity(),
        "entropy": model.get_entropy(normalised=True),
        "diversity": model.get_topic_diversity(),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
