"""Train COSMOS baseline."""
import json, os, sys, warnings
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    from COSMOS import cosmos
    from SpatialGlue.preprocess import lsi

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]
    ds_cfg = cfg["datasets"][dataset]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu = int(device.split(":")[-1]) if "cuda" in device else -1

    print(f"=== COSMOS: {dataset} ===")

    mdata = load_dataset(dataset, data_dir, cfg["datasets"])
    modalities = ds_cfg["modalities"]

    # Mod1 (RNA): HVG + normalize + scale
    adata_mod1 = mdata.mod[modalities[0]].copy()
    if "counts" in adata_mod1.layers:
        adata_mod1.X = adata_mod1.layers["counts"].copy()
    sc.pp.highly_variable_genes(adata_mod1, n_top_genes=3000, flavor="seurat_v3")
    adata_mod1 = adata_mod1[:, adata_mod1.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.scale(adata_mod1)
    if sp.issparse(adata_mod1.X):
        adata_mod1.X = np.asarray(adata_mod1.X.toarray(), dtype=np.float64)
    else:
        adata_mod1.X = np.asarray(adata_mod1.X, dtype=np.float64)

    # Mod2: LSI (ATAC) or normalize+scale (protein)
    adata_mod2 = mdata.mod[modalities[1]].copy()
    if "counts" in adata_mod2.layers:
        adata_mod2.X = adata_mod2.layers["counts"].copy()

    likelihoods = ds_cfg["likelihoods"]
    if likelihoods[1] == "bernoulli":
        # ATAC: LSI
        lsi(adata_mod2, n_components=50)
        lsi_X = adata_mod2.obsm["X_lsi"].astype(np.float64)
        adata_mod2_cosmos = ad.AnnData(lsi_X)
        adata_mod2_cosmos.obsm["spatial"] = adata_mod2.obsm["spatial"].copy()
    else:
        # Protein: normalize + scale
        sc.pp.normalize_total(adata_mod2, target_sum=1e4)
        sc.pp.log1p(adata_mod2)
        sc.pp.scale(adata_mod2)
        if sp.issparse(adata_mod2.X):
            adata_mod2.X = np.asarray(adata_mod2.X.toarray(), dtype=np.float64)
        else:
            adata_mod2.X = np.asarray(adata_mod2.X, dtype=np.float64)
        adata_mod2_cosmos = adata_mod2

    # Spatial regularization strength varies by dataset
    spatial_reg = 0.01
    if dataset == "dbit_seq":
        spatial_reg = 0.05

    model = cosmos.Cosmos(adata1=adata_mod1, adata2=adata_mod2_cosmos)
    model.preprocessing_data(n_neighbors=5)

    os.makedirs(out_dir, exist_ok=True)

    model.train(
        embedding_save_filepath=os.path.join(out_dir, "cosmos_embedding.tsv"),
        weights_save_filepath=os.path.join(out_dir, "cosmos_weights.tsv"),
        spatial_regularization_strength=spatial_reg,
        z_dim=bl["n_latent"],
        lr=1e-3,
        wnn_epoch=500,
        total_epoch=1000,
        max_patience_bef=10,
        max_patience_aft=30,
        min_stop=200,
        random_seed=20,
        gpu=gpu,
        regularization_acceleration=True,
        edge_subset_sz=1000000,
    )

    latent = model.embedding
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": bl["n_latent"], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
