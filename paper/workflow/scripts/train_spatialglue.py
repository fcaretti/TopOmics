"""Train SpatialGlue baseline."""
import json, os, sys, warnings
import numpy as np
import scanpy as sc
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    from SpatialGlue.preprocess import construct_neighbor_graph, lsi, pca
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]
    ds_cfg = cfg["datasets"][dataset]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"=== SpatialGlue: {dataset} ===")

    mdata = load_dataset(dataset, data_dir, cfg["datasets"])
    modalities = ds_cfg["modalities"]

    adata_mod1 = mdata.mod[modalities[0]].copy()
    adata_mod2 = mdata.mod[modalities[1]].copy()

    # Mod1 (RNA-like): normalize + PCA
    if "counts" in adata_mod1.layers:
        adata_mod1.X = adata_mod1.layers["counts"].copy()
    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    adata_mod1.obsm["feat"] = pca(adata_mod1, n_comps=min(50, adata_mod1.n_vars - 1))

    # Mod2: LSI (for ATAC) or PCA (for protein)
    if "counts" in adata_mod2.layers:
        adata_mod2.X = adata_mod2.layers["counts"].copy()
    likelihoods = ds_cfg["likelihoods"]
    if likelihoods[1] == "bernoulli":
        # ATAC-like: LSI
        lsi(adata_mod2, n_components=min(50, adata_mod2.n_vars - 1))
        adata_mod2.obsm["feat"] = adata_mod2.obsm["X_lsi"]
        datatype = "Spatial-epigenome-transcriptome"
    else:
        # Protein-like: normalize + PCA
        sc.pp.normalize_total(adata_mod2, target_sum=1e4)
        sc.pp.log1p(adata_mod2)
        adata_mod2.obsm["feat"] = pca(adata_mod2, n_comps=min(50, adata_mod2.n_vars - 1))
        datatype = "Spatial-CITE-seq"

    graph_data = construct_neighbor_graph(adata_mod1, adata_mod2, datatype=datatype, n_neighbors=6)

    model = Train_SpatialGlue(
        graph_data,
        datatype=datatype,
        device=torch.device(device),
        random_seed=42,
        learning_rate=0.0001,
        epochs=600,
        dim_input=graph_data["adata_omics1"].obsm["feat"].shape[1],
        dim_output=bl["n_latent"],
        weight_factors=[1, 5, 1, 1],
    )

    output = model.train()
    latent = output["SpatialGlue"]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)
    np.save(os.path.join(out_dir, "alpha.npy"), output["alpha"])

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": bl["n_latent"], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
