#!/usr/bin/env python
"""
Training script for baseline models on Mouse Brain Spatial Multiome dataset.

This script trains baseline multimodal/spatial integration methods for comparison:
- SpatialGlue: Spatial multi-omics integration with graph attention
- STAMP: Spatial topic model (RNA only, uses sctm package)
- MultiVI: Deep generative model for RNA + ATAC
- MOFA+: Multi-Omics Factor Analysis

All models use a latent dimension of 10.

Usage:
    python train_baselines_mouse_brain.py --n_latent 10 --max_epochs 300
    python train_baselines_mouse_brain.py --skip_spatialglue  # Skip SpatialGlue
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# NumPy 2.0 compatibility fix for sctm package
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import anndata as ad
import mudata as md
import muon as mu
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore")

# Add sctm and SpatialGlue to path if needed
SCRATCH_SITE_PACKAGES = "/scratch/fcaretti/omics_topic_uv/lib/python3.12/site-packages"
if SCRATCH_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SCRATCH_SITE_PACKAGES)

# Data paths
RNA_PATH = "/data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_RNA.h5ad"
ATAC_PATH = "/data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_peaks_normalized.h5ad"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline models on Mouse Brain Spatial Multiome dataset"
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=10,
        help="Number of latent dimensions (default: 10)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Maximum training epochs for deep learning models (default: 300)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/mouse_brain_spatial/baselines",
        help="Output directory for models",
    )
    parser.add_argument(
        "--skip_spatialglue",
        action="store_true",
        help="Skip SpatialGlue training",
    )
    parser.add_argument(
        "--skip_stamp",
        action="store_true",
        help="Skip STAMP training",
    )
    parser.add_argument(
        "--skip_multivi",
        action="store_true",
        help="Skip MultiVI training",
    )
    parser.add_argument(
        "--skip_mofa",
        action="store_true",
        help="Skip MOFA+ training",
    )
    parser.add_argument(
        "--skip_cosmos",
        action="store_true",
        help="Skip COSMOS training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for training (default: cuda:0 if available)",
    )
    return parser.parse_args()


def binarize_atac(adata_atac):
    """Binarize ATAC data."""
    X = adata_atac.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.ones_like(X.data)
        X.eliminate_zeros()
        adata_atac.layers["binary"] = X
        adata_atac.layers["counts"] = X.copy()
    else:
        binary = (X != 0).astype(np.float32)
        adata_atac.layers["binary"] = binary
        adata_atac.layers["counts"] = binary.copy()


def load_data():
    """Load and preprocess Mouse Brain Spatial Multiome data."""
    print("Loading Mouse Brain Spatial Multiome data...")

    adata_rna = sc.read_h5ad(RNA_PATH)
    adata_atac = sc.read_h5ad(ATAC_PATH)

    # Ensure counts layer for RNA
    if "counts" not in adata_rna.layers:
        adata_rna.layers["counts"] = adata_rna.X.copy()

    # Binarize ATAC data
    binarize_atac(adata_atac)

    # Create MuData
    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    # Build spatial neighbor graph from RNA coordinates
    sc.pp.neighbors(
        mdata.mod["rna"],
        use_rep="spatial",
        n_neighbors=5,
        metric="euclidean",
        key_added="spatial",
    )

    # Share spatial connectivities across modalities
    mdata.obsp["spatial_connectivities"] = mdata.mod["rna"].obsp["spatial_connectivities"]
    mdata.mod["atac"].obsp["spatial_connectivities"] = mdata.mod["rna"].obsp[
        "spatial_connectivities"
    ]
    mdata.mod["atac"].obsp["spatial_distances"] = mdata.mod["rna"].obsp["spatial_distances"]

    print(f"  RNA: {mdata.mod['rna'].shape}")
    print(f"  ATAC: {mdata.mod['atac'].shape}")
    print(f"  Total spots: {mdata.n_obs}")

    return mdata


def _history_to_df(history):
    """Convert training history to DataFrame."""
    if isinstance(history, pd.DataFrame):
        return history
    try:
        return pd.DataFrame(history)
    except ValueError:
        return pd.DataFrame({k: [v] if np.isscalar(v) else v for k, v in history.items()})


def _save_history(history, path):
    """Save training history to CSV."""
    history_df = _history_to_df(history)
    history_df.to_csv(path, index=False)


# =============================================================================
# SpatialGlue
# =============================================================================
def train_spatialglue(mdata, n_latent, output_dir, device):
    """Train SpatialGlue model."""
    from SpatialGlue.preprocess import (
        clr_normalize_each_cell,
        construct_neighbor_graph,
        lsi,
        pca,
    )
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

    print("\n" + "=" * 70)
    print("Training SpatialGlue (RNA + ATAC)")
    print("=" * 70)

    # Prepare data for SpatialGlue
    adata_rna = mdata.mod["rna"].copy()
    adata_atac = mdata.mod["atac"].copy()

    # RNA preprocessing: normalize, log1p, then PCA for features
    adata_rna.X = adata_rna.layers["counts"].copy()
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)

    # Compute PCA features for RNA
    adata_rna.obsm["feat"] = pca(adata_rna, n_comps=min(50, adata_rna.n_vars - 1))

    # ATAC preprocessing: LSI for features
    adata_atac.X = adata_atac.layers["counts"].copy()
    # LSI requires binary/count data
    lsi(adata_atac, n_components=min(50, adata_atac.n_vars - 1))
    adata_atac.obsm["feat"] = adata_atac.obsm["X_lsi"]

    # Construct neighbor graphs (spatial + feature graphs)
    data = construct_neighbor_graph(
        adata_rna, adata_atac, datatype="Spatial-epigenome-transcriptome", n_neighbors=6
    )

    # Initialize and train SpatialGlue
    model = Train_SpatialGlue(
        data,
        datatype="Spatial-epigenome-transcriptome",
        device=torch.device(device),
        random_seed=42,
        learning_rate=0.0001,
        epochs=600,
        dim_input=data["adata_omics1"].obsm["feat"].shape[1],
        dim_output=n_latent,
        weight_factors=[1, 5, 1, 1],
    )

    print("Training...")
    output = model.train()

    # Get combined latent representation
    latent = output["SpatialGlue"]
    print(f"Latent shape: {latent.shape}")

    # Save results
    model_dir = os.path.join(output_dir, "spatialglue")
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "latent_spatialglue.npy"), latent)
    np.save(os.path.join(model_dir, "latent_omics1.npy"), output["emb_latent_omics1"])
    np.save(os.path.join(model_dir, "latent_omics2.npy"), output["emb_latent_omics2"])
    np.save(os.path.join(model_dir, "alpha.npy"), output["alpha"])

    print(f"Results saved to: {model_dir}")

    return latent


# =============================================================================
# STAMP (RNA only)
# =============================================================================
def train_stamp(mdata, n_latent, max_epochs, output_dir, device):
    """Train STAMP model on RNA modality only."""
    from sctm.stamp import STAMP

    print("\n" + "=" * 70)
    print("Training STAMP (RNA only)")
    print("=" * 70)

    # Prepare RNA data for STAMP
    adata_rna = mdata.mod["rna"].copy()

    # STAMP needs counts in X and spatial_connectivities in obsp
    adata_rna.X = adata_rna.layers["counts"].copy()

    # Ensure spatial connectivities are available
    if "spatial_connectivities" not in adata_rna.obsp:
        adata_rna.obsp["spatial_connectivities"] = mdata.mod["rna"].obsp[
            "spatial_connectivities"
        ]

    # Initialize STAMP model
    model = STAMP(
        adata_rna,
        n_topics=n_latent,
        n_layers=1,  # SGC layers for spatial smoothing
        hidden_size=128,
        layer=None,
        dropout=0.1,
        enc_distribution="mvn",
        gene_likelihood="nb",
        mode="sign",
        verbose=True,
    )

    # Train
    print("Training...")
    model.train(
        max_epochs=max_epochs,
        learning_rate=0.01,
        device=device,
        batch_size=256,
        early_stop=True,
        patience=20,
    )

    # Get latent representation (cell by topic)
    cell_topic = model.get_cell_by_topic(device=device)
    latent = cell_topic.values
    print(f"Latent shape: {latent.shape}")

    # Save results
    model_dir = os.path.join(output_dir, "stamp")
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "latent_stamp.npy"), latent)
    cell_topic.to_csv(os.path.join(model_dir, "cell_topic.csv"))

    # Save model parameters
    model.save(os.path.join(model_dir, "stamp_params.pt"))

    # Get and save topic-gene matrix
    feature_topic = model.get_feature_by_topic()
    feature_topic.to_csv(os.path.join(model_dir, "feature_topic.csv"))

    print(f"Results saved to: {model_dir}")

    return latent


# =============================================================================
# MultiVI (RNA + ATAC)
# =============================================================================
def train_multivi(mdata, n_latent, max_epochs, output_dir):
    """Train MultiVI model on RNA + ATAC."""
    import scvi

    print("\n" + "=" * 70)
    print("Training MultiVI (RNA + ATAC)")
    print("=" * 70)

    # Create a copy for MultiVI
    mdata_multivi = mdata.copy()

    # MultiVI needs counts in .X
    mdata_multivi.mod["rna"].X = mdata_multivi.mod["rna"].layers["counts"].copy()
    mdata_multivi.mod["atac"].X = mdata_multivi.mod["atac"].layers["counts"].copy()

    # Setup MultiVI
    scvi.model.MULTIVI.setup_mudata(
        mdata_multivi,
        rna_layer=None,
        atac_layer=None,
        batch_key=None,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )

    # Create model
    model = scvi.model.MULTIVI(
        mdata_multivi,
        n_latent=n_latent,
        n_hidden=128,
    )

    # Train
    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
        early_stopping=True,
    )

    # Get latent representation
    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")

    # Save model
    model_path = os.path.join(output_dir, "multivi")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Save latent representation
    np.save(os.path.join(output_dir, "latent_multivi.npy"), latent)

    # Save training history
    _save_history(model.history, os.path.join(output_dir, "multivi_history.csv"))

    return latent


# =============================================================================
# MOFA+
# =============================================================================
def train_mofa(mdata, n_latent, output_dir):
    """Train MOFA+ model."""
    print("\n" + "=" * 70)
    print("Training MOFA+ (RNA + ATAC)")
    print("=" * 70)

    # Create a copy for MOFA
    mdata_mofa = mdata.copy()

    # MOFA needs normalized data
    # RNA: log-normalize and scale
    mdata_mofa.mod["rna"].X = mdata_mofa.mod["rna"].layers["counts"].copy()
    sc.pp.normalize_total(mdata_mofa.mod["rna"], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod["rna"])
    sc.pp.scale(mdata_mofa.mod["rna"])

    # ATAC: normalize and scale (already binarized)
    mdata_mofa.mod["atac"].X = mdata_mofa.mod["atac"].layers["counts"].copy()
    sc.pp.normalize_total(mdata_mofa.mod["atac"], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod["atac"])
    sc.pp.scale(mdata_mofa.mod["atac"])

    # Train MOFA+
    print("Training...")
    mu.tl.mofa(
        mdata_mofa,
        n_factors=n_latent,
        convergence_mode="medium",
        use_obs="intersection",
    )

    # Get latent representation
    latent = mdata_mofa.obsm["X_mofa"]
    print(f"Latent shape: {latent.shape}")

    # Save MuData with MOFA results
    mdata_path = os.path.join(output_dir, "mdata_mofa.h5mu")
    mdata_mofa.write(mdata_path)
    print(f"MuData saved to: {mdata_path}")

    # Save latent representation
    np.save(os.path.join(output_dir, "latent_mofa.npy"), latent)

    return latent


# =============================================================================
# COSMOS
# =============================================================================
def train_cosmos(mdata, n_latent, output_dir, device):
    """Train COSMOS model on RNA + ATAC.

    The COSMOS tutorial (tutorial_ATAC_RNA_Seq_MouseBrain.ipynb) expects
    pre-processed inputs: HVG-selected + normalized + scaled RNA (3000 genes),
    and LSI-reduced ATAC (50 components). We replicate that here from raw counts,
    then pass to COSMOS with preprocessing_data defaults (no further transform).
    """
    from COSMOS import cosmos

    print("\n" + "=" * 70)
    print("Training COSMOS (RNA + ATAC)")
    print("=" * 70)

    # --- RNA: HVG selection + normalize + log1p + scale (matching tutorial's 3000 features) ---
    adata_rna = mdata.mod["rna"].copy()
    if "counts" in adata_rna.layers:
        adata_rna.X = adata_rna.layers["counts"].copy()
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000, flavor="seurat_v3")
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    # Convert to dense float64
    if sp.issparse(adata_rna.X):
        adata_rna.X = np.asarray(adata_rna.X.toarray(), dtype=np.float64)
    else:
        adata_rna.X = np.asarray(adata_rna.X, dtype=np.float64)
    print(f"  RNA preprocessed: {adata_rna.shape}")

    # --- ATAC: LSI reduction to 50 components (matching tutorial) ---
    from SpatialGlue.preprocess import lsi

    adata_atac = mdata.mod["atac"].copy()
    if "counts" in adata_atac.layers:
        adata_atac.X = adata_atac.layers["counts"].copy()
    lsi(adata_atac, n_components=50)
    lsi_X = adata_atac.obsm["X_lsi"].astype(np.float64)
    # Create new AnnData with LSI components as features
    adata_atac_lsi = ad.AnnData(lsi_X)
    adata_atac_lsi.obsm["spatial"] = adata_atac.obsm["spatial"].copy()
    print(f"  ATAC preprocessed (LSI): {adata_atac_lsi.shape}")

    # Build model — no further preprocessing (data already transformed)
    model = cosmos.Cosmos(adata1=adata_rna, adata2=adata_atac_lsi)
    model.preprocessing_data(n_neighbors=5)

    # Parse GPU index from device string
    gpu = int(device.split(":")[-1]) if "cuda" in device else -1

    # Train — parameters from tutorial_ATAC_RNA_Seq_MouseBrain.ipynb
    embedding_path = os.path.join(output_dir, "cosmos_embedding.tsv")
    weights_path = os.path.join(output_dir, "cosmos_weights.tsv")

    print("Training...")
    model.train(
        embedding_save_filepath=embedding_path,
        weights_save_filepath=weights_path,
        spatial_regularization_strength=0.01,
        z_dim=n_latent,
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

    # Get latent representation
    latent = model.embedding
    print(f"Latent shape: {latent.shape}")

    # Save results
    model_dir = os.path.join(output_dir, "cosmos")
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "latent_cosmos.npy"), latent)
    np.save(os.path.join(model_dir, "weights.npy"), model.weights)

    print(f"Results saved to: {model_dir}")

    return latent


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Mouse Brain Spatial Multiome - Baseline Models Training")
    print("=" * 70)
    print(f"N latent dimensions: {args.n_latent}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load data
    mdata = load_data()

    results = {}

    # Train SpatialGlue
    if not args.skip_spatialglue:
        try:
            latent = train_spatialglue(
                mdata, args.n_latent, args.output_dir, args.device
            )
            results["spatialglue"] = latent
        except Exception as e:
            print(f"SpatialGlue training failed: {e}")
            import traceback

            traceback.print_exc()

    # Train STAMP (RNA only)
    if not args.skip_stamp:
        try:
            latent = train_stamp(
                mdata, args.n_latent, args.max_epochs, args.output_dir, args.device
            )
            results["stamp"] = latent
        except Exception as e:
            print(f"STAMP training failed: {e}")
            import traceback

            traceback.print_exc()

    # Train MultiVI (RNA + ATAC)
    if not args.skip_multivi:
        try:
            latent = train_multivi(mdata, args.n_latent, args.max_epochs, args.output_dir)
            results["multivi"] = latent
        except Exception as e:
            print(f"MultiVI training failed: {e}")
            import traceback

            traceback.print_exc()

    # Train MOFA+ (RNA + ATAC)
    if not args.skip_mofa:
        try:
            latent = train_mofa(mdata, args.n_latent, args.output_dir)
            results["mofa"] = latent
        except Exception as e:
            print(f"MOFA+ training failed: {e}")
            import traceback

            traceback.print_exc()

    # Train COSMOS (RNA + ATAC)
    if not args.skip_cosmos:
        try:
            latent = train_cosmos(
                mdata, args.n_latent, args.output_dir, args.device
            )
            results["cosmos"] = latent
        except Exception as e:
            print(f"COSMOS training failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for name, latent in results.items():
        print(f"  {name}: latent shape = {latent.shape}")
    print(f"\nAll results saved to: {args.output_dir}")

    # Save summary
    if results:
        summary = {
            "model": list(results.keys()),
            "n_latent": [r.shape[1] for r in results.values()],
            "n_cells": [r.shape[0] for r in results.values()],
        }
        pd.DataFrame(summary).to_csv(
            os.path.join(args.output_dir, "training_summary.csv"), index=False
        )


if __name__ == "__main__":
    main()
