#!/usr/bin/env python
"""
Training script for baseline models on Visium H&E dataset.

This script trains baseline methods for comparison:
- scVI: Deep generative model for scRNA-seq
- STAMP: Spatial topic model (from sctm package)
- AmortizedLDA: Amortized LDA from scvi-tools (non-spatial)

All models use a latent dimension of 10.

Usage:
    python train_baselines_visium.py --n_latent 10 --max_epochs 400
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

# NumPy 2.0 compatibility fix for sctm package
if not hasattr(np, "Inf"):
    np.Inf = np.inf

warnings.filterwarnings("ignore")

# Add sctm to path if needed
SCRATCH_SITE_PACKAGES = "/scratch/fcaretti/omics_topic_uv/lib/python3.12/site-packages"
if SCRATCH_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SCRATCH_SITE_PACKAGES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline models on Visium H&E dataset"
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=10,
        help="Number of latent dimensions/topics (default: 10)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=400,
        help="Maximum training epochs (default: 400)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/sctm_comparison/baselines",
        help="Output directory for models",
    )
    parser.add_argument(
        "--skip_scvi",
        action="store_true",
        help="Skip scVI training",
    )
    parser.add_argument(
        "--skip_stamp",
        action="store_true",
        help="Skip STAMP training",
    )
    parser.add_argument(
        "--skip_lda",
        action="store_true",
        help="Skip AmortizedLDA training",
    )
    return parser.parse_args()


def load_data():
    """Load and preprocess Visium H&E data."""
    print("Loading Visium H&E data...")
    adata = sq.datasets.visium_hne_adata()

    # Use raw counts
    adata.X = adata.raw.X.copy()

    # Ensure counts layer exists
    adata.layers["counts"] = adata.X.copy()

    # Build spatial neighbor graph if not present
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic")

    print(f"  Spots: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")

    return adata


# =============================================================================
# scVI
# =============================================================================
def train_scvi(adata, n_latent, max_epochs, output_dir):
    """Train scVI model."""
    import scvi

    print("\n" + "=" * 70)
    print("Training scVI")
    print("=" * 70)

    # Create a copy for scVI
    adata_scvi = adata.copy()

    # Setup scVI
    scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts")

    # Create model
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_hidden=128,
        n_layers=1,
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

    # Save model and latent
    model_path = os.path.join(output_dir, "scvi")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)

    np.save(os.path.join(output_dir, "latent_scvi.npy"), latent)
    print(f"Model saved to: {model_path}")

    return latent


# =============================================================================
# STAMP
# =============================================================================
def train_stamp(adata, n_latent, max_epochs, output_dir):
    """Train STAMP model."""
    from sctm.stamp import STAMP

    print("\n" + "=" * 70)
    print("Training STAMP")
    print("=" * 70)

    # Create a copy for STAMP
    adata_stamp = adata.copy()

    # STAMP needs counts in X and spatial_connectivities in obsp
    adata_stamp.X = adata_stamp.layers["counts"].copy()

    # Initialize STAMP model
    model = STAMP(
        adata_stamp,
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
        device="cuda",
        batch_size=256,
        early_stop=True,
        patience=20,
    )

    # Get latent representation (cell by topic)
    cell_topic = model.get_cell_by_topic(device="cuda")
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
# AmortizedLDA (scvi-tools)
# =============================================================================
def train_amortized_lda(adata, n_latent, max_epochs, output_dir):
    """Train AmortizedLDA from scvi-tools."""
    import scvi

    print("\n" + "=" * 70)
    print("Training AmortizedLDA (scvi-tools)")
    print("=" * 70)

    # Create a copy
    adata_lda = adata.copy()

    # Setup AmortizedLDA
    scvi.model.AmortizedLDA.setup_anndata(adata_lda, layer="counts")

    # Create model
    model = scvi.model.AmortizedLDA(
        adata_lda,
        n_topics=n_latent,
        n_hidden=128,
    )

    # Train
    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
    )

    # Get latent representation (topic proportions)
    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")

    # Save model and latent
    model_path = os.path.join(output_dir, "amortized_lda")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)

    np.save(os.path.join(output_dir, "latent_amortized_lda.npy"), latent)
    print(f"Model saved to: {model_path}")

    return latent


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Visium H&E - Baseline Models Training")
    print("=" * 70)
    print(f"N latent dimensions: {args.n_latent}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load data
    adata = load_data()

    results = {}

    # Train scVI
    if not args.skip_scvi:
        try:
            latent = train_scvi(adata, args.n_latent, args.max_epochs, args.output_dir)
            results["scvi"] = latent
        except Exception as e:
            print(f"scVI training failed: {e}")
            import traceback
            traceback.print_exc()

    # Train STAMP
    if not args.skip_stamp:
        try:
            latent = train_stamp(adata, args.n_latent, args.max_epochs, args.output_dir)
            results["stamp"] = latent
        except Exception as e:
            print(f"STAMP training failed: {e}")
            import traceback
            traceback.print_exc()

    # Train AmortizedLDA
    if not args.skip_lda:
        try:
            latent = train_amortized_lda(adata, args.n_latent, args.max_epochs, args.output_dir)
            results["amortized_lda"] = latent
        except Exception as e:
            print(f"AmortizedLDA training failed: {e}")
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
