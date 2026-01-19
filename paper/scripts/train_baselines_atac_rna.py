#!/usr/bin/env python
"""
Training script for baseline models (MultiVI, MOFA+, GLUE) on ATAC+RNA lymphoma data.

Usage:
    python train_baselines_atac_rna.py --n_latent 10 --max_epochs 300
    python train_baselines_atac_rna.py --skip_glue  # Skip GLUE if issues arise
"""

import argparse
import os
import warnings

import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# Data path
DATA_PATH = "/data/nelkazwi/share-topic/lymphoma_data/mdata_lymphoma.h5mu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models on ATAC+RNA lymphoma dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the input MuData file",
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
        help="Maximum training epochs (default: 300)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/atac_rna/baselines",
        help="Output directory for models",
    )
    parser.add_argument(
        "--n_top_rna",
        type=int,
        default=2000,
        help="Number of RNA HVGs to keep (default: 2000)",
    )
    parser.add_argument(
        "--n_top_atac",
        type=int,
        default=10000,
        help="Number of ATAC HVGs to keep (default: 10000)",
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
        "--skip_glue",
        action="store_true",
        help="Skip GLUE training",
    )
    return parser.parse_args()


def ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()


def binarize_atac(adata_atac):
    X = adata_atac.layers["counts"] if "counts" in adata_atac.layers else adata_atac.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.ones_like(X.data)
        X.eliminate_zeros()
    else:
        X = (X > 0).astype(np.int8)
    adata_atac.layers["counts"] = X
    adata_atac.X = X.copy()


def load_data(data_path, n_top_rna, n_top_atac):
    """Load and preprocess ATAC+RNA lymphoma data."""
    print("Loading ATAC+RNA lymphoma data...")
    mdata = mu.read_h5mu(data_path)

    # Ensure counts layers exist
    ensure_counts_layer(mdata.mod["rna"])
    ensure_counts_layer(mdata.mod["atac"])

    # Binarize ATAC data
    binarize_atac(mdata.mod["atac"])

    # Filter genes/regions with zero counts
    rna_total_counts = np.array(mdata.mod["rna"].layers["counts"].sum(axis=0)).flatten()
    rna_nonzero = rna_total_counts > 0
    mdata.mod["rna"] = mdata.mod["rna"][:, rna_nonzero]

    atac_total_counts = np.array(mdata.mod["atac"].layers["counts"].sum(axis=0)).flatten()
    atac_nonzero = atac_total_counts > 0
    mdata.mod["atac"] = mdata.mod["atac"][:, atac_nonzero]

    # Filter to highly variable genes for RNA
    n_rna = min(n_top_rna, mdata.mod["rna"].n_vars)
    sc.pp.highly_variable_genes(
        mdata.mod["rna"], n_top_genes=n_rna, flavor="seurat_v3", layer="counts"
    )
    mdata.mod["rna"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()

    # Filter to highly variable peaks for ATAC
    n_atac = min(n_top_atac, mdata.mod["atac"].n_vars)
    sc.pp.highly_variable_genes(
        mdata.mod["atac"], n_top_genes=n_atac, flavor="seurat_v3", layer="counts"
    )
    mdata.mod["atac"] = mdata.mod["atac"][:, mdata.mod["atac"].var["highly_variable"]].copy()

    # Sync MuData axes after feature filtering
    mdata.update()

    print(f"  RNA: {mdata.mod['rna'].shape}")
    print(f"  ATAC: {mdata.mod['atac'].shape}")

    return mdata


# =============================================================================
# MultiVI (RNA + ATAC)
# =============================================================================
def train_multivi(mdata, n_latent, max_epochs, output_dir):
    """Train MultiVI model on RNA + ATAC."""
    import scvi

    print("\n" + "=" * 70)
    print("Training MultiVI (RNA + ATAC)")
    print("=" * 70)

    mdata_multivi = mdata.copy()
    mdata_multivi.mod["rna"].X = mdata_multivi.mod["rna"].layers["counts"].copy()
    mdata_multivi.mod["atac"].X = mdata_multivi.mod["atac"].layers["counts"].copy()

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

    model = scvi.model.MULTIVI(
        mdata_multivi,
        n_latent=n_latent,
        n_hidden=128,
    )

    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
        early_stopping=True,
    )

    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")

    model_path = os.path.join(output_dir, "multivi")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    np.save(os.path.join(output_dir, "latent_multivi.npy"), latent)

    history = model.history
    if isinstance(history, pd.DataFrame):
        history_df = history
    else:
        try:
            history_df = pd.DataFrame(history)
        except ValueError:
            history_df = pd.DataFrame(
                {k: [v] if np.isscalar(v) else v for k, v in history.items()}
            )
    history_df.to_csv(os.path.join(output_dir, "multivi_history.csv"), index=False)

    return latent, mdata_multivi


# =============================================================================
# MOFA+
# =============================================================================
def train_mofa(mdata, n_latent, output_dir):
    """Train MOFA+ model."""
    print("\n" + "=" * 70)
    print("Training MOFA+ (RNA + ATAC)")
    print("=" * 70)

    mdata_mofa = mdata.copy()

    mdata_mofa.mod["rna"].X = mdata_mofa.mod["rna"].layers["counts"].copy()
    sc.pp.normalize_total(mdata_mofa.mod["rna"], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod["rna"])
    sc.pp.scale(mdata_mofa.mod["rna"])

    mdata_mofa.mod["atac"].X = mdata_mofa.mod["atac"].layers["counts"].copy()
    sc.pp.normalize_total(mdata_mofa.mod["atac"], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod["atac"])
    sc.pp.scale(mdata_mofa.mod["atac"])

    print("Training...")
    mu.tl.mofa(
        mdata_mofa,
        n_factors=n_latent,
        convergence_mode="medium",
        use_obs="intersection",
    )

    latent = mdata_mofa.obsm["X_mofa"]
    print(f"Latent shape: {latent.shape}")

    mdata_path = os.path.join(output_dir, "mdata_mofa.h5mu")
    mdata_mofa.write(mdata_path)
    print(f"MuData saved to: {mdata_path}")

    np.save(os.path.join(output_dir, "latent_mofa.npy"), latent)

    return latent, mdata_mofa


# =============================================================================
# GLUE (RNA + ATAC with genomic guidance)
# =============================================================================
def train_glue(mdata, n_latent, max_epochs, output_dir):
    """Train GLUE model on RNA + ATAC."""
    import scglue

    print("\n" + "=" * 70)
    print("Training GLUE (RNA + ATAC)")
    print("=" * 70)

    rna = mdata.mod["rna"].copy()
    atac = mdata.mod["atac"].copy()

    rna.X = rna.layers["counts"].copy()
    atac.X = atac.layers["counts"].copy()

    has_coords = "chromStart" in atac.var.columns or "start" in atac.var.columns
    if not has_coords:
        print("ATAC peaks lack genomic coordinates.")
        print("Attempting to parse from var_names (format: chr:start-end or chr_start_end)...")
        try:
            coords = atac.var_names.str.extract(r"(chr[^:_]+)[:\\-_](\\d+)[:\\-_](\\d+)")
            if coords.isna().any().any():
                raise ValueError("Could not parse coordinates")
            atac.var["chrom"] = coords[0].values
            atac.var["chromStart"] = coords[1].astype(int).values
            atac.var["chromEnd"] = coords[2].astype(int).values
            has_coords = True
            print("Successfully parsed coordinates from var_names")
        except Exception as exc:
            print(f"Failed to parse coordinates: {exc}")
            print("GLUE training skipped - genomic coordinates required")
            return None, None

    print("Preprocessing for GLUE...")
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, use_highly_variable=False)

    atac.layers["counts"] = atac.X.copy()
    scglue.data.lsi(atac, n_components=100)

    print("Building guidance graph...")
    if "chromStart" not in rna.var.columns:
        print("RNA genes lack genomic coordinates.")
        print("Attempting to get gene coordinates from Ensembl/UCSC...")
        try:
            scglue.data.get_gene_annotation(
                rna,
                gtf="http://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
                gtf_by="gene_name",
            )
            print("Successfully added gene coordinates")
        except Exception as exc:
            print(f"Could not get gene coordinates: {exc}")
            print("Using correlation-based guidance graph instead...")

    try:
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        print(
            f"Guidance graph: {guidance.number_of_nodes()} nodes, "
            f"{guidance.number_of_edges()} edges"
        )
    except Exception as exc:
        print(f"Could not build genomic guidance graph: {exc}")
        print("GLUE training skipped")
        return None, None

    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=False, use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=False, use_layer="counts", use_rep="X_lsi"
    )

    print("Training GLUE model...")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance,
        fit_kws={"directory": os.path.join(output_dir, "glue_checkpoints")},
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    latent = rna.obsm["X_glue"]
    print(f"Latent shape: {latent.shape}")

    model_path = os.path.join(output_dir, "glue_model")
    os.makedirs(model_path, exist_ok=True)
    glue.save(os.path.join(model_path, "glue.dill"))
    print(f"Model saved to: {model_path}")

    np.save(os.path.join(output_dir, "latent_glue.npy"), latent)

    rna.write(os.path.join(output_dir, "rna_glue.h5ad"))
    atac.write(os.path.join(output_dir, "atac_glue.h5ad"))

    return latent, rna


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ATAC+RNA Baseline Models Training")
    print("=" * 70)
    print(f"N latent dimensions: {args.n_latent}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    mdata = load_data(args.data_path, args.n_top_rna, args.n_top_atac)

    results = {}

    if not args.skip_multivi:
        try:
            latent, _ = train_multivi(
                mdata, args.n_latent, args.max_epochs, args.output_dir
            )
            results["multivi"] = latent
        except Exception as exc:
            print(f"MultiVI training failed: {exc}")
            import traceback

            traceback.print_exc()

    if not args.skip_mofa:
        try:
            latent, _ = train_mofa(mdata, args.n_latent, args.output_dir)
            results["mofa"] = latent
        except Exception as exc:
            print(f"MOFA+ training failed: {exc}")
            import traceback

            traceback.print_exc()

    if not args.skip_glue:
        try:
            latent, _ = train_glue(
                mdata, args.n_latent, args.max_epochs, args.output_dir
            )
            if latent is not None:
                results["glue"] = latent
        except Exception as exc:
            print(f"GLUE training failed: {exc}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for name, latent in results.items():
        print(f"  {name}: latent shape = {latent.shape}")
    print(f"\nAll results saved to: {args.output_dir}")

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
