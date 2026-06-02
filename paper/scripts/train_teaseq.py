#!/usr/bin/env python
"""
Training script for TEA-seq dataset (RNA + ATAC + Protein).

Hyperparameters configurable via command line:
- feature_prior_type: "logistic_normal" or "horseshoe"
- weight_mode: "equal", "universal", or "cell" (aggregation strategy)
- learnable_dispersion: whether to learn dispersion
- global_dispersion: global vs per-gene dispersion
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import anndata as ad

from topomics import MultimodalAmortizedLDA

warnings.filterwarnings('ignore', message='.*was not registered in the param store.*')
warnings.filterwarnings('ignore', message='.*Found plate statements in guide but not model.*')


def parse_args():
    parser = argparse.ArgumentParser(description="Train topic model on TEA-seq dataset")
    parser.add_argument(
        "--feature_prior_type",
        type=str,
        default="logistic_normal",
        choices=["logistic_normal", "horseshoe"],
        help="Feature prior type (default: logistic_normal)"
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="cell",
        choices=["equal", "universal", "cell"],
        help="Aggregation strategy for modalities (default: cell)"
    )
    parser.add_argument(
        "--likelihood_weight_mode",
        type=str,
        default="none",
        choices=["none", "inverse_features", "sqrt_inverse_features"],
        help="Rescale per-modality likelihoods (default: none)"
    )
    parser.add_argument(
        "--likelihood_weight_ref",
        type=str,
        default="mean",
        choices=["mean", "median", "max"],
        help="Reference feature count for likelihood rescaling (default: mean)"
    )
    parser.add_argument(
        "--learnable_dispersion",
        action="store_true",
        help="Learn dispersion parameters (default: False)"
    )
    parser.add_argument(
        "--global_dispersion",
        action="store_true",
        help="Use global dispersion instead of per-gene (default: False)"
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="moe",
        choices=["moe", "attention"],
        help="Aggregation type for multimodal (default: moe)"
    )
    parser.add_argument(
        "--att_dim",
        type=int,
        default=16,
        help="Attention projection dimension (default: 16)"
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=10,
        help="Number of topics (default: 10)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=500,
        help="Maximum training epochs (default: 1000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/teaseq",
        help="Output directory for model and plots"
    )
    return parser.parse_args()


DATA_PATH = "/data/GSE158013/GSM5123951_celltypist.h5mu"


def _ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()


def _find_true_umap(mdata):
    if "X_umap" in mdata.obsm:
        return mdata.obsm["X_umap"], "mdata"
    for mod_name in ("rna", "atac", "prot"):
        if mod_name in mdata.mod and "X_umap" in mdata.mod[mod_name].obsm:
            return mdata.mod[mod_name].obsm["X_umap"], mod_name
    return None, None


def load_data():
    """Load and preprocess TEA-seq data (CellTypist annotated)."""
    mdata = mu.read_h5mu(DATA_PATH)

    _ensure_counts_layer(mdata.mod["rna"])
    _ensure_counts_layer(mdata.mod["atac"])
    _ensure_counts_layer(mdata.mod["prot"])

    # Binarize ATAC data
    mdata.mod['atac'].layers['counts'] = (mdata.mod['atac'].layers['counts'] > 0).astype(int)

    # Filter to highly variable genes
    sc.pp.highly_variable_genes(mdata.mod['rna'], n_top_genes=2000, flavor='seurat_v3', layer='counts')
    mdata.mod['rna'] = mdata.mod['rna'][:, mdata.mod['rna'].var['highly_variable']].copy()

    sc.pp.highly_variable_genes(mdata.mod['atac'], n_top_genes=10000, flavor='seurat_v3', layer='counts')
    mdata.mod['atac'] = mdata.mod['atac'][:, mdata.mod['atac'].var['highly_variable']].copy()

    mdata.update()

    return mdata


def create_model(mdata, args):
    """Create the topic model with specified hyperparameters."""
    model = MultimodalAmortizedLDA.from_data(
        mdata,
        modalities=["rna", "atac", "prot"],
        n_topics=args.n_topics,
        likelihoods=["gamma_poisson", "bernoulli", "gamma_poisson"],
        layers='counts',
        n_hidden=64,
        cell_topic_prior=1/args.n_topics,
        weight_mode=args.weight_mode,
        aggregation_type=args.aggregation_type,
        att_dim=args.att_dim,
        likelihood_weight_mode=args.likelihood_weight_mode,
        likelihood_weight_ref=args.likelihood_weight_ref,
        normalize_encoder_inputs=True,
        topic_feature_prior_type=args.feature_prior_type,
        learnable_dispersion=args.learnable_dispersion,
        global_dispersion=args.global_dispersion,
    )
    return model


def train_model(model, args):
    """Train the model."""
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=0.8,
        validation_size=0.2,
        log_every_n_steps=1,
    )
    return model


def save_results(model, mdata, output_dir):
    """Save the model and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model")
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Get latent representation
    adata_concat = mdata.uns["_flattened_ann_data"]
    theta = model.get_latent_representation(adata_concat, batch_size=mdata.n_obs)
    np.save(os.path.join(output_dir, "latent_representation.npy"), theta.values)
    print(f"Latent representation saved to: {os.path.join(output_dir, 'latent_representation.npy')}")

    # Run Leiden clustering and UMAP on topic space
    topic_X = theta.values - 1 / theta.values.shape[1]
    topic_adata = ad.AnnData(np.asarray(topic_X))
    topic_adata.obs = mdata.obs.copy()
    sc.pp.neighbors(topic_adata, n_neighbors=15, metric="cosine")
    sc.tl.leiden(topic_adata, key_added="topic_leiden")
    sc.tl.umap(topic_adata, min_dist=0.3)
    mdata.obsm["X_topic"] = topic_X
    mdata.obs["topic_leiden"] = topic_adata.obs["topic_leiden"].values

    # Training curve (per-cell normalized)
    n_train = int(mdata.n_obs * 0.8)
    n_val = mdata.n_obs - n_train
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.history['elbo_train'] / n_train, label='Train ELBO (per cell)')
    ax.plot(model.history['elbo_val'] / n_val, label='Val ELBO (per cell)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO / cell')
    ax.set_title('Training Curve')
    ax.legend()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # UMAP colored by topic clusters (prefer true UMAP if available)
    true_umap, true_umap_source = _find_true_umap(mdata)
    if true_umap is not None and np.asarray(true_umap).shape[0] == topic_adata.n_obs:
        umap_adata = ad.AnnData(np.zeros((topic_adata.n_obs, 1)))
        umap_adata.obs["topic_leiden"] = topic_adata.obs["topic_leiden"].values
        umap_adata.obsm["X_umap"] = np.asarray(true_umap)
        title = f"UMAP (true, {true_umap_source}) colored by topic clusters"
    else:
        umap_adata = topic_adata
        title = "UMAP (topic space) colored by topic clusters"

    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        umap_adata,
        color="topic_leiden",
        frameon=False,
        s=20,
        title=title,
        show=False,
        ax=ax,
        legend_loc="right margin",
    )
    plt.savefig(os.path.join(output_dir, "umap_topic_leiden.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # UMAP in topic space colored by cell type
    cell_type_key = None
    if "celltypist_label" in topic_adata.obs.columns:
        cell_type_key = "celltypist_label"
    elif "celltypist_label" in mdata.mod["rna"].obs.columns:
        topic_adata.obs["celltypist_label"] = mdata.mod["rna"].obs["celltypist_label"].values
        cell_type_key = "celltypist_label"

    if cell_type_key is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            topic_adata,
            color=cell_type_key,
            frameon=False,
            s=20,
            title="UMAP (topic space) colored by cell type",
            show=False,
            ax=ax,
            legend_loc="right margin",
        )
        plt.savefig(os.path.join(output_dir, "umap_topic_cell_type.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print("CellTypist labels not found; skipping topic-space UMAP colored by cell type.")

    # Topic distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    means = theta.mean(axis=0)
    stds = theta.std(axis=0)
    ax.bar(np.arange(len(means)), means.values, yerr=stds.values, capsize=3)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_ylabel("Topic proportion")
    ax.set_title("Global topic distribution (mean +/- std)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topic_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Metrics
    metrics = {}
    metrics['perplexity'] = model.get_perplexity()
    metrics['entropy'] = model.get_entropy(normalised=True)
    metrics['diversity'] = model.get_topic_diversity()

    # Per-modality metrics
    perplexity_per_mod = model.get_perplexity_per_modality()
    for mod_name, ppl in perplexity_per_mod.items():
        metrics[f'perplexity_{mod_name}'] = ppl

    diversity_rna = model.get_topic_diversity(modality='rna')
    diversity_atac = model.get_topic_diversity(modality='atac')
    diversity_prot = model.get_topic_diversity(modality='prot')
    metrics['diversity_rna'] = diversity_rna
    metrics['diversity_atac'] = diversity_atac
    metrics['diversity_prot'] = diversity_prot

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"Metrics saved to: {os.path.join(output_dir, 'metrics.csv')}")

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")
    print(f"Diversity: {metrics['diversity']:.4f}")
    for mod_name, ppl in perplexity_per_mod.items():
        print(f"  {mod_name} perplexity: {ppl:.4f}")


def main():
    args = parse_args()

    # Create output directory with hyperparameter info
    hyperparam_str = f"prior_{args.feature_prior_type}_weight_{args.weight_mode}"
    if args.likelihood_weight_mode != "none" or args.likelihood_weight_ref != "mean":
        hyperparam_str += f"_likewt_{args.likelihood_weight_mode}"
        if args.likelihood_weight_ref != "mean":
            hyperparam_str += f"_{args.likelihood_weight_ref}"
    if args.learnable_dispersion:
        hyperparam_str += f"_learnable_disp"
        if args.global_dispersion:
            hyperparam_str += "_global"
        else:
            hyperparam_str += "_pergene"

    output_dir = os.path.join(args.output_dir, hyperparam_str)

    print("=" * 70)
    print("TEA-seq Training")
    print("=" * 70)
    print(f"Feature prior type: {args.feature_prior_type}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Likelihood weight mode: {args.likelihood_weight_mode}")
    print(f"Likelihood weight ref: {args.likelihood_weight_ref}")
    print(f"Learnable dispersion: {args.learnable_dispersion}")
    print(f"Global dispersion: {args.global_dispersion}")
    print(f"N topics: {args.n_topics}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    print("\nLoading data...")
    mdata = load_data()
    print(f"Loaded MuData with {mdata.n_obs} cells")

    print("\nCreating model...")
    model = create_model(mdata, args)

    print("\nTraining model...")
    train_model(model, args)

    print("\nSaving results...")
    save_results(model, mdata, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
