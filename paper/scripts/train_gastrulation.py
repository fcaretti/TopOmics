#!/usr/bin/env python
"""
Training script for scvelo gastrulation erythroid dataset (spliced + unspliced RNA).

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
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc

try:
    import scvelo as scv
except ImportError as err:
    raise ImportError(
        "scvelo is required for this script. Install with `pip install scvelo` "
        "or `uv pip install scvelo`."
    ) from err

from topomics.models.amortizedLDA import MultimodalAmortizedLDA

warnings.filterwarnings("ignore", message=".*was not registered in the param store.*")
warnings.filterwarnings("ignore", message=".*Found plate statements in guide but not model.*")


def parse_args():
    parser = argparse.ArgumentParser(description="Train topic model on gastrulation dataset")
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
        default=300,
        help="Maximum training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/gastrulation",
        help="Output directory for model and plots"
    )
    return parser.parse_args()


def load_data():
    """Load and preprocess gastrulation erythroid data."""
    adata = scv.datasets.gastrulation_erythroid()

    # Filter genes with non-zero counts in either layer
    spliced = adata.layers["spliced"]
    unspliced = adata.layers["unspliced"]

    mature_counts = np.asarray(spliced.sum(axis=0)).ravel()
    immature_counts = np.asarray(unspliced.sum(axis=0)).ravel()
    nonzero = (mature_counts > 0) | (immature_counts > 0)
    adata = adata[:, nonzero].copy()

    # Create separate AnnData objects for each modality
    adata_mature = adata.copy()
    adata_mature.X = adata.layers["spliced"]
    adata_mature.layers.clear()

    adata_immature = adata.copy()
    adata_immature.X = adata.layers["unspliced"]
    adata_immature.layers.clear()

    # Create MuData
    mdata = md.MuData({"mature": adata_mature, "immature": adata_immature})

    return mdata, adata


def create_model(mdata, args):
    """Create the topic model with specified hyperparameters."""
    model = MultimodalAmortizedLDA.from_data(
        mdata,
        modalities=["mature", "immature"],
        n_topics=args.n_topics,
        n_hidden=128,
        likelihoods=["gamma_poisson", "gamma_poisson"],
        weight_mode=args.weight_mode,
        aggregation_type=args.aggregation_type,
        att_dim=args.att_dim,
        cell_topic_prior=1/args.n_topics,
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
        validation_size=0.1,
        log_every_n_steps=10,
        plan_kwargs={"optim_kwargs": {"lr": 1e-3}},
    )
    return model


def save_results(model, mdata, adata, output_dir):
    """Save the model and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model")
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Get latent representation
    theta = model.get_latent_representation(batch_size=mdata.n_obs)

    # Add to adata
    adata.obsm["X_topic"] = theta.values - 1
    adata.obs["top_topic"] = theta.idxmax(axis=1).astype(str)

    # Create topic-based UMAP
    sc.pp.neighbors(
        adata,
        use_rep="X_topic",
        n_neighbors=30,
        metric="cosine",
        key_added="topic_neighbors",
    )
    sc.tl.umap(adata, neighbors_key="topic_neighbors", min_dist=0.3)
    sc.tl.leiden(adata, neighbors_key="topic_neighbors")
    adata.obsm["X_topic_umap"] = adata.obsm["X_umap"].copy()

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

    # Topic UMAP colored by stage and leiden
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.embedding(adata, basis="topic_umap", color="leiden", frameon=False, ax=axes[0], show=False)
    sc.pl.embedding(adata, basis="topic_umap", color="stage", frameon=False, ax=axes[1], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_topic.png"), dpi=150, bbox_inches='tight')
    plt.close()

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

    diversity_mature = model.get_topic_diversity(modality='mature')
    diversity_immature = model.get_topic_diversity(modality='immature')
    metrics['diversity_mature'] = diversity_mature
    metrics['diversity_immature'] = diversity_immature

    # Modality weights
    weights = model.get_modality_weights()
    metrics['mean_mature_weight'] = weights['mature'].mean()
    metrics['mean_immature_weight'] = weights['immature'].mean()

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
    if args.learnable_dispersion:
        hyperparam_str += f"_learnable_disp"
        if args.global_dispersion:
            hyperparam_str += "_global"
        else:
            hyperparam_str += "_pergene"

    output_dir = os.path.join(args.output_dir, hyperparam_str)

    print("=" * 70)
    print("Gastrulation Erythroid Training")
    print("=" * 70)
    print(f"Feature prior type: {args.feature_prior_type}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Learnable dispersion: {args.learnable_dispersion}")
    print(f"Global dispersion: {args.global_dispersion}")
    print(f"N topics: {args.n_topics}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    print("\nLoading data...")
    mdata, adata = load_data()
    print(f"Loaded MuData with {mdata.n_obs} cells")

    print("\nCreating model...")
    model = create_model(mdata, args)

    print("\nTraining model...")
    train_model(model, args)

    print("\nSaving results...")
    save_results(model, mdata, adata, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
