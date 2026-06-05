#!/usr/bin/env python
"""
Training script for SCTM comparison dataset (squidpy Visium H&E).
This is a unimodal spatial RNA dataset.

Hyperparameters configurable via command line:
- feature_prior_type: "logistic_normal" or "horseshoe"
- learnable_dispersion: whether to learn dispersion
- global_dispersion: global vs per-gene dispersion

Note: weight_mode is not applicable for unimodal datasets.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from topomics.models.amortizedLDA import MultimodalAmortizedLDA

warnings.filterwarnings("ignore", message=".*was not registered in the param store.*")
warnings.filterwarnings("ignore", message=".*Found plate statements in guide but not model.*")


def parse_args():
    parser = argparse.ArgumentParser(description="Train topic model on SCTM comparison dataset")
    parser.add_argument(
        "--feature_prior_type",
        type=str,
        default="logistic_normal",
        choices=["logistic_normal", "horseshoe"],
        help="Feature prior type (default: logistic_normal)",
    )
    parser.add_argument(
        "--gcn_n_layers", type=int, default=1, help="Number of GCN layers for spatial encoder (default: 1)"
    )
    parser.add_argument(
        "--gcn_layers_type",
        type=str,
        default="GATv2Conv",
        choices=["GATv2Conv", "GCNConv"],
        help="Type of Graph Convolution",
    )
    parser.add_argument(
        "--gcn_alpha",
        type=float,
        default=0.2,
        help="GCN alpha parameter for skip connection (default: 0.2). 0 = neighbors only, 1 = self only.",
    )
    parser.add_argument("--fixed_alpha", action="store_true", help="Fix alpha (don't learn it during training)")
    parser.add_argument(
        "--learnable_dispersion", action="store_true", help="Learn dispersion parameters (default: False)"
    )
    parser.add_argument(
        "--global_dispersion", action="store_true", help="Use global dispersion instead of per-gene (default: False)"
    )
    parser.add_argument("--n_topics", type=int, default=10, help="Number of topics (default: 10)")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8). Set to 1.0 to train on all cells.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/topomics_models/sctm_comparison",
        help="Output directory for model and plots",
    )
    return parser.parse_args()


def load_data():
    """Load and preprocess SCTM comparison data (Visium H&E)."""
    adata = sq.datasets.visium_hne_adata()

    # Use raw counts
    adata.X = adata.raw.X
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic")

    return adata


def create_model(adata, args):
    """Create the topic model with specified hyperparameters."""
    model = MultimodalAmortizedLDA.from_data(
        adata,
        n_topics=args.n_topics,
        likelihoods=["gamma_poisson"],
        cell_topic_prior=1 / args.n_topics,
        spatial_keys="spatial_connectivities",
        gcn_n_layers=args.gcn_n_layers,
        gcn_conv_type=args.gcn_layers_type,
        gcn_alpha_init=args.gcn_alpha,
        gcn_use_learned_alpha=not args.fixed_alpha,
        kl_weight=1,
        use_feature_background=False,
        topic_feature_prior_type=args.feature_prior_type,
        learnable_dispersion=args.learnable_dispersion,
        global_dispersion=args.global_dispersion,
    )
    return model


def train_model(model, args):
    """Train the model."""
    val_size = 1.0 - args.train_size if args.train_size < 1.0 else 0.0
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        validation_size=val_size,
        log_every_n_steps=1,
        plan_kwargs={"optim_kwargs": {"lr": 1e-2}},
    )
    return model


def save_results(model, adata, output_dir):
    """Save the model and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model")
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Get latent representation
    theta = model.get_latent_representation(adata, batch_size=adata.n_obs)

    # Save latent representation for easy loading (avoids GCN architecture issues)
    np.save(os.path.join(output_dir, "latent_representation.npy"), theta.values)
    print(f"Latent representation saved to: {os.path.join(output_dir, 'latent_representation.npy')}")

    # Add to adata
    adata.obsm["X_topic"] = theta.values - 1
    adata.obs["top_topic"] = theta.idxmax(axis=1)

    # Create topic-based clustering
    sc.pp.neighbors(adata, metric="cosine", use_rep="X_topic", key_added="topic_neighbors")
    sc.tl.leiden(adata, neighbors_key="topic_neighbors", key_added="topic_clusters")
    sc.tl.umap(adata, neighbors_key="topic_neighbors")

    # Training curve (per-cell normalized)
    n_train = int(adata.n_obs * 0.8)
    n_val = adata.n_obs - n_train
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.history["elbo_train"] / n_train, label="Train ELBO (per cell)")
    ax.plot(model.history["elbo_val"] / n_val, label="Val ELBO (per cell)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO / cell")
    ax.set_title("Training Curve")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # UMAP colored by topic clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color="topic_clusters", frameon=False, ax=ax, show=False)
    plt.savefig(os.path.join(output_dir, "umap_topic_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Spatial plot colored by topic clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    sq.pl.spatial_scatter(adata, color=["topic_clusters"], ax=ax)
    plt.savefig(os.path.join(output_dir, "spatial_topic_clusters.png"), dpi=150, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "topic_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Metrics
    metrics = {}
    metrics["perplexity"] = model.get_perplexity()
    metrics["entropy"] = model.get_entropy(normalised=True)
    metrics["diversity"] = model.get_topic_diversity()

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


def main():
    args = parse_args()

    # Create output directory with hyperparameter info
    alpha_str = f"alpha{args.gcn_alpha}" + ("_fixed" if args.fixed_alpha else "_learned")
    hyperparam_str = f"prior_{args.feature_prior_type}_{alpha_str}"
    if args.gcn_n_layers != 1:
        hyperparam_str += f"_gcn{args.gcn_n_layers}"
    if args.learnable_dispersion:
        hyperparam_str += "_learnable_disp"
        if args.global_dispersion:
            hyperparam_str += "_global"
        else:
            hyperparam_str += "_pergene"
    if args.train_size >= 1.0:
        hyperparam_str += "_allcells"

    output_dir = os.path.join(args.output_dir, hyperparam_str)

    print("=" * 70)
    print("SCTM Comparison (Visium H&E) Training")
    print("=" * 70)
    print(f"Feature prior type: {args.feature_prior_type}")
    print(f"GCN layers: {args.gcn_n_layers}, type: {args.gcn_layers_type}")
    print(f"GCN alpha: {args.gcn_alpha} ({'fixed' if args.fixed_alpha else 'learned'})")
    print(f"Learnable dispersion: {args.learnable_dispersion}")
    print(f"Global dispersion: {args.global_dispersion}")
    print(f"N topics: {args.n_topics}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    print("\nLoading data...")
    adata = load_data()
    print(f"Loaded AnnData with {adata.n_obs} spots and {adata.n_vars} genes")

    print("\nCreating model...")
    model = create_model(adata, args)

    print("\nTraining model...")
    train_model(model, args)

    print("\nSaving results...")
    save_results(model, adata, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
