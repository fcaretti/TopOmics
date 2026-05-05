#!/usr/bin/env python
"""
Training script for Mouse Brain Spatial Multiome dataset (RNA + ATAC with spatial info).
Dataset: SpatialGlue Dataset10 Mouse Brain H3K27me3

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
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from topomics.models.amortizedLDA import MultimodalAmortizedLDA

warnings.filterwarnings('ignore', message='.*was not registered in the param store.*')
warnings.filterwarnings('ignore', message='.*Found plate statements in guide but not model.*')


def parse_args():
    parser = argparse.ArgumentParser(description="Train topic model on Mouse Brain Spatial Multiome")
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
        default="universal",
        choices=["equal", "universal", "cell"],
        help="Aggregation strategy for modalities (default: universal)"
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
        default=100,
        help="Maximum training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--gcn_n_layers",
        type=int,
        default=1,
        help="Number of GCN layers for spatial encoder (default: 1)"
    )
    parser.add_argument(
        "--gcn_layers_type",
        type=str,
        default="GATv2Conv",
        choices=["GATv2Conv","GCNConv"],
        help="Type of Graph Convolution"
    )
    parser.add_argument(
        "--gcn_alpha",
        type=float,
        default=0.2,
        help="GCN alpha parameter for skip connection (default: 0.2). "
             "0 = neighbors only, 1 = self only."
    )
    parser.add_argument(
        "--fixed_alpha",
        action="store_true",
        help="Fix alpha (don't learn it during training)"
    )
    parser.add_argument(
        "--meanfield",
        action="store_true",
        help="Use TraceMeanField_ELBO instead of Trace_ELBO (analytic KL, lower variance gradients)"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8). Set to 1.0 to train on all cells."
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=5,
        help="Number of spatial neighbors for graph construction (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/topomics_models/mouse_brain_spatial",
        help="Output directory for model and plots"
    )
    return parser.parse_args()


def load_data(n_neighbors=5):
    """Load and preprocess Mouse Brain Spatial Multiome data."""
    adata_atac = sc.read_h5ad('/data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_peaks_normalized.h5ad')
    adata_rna = sc.read_h5ad('/data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_RNA.h5ad')

    # Binarize ATAC data
    X = adata_atac.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.ones_like(X.data)
        X.eliminate_zeros()
        adata_atac.layers["binary"] = X
    else:
        adata_atac.layers["binary"] = (X != 0).astype(np.float32)

    # Create MuData
    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    # Build spatial neighbor graph from RNA coordinates
    sc.pp.neighbors(
        mdata.mod["rna"],
        use_rep="spatial",
        n_neighbors=n_neighbors,
        metric="euclidean",
        key_added="spatial",
    )

    # Store spatial connectivities for all modalities
    mdata.obsp["spatial_connectivities"] = mdata.mod["rna"].obsp["spatial_connectivities"]
    mdata.mod["atac"].obsp["spatial_connectivities"] = mdata.mod["rna"].obsp["spatial_connectivities"]
    mdata.mod["atac"].obsp["spatial_distances"] = mdata.mod["rna"].obsp["spatial_distances"]

    return mdata


def create_model(mdata, args):
    """Create the topic model with specified hyperparameters."""
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        layer_dict={"rna": None, "atac": 'binary'},
        spatial_key="spatial_connectivities",
        n_topics=args.n_topics,
        likelihoods=["gamma_poisson", "bernoulli"],
        weight_mode=args.weight_mode,
        aggregation_type=args.aggregation_type,
        att_dim=args.att_dim,
        cell_topic_prior=1/args.n_topics,
        gcn_n_layers=args.gcn_n_layers,
        gcn_conv_type=args.gcn_layers_type,
        gcn_alpha_init=args.gcn_alpha,
        gcn_use_learned_alpha=not args.fixed_alpha,
        kl_weight=1,
        topic_feature_prior_type=args.feature_prior_type,
        learnable_dispersion=args.learnable_dispersion,
        global_dispersion=args.global_dispersion,
    )

    return model


def train_model(model, args):
    """Train the model."""
    plan_kwargs = {"optim_kwargs": {"lr": 1e-2}}
    if args.meanfield:
        from pyro.infer import TraceMeanField_ELBO
        plan_kwargs["loss_fn"] = TraceMeanField_ELBO()
    else:
        from pyro.infer import Trace_ELBO
        plan_kwargs["loss_fn"] = Trace_ELBO()
    val_size = 1.0 - args.train_size if args.train_size < 1.0 else 0.0
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        validation_size=val_size,
        plan_kwargs=plan_kwargs,
    )
    return model


def print_diagnostics(model):
    """Print learned model diagnostics: skip connection, modality weights, background."""
    import torch

    print("\n" + "=" * 70)
    print("MODEL DIAGNOSTICS")
    print("=" * 70)

    # Skip connection alpha
    gcn_encoders = getattr(model.module.guide, 'gcn_encoders', None)
    if gcn_encoders is not None:
        for i, enc in enumerate(gcn_encoders):
            print(f"  GCN encoder {i} skip connection alpha: {enc.alpha:.4f}")

    # Modality weights
    weight_mode = model.module.model.weight_mode if hasattr(model.module.model, 'weight_mode') else "unknown"
    print(f"  Weight mode: {weight_mode}")
    weights = model.get_modality_weights()
    if weight_mode == "universal":
        for mod_name in weights.columns:
            print(f"    {mod_name} weight: {weights[mod_name].iloc[0]:.4f}")
    elif weight_mode == "cell":
        for mod_name in weights.columns:
            w = weights[mod_name].values
            print(f"    {mod_name} weight: mean={w.mean():.4f}, std={w.std():.4f}, "
                  f"min={w.min():.4f}, max={w.max():.4f}")
    else:
        for mod_name in weights.columns:
            print(f"    {mod_name} weight: {weights[mod_name].iloc[0]:.4f}")

    # Feature background
    use_bg = getattr(model.module.model, 'use_feature_background', False)
    learnable_bg = getattr(model.module.model, 'learnable_bg', False)
    print(f"  Feature background: enabled={use_bg}, learnable={learnable_bg}")

    # Learnable dispersion
    learnable_disp = getattr(model.module.guide, 'learnable_dispersion', False)
    print(f"  Learnable dispersion: {learnable_disp}")

    print("=" * 70)


def save_results(model, mdata, output_dir, args):
    """Save the model and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model")
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Get latent representation
    theta = model.get_latent_representation(batch_size=256)

    # Save latent representation for easy loading (avoids GCN architecture issues)
    np.save(os.path.join(output_dir, "latent_representation.npy"), theta.values)
    print(f"Latent representation saved to: {os.path.join(output_dir, 'latent_representation.npy')}")

    # Add to mdata
    mdata.obsm["X_topic"] = theta.values - 1
    mdata.obs["top_topic"] = theta.idxmax(axis=1)
    mdata['rna'].obsm['X_topic'] = theta.values - 1

    # Create topic-based clustering
    sc.pp.neighbors(mdata['rna'], metric='cosine', use_rep='X_topic', key_added='topic_neighbors')
    sc.tl.umap(mdata['rna'], neighbors_key='topic_neighbors', key_added='topic_umap')
    sc.tl.leiden(mdata['rna'], neighbors_key='topic_neighbors')
    mdata.obs['leiden'] = mdata['rna'].obs['leiden']

    # Training curve (per-cell normalized)
    # Trace_ELBO with pyro.plate scales loss by n_obs, so divide by n_cells to get per-cell
    n_train_cells = int(mdata.n_obs * args.train_size)
    n_val_cells = mdata.n_obs - n_train_cells
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.history['elbo_train'] / n_train_cells, label='Train ELBO (per cell)')
    if 'elbo_val' in model.history:
        ax.plot(model.history['elbo_val'] / n_val_cells, label='Val ELBO (per cell)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO / cell')
    ax.set_title('Training Curve')
    ax.legend()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Topic UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    mu.pl.embedding(mdata, basis='rna:topic_umap', color='leiden', ax=ax, show=False)
    plt.savefig(os.path.join(output_dir, "umap_leiden.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Spatial plot colored by leiden clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    mu.pl.embedding(mdata, basis="rna:spatial", color="leiden", ax=ax, show=False)
    plt.savefig(os.path.join(output_dir, "spatial_leiden.png"), dpi=150, bbox_inches='tight')
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

    diversity_rna = model.get_topic_diversity(modality='rna')
    diversity_atac = model.get_topic_diversity(modality='atac')
    metrics['diversity_rna'] = diversity_rna
    metrics['diversity_atac'] = diversity_atac

    # Modality weights
    weights = model.get_modality_weights()
    metrics['mean_rna_weight'] = weights['rna'].mean()
    metrics['mean_atac_weight'] = weights['atac'].mean()

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
    alpha_str = f"alpha{args.gcn_alpha}" + ("_fixed" if args.fixed_alpha else "_learned")
    hyperparam_str = f"prior_{args.feature_prior_type}_weight_{args.weight_mode}_{alpha_str}"
    if args.gcn_n_layers != 1:
        hyperparam_str += f"_gcn{args.gcn_n_layers}"
    if args.n_neighbors != 5:
        hyperparam_str += f"_nn{args.n_neighbors}"
    if args.learnable_dispersion:
        hyperparam_str += f"_learnable_disp"
        if args.global_dispersion:
            hyperparam_str += "_global"
        else:
            hyperparam_str += "_pergene"
    if args.train_size >= 1.0:
        hyperparam_str += "_allcells"
    output_dir = os.path.join(args.output_dir, hyperparam_str)

    print("=" * 70)
    print("Mouse Brain Spatial Multiome Training")
    print("=" * 70)
    print(f"Feature prior type: {args.feature_prior_type}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"GCN layers: {args.gcn_n_layers}")
    print(f"Learnable dispersion: {args.learnable_dispersion}")
    print(f"Global dispersion: {args.global_dispersion}")
    print(f"GCN alpha: {args.gcn_alpha} ({'fixed' if args.fixed_alpha else 'learned'})")
    print(f"N neighbors: {args.n_neighbors}")
    print(f"Loss: {'TraceMeanField_ELBO' if args.meanfield else 'Trace_ELBO'}")
    print(f"N topics: {args.n_topics}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    print("\nLoading data...")
    mdata = load_data(n_neighbors=args.n_neighbors)
    print(f"Loaded MuData with {mdata.n_obs} spots (n_neighbors={args.n_neighbors})")

    print("\nCreating model...")
    model = create_model(mdata, args)
    print(f"Spatial mode enabled: {model.spatial}")
    print(f"Using GCN: {model.module.guide.use_gcn}")

    print("\nTraining model...")
    train_model(model, args)

    print_diagnostics(model)

    print("\nSaving results...")
    save_results(model, mdata, output_dir, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
