#!/usr/bin/env python
"""
Training script for Mouse Brain ATAC+RNA from the COSMOS benchmark (13932144.zip)
using Gaussian likelihoods.

This dataset has pre-processed continuous features:
  - RNA: 3000 genes, log-normalized
  - ATAC: 50 LSI dimensions
  - Ground truth: 9 brain regions (LayerName)

Since both modalities are continuous, we use Gaussian likelihood for each.
"""

import argparse
import os
import warnings

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from omics_topic.models.amortizedLDA import MultimodalAmortizedLDA

warnings.filterwarnings('ignore', message='.*was not registered in the param store.*')
warnings.filterwarnings('ignore', message='.*Found plate statements in guide but not model.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GaussianLDA on Mouse Brain COSMOS benchmark"
    )
    parser.add_argument("--feature_prior_type", type=str, default="logistic_normal",
                        choices=["logistic_normal", "horseshoe"])
    parser.add_argument("--weight_mode", type=str, default="universal",
                        choices=["equal", "universal", "cell"])
    parser.add_argument("--n_topics", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gcn_n_layers", type=int, default=1)
    parser.add_argument("--gcn_layers_type", type=str, default="GATv2Conv",
                        choices=["GATv2Conv", "GCNConv"])
    parser.add_argument("--gcn_alpha", type=float, default=0.2)
    parser.add_argument("--fixed_alpha", action="store_true")
    parser.add_argument("--meanfield", action="store_true",
                        help="Use TraceMeanField_ELBO instead of Trace_ELBO")
    parser.add_argument("--no_spatial", action="store_true",
                        help="Disable spatial graph (ablation: non-spatial baseline)")
    parser.add_argument("--data_path", type=str,
                        default="/tmp/mouse_brain_data/ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5")
    parser.add_argument("--output_dir", type=str,
                        default="/data/omics_topic_models/mouse_brain_gaussian")
    return parser.parse_args()


def load_data(data_path, use_spatial=True):
    """Load COSMOS benchmark h5 into MuData with spatial graph and ground truth."""
    with h5py.File(data_path, 'r') as f:
        cells = [c.decode() if isinstance(c, bytes) else c for c in f['Cell'][:]]
        genes = [g.decode() if isinstance(g, bytes) else g for g in f['Gene'][:]]
        layers = [l.decode() if isinstance(l, bytes) else l for l in f['LayerName'][:]]
        x_rna = f['X_RNA'][:]
        x_atac = f['X_ATAC'][:]
        pos = f['Pos'][:]

    obs = pd.DataFrame({'spatial_area': layers}, index=cells)

    adata_rna = ad.AnnData(
        X=x_rna.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=genes),
    )
    adata_rna.obsm['spatial'] = pos

    atac_features = [f'LSI_{i+1}' for i in range(x_atac.shape[1])]
    adata_atac = ad.AnnData(
        X=x_atac.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=atac_features),
    )
    adata_atac.obsm['spatial'] = pos

    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    if use_spatial:
        sc.pp.neighbors(adata_rna, use_rep='spatial', n_neighbors=5,
                        metric='euclidean', key_added='spatial')
        mdata.obsp["spatial_connectivities"] = adata_rna.obsp["spatial_connectivities"]
        mdata.mod["atac"].obsp["spatial_connectivities"] = adata_rna.obsp["spatial_connectivities"]
        mdata.mod["atac"].obsp["spatial_distances"] = adata_rna.obsp["spatial_distances"]

    print(f"Loaded: {mdata.n_obs} cells, RNA={x_rna.shape[1]} genes, ATAC={x_atac.shape[1]} LSI dims")
    print(f"Spatial areas: {len(set(layers))} regions")
    print(f"Spatial graph: {'enabled' if use_spatial else 'disabled'}")

    return mdata


def create_model(mdata, args):
    """Create GaussianLDA topic model."""
    spatial_key = "spatial_connectivities" if not args.no_spatial else None

    model_kwargs = dict(
        n_topics=args.n_topics,
        likelihoods=["gaussian", "gaussian"],
        weight_mode=args.weight_mode,
        cell_topic_prior=1 / args.n_topics,
        kl_weight=1,
        topic_feature_prior_type=args.feature_prior_type,
    )

    if spatial_key is not None:
        model_kwargs.update(
            gcn_n_layers=args.gcn_n_layers,
            gcn_conv_type=args.gcn_layers_type,
            gcn_alpha_init=args.gcn_alpha,
            gcn_use_learned_alpha=not args.fixed_alpha,
        )

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        layer_dict={"rna": None, "atac": None},
        spatial_key=spatial_key,
        **model_kwargs,
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

    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=0.8,
        validation_size=0.2,
        plan_kwargs=plan_kwargs,
    )
    return model


def evaluate_clustering(theta, labels, resolutions=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)):
    """Best ARI/NMI over Leiden resolutions."""
    adata_tmp = ad.AnnData(X=theta)
    sc.pp.neighbors(adata_tmp, use_rep='X', metric='cosine')

    best_ari, best_nmi, best_res = -1, -1, -1
    for res in resolutions:
        sc.tl.leiden(adata_tmp, resolution=res, key_added='leiden')
        ari = adjusted_rand_score(labels, adata_tmp.obs['leiden'])
        nmi = normalized_mutual_info_score(labels, adata_tmp.obs['leiden'])
        if ari > best_ari:
            best_ari, best_nmi, best_res = ari, nmi, res

    return best_ari, best_nmi, best_res


def print_diagnostics(model):
    """Print learned model diagnostics."""
    print("\n" + "=" * 70)
    print("MODEL DIAGNOSTICS")
    print("=" * 70)

    gcn_encoders = getattr(model.module.guide, 'gcn_encoders', None)
    if gcn_encoders is not None:
        for i, enc in enumerate(gcn_encoders):
            print(f"  GCN encoder {i} skip connection alpha: {enc.alpha:.4f}")

    weights = model.get_modality_weights()
    for mod_name in weights.columns:
        w = weights[mod_name]
        print(f"  {mod_name} weight: mean={w.mean():.4f}, std={w.std():.4f}")

    print("=" * 70)


def save_results(model, mdata, output_dir):
    """Save model, metrics, and plots."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model")
    model.save(model_path, overwrite=True)

    theta = model.get_latent_representation(batch_size=mdata.n_obs)
    np.save(os.path.join(output_dir, "latent_representation.npy"), theta.values)

    mdata.obsm["X_topic"] = theta.values - 1
    mdata.obs["top_topic"] = theta.idxmax(axis=1)
    mdata['rna'].obsm['X_topic'] = theta.values - 1

    # Evaluate against ground truth
    labels = mdata['rna'].obs['spatial_area'].values
    ari, nmi, best_res = evaluate_clustering(theta.values, labels)

    # Leiden clustering
    sc.pp.neighbors(mdata['rna'], metric='cosine', use_rep='X_topic', key_added='topic_neighbors')
    sc.tl.umap(mdata['rna'], neighbors_key='topic_neighbors', key_added='topic_umap')
    sc.tl.leiden(mdata['rna'], neighbors_key='topic_neighbors', resolution=best_res)
    mdata.obs['leiden'] = mdata['rna'].obs['leiden']

    # Metrics
    metrics = {
        'ARI': ari,
        'NMI': nmi,
        'best_leiden_resolution': best_res,
        'perplexity': model.get_perplexity(),
        'entropy': model.get_entropy(normalised=True),
        'diversity': model.get_topic_diversity(),
    }
    for mod_name, ppl in model.get_perplexity_per_modality().items():
        metrics[f'perplexity_{mod_name}'] = ppl
    weights = model.get_modality_weights()
    metrics['mean_rna_weight'] = weights['rna'].mean()
    metrics['mean_atac_weight'] = weights['atac'].mean()

    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # --- Plots ---

    # Training curve (per-cell normalized)
    n_train = int(mdata.n_obs * 0.8)
    n_val = mdata.n_obs - n_train
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.history['elbo_train'] / n_train, label='Train ELBO (per cell)')
    ax.plot(model.history['elbo_val'] / n_val, label='Val ELBO (per cell)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ELBO / cell'); ax.set_title('Training Curve')
    ax.legend()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Topic UMAP: ground truth vs Leiden
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    mu.pl.embedding(mdata, basis='rna:topic_umap', color='rna:spatial_area', ax=axes[0], show=False)
    axes[0].set_title('Topic UMAP - Ground truth areas')
    mu.pl.embedding(mdata, basis='rna:topic_umap', color='leiden', ax=axes[1], show=False)
    axes[1].set_title(f'Topic UMAP - Leiden (ARI={ari:.3f}, NMI={nmi:.3f})')
    plt.savefig(os.path.join(output_dir, "umap_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Spatial: ground truth vs Leiden
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    mu.pl.embedding(mdata, basis='rna:spatial', color='rna:spatial_area', ax=axes[0], show=False)
    axes[0].set_title('Spatial - Ground truth areas')
    mu.pl.embedding(mdata, basis='rna:spatial', color='leiden', ax=axes[1], show=False)
    axes[1].set_title(f'Spatial - Leiden (ARI={ari:.3f})')
    plt.savefig(os.path.join(output_dir, "spatial_comparison.png"), dpi=150, bbox_inches='tight')
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

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")
    print(f"Diversity: {metrics['diversity']:.4f}")
    for mod_name, ppl in model.get_perplexity_per_modality().items():
        print(f"  {mod_name} perplexity: {ppl:.4f}")
    print("=" * 70)


def main():
    args = parse_args()

    # Build output directory name
    alpha_str = f"alpha{args.gcn_alpha}" + ("_fixed" if args.fixed_alpha else "_learned")
    hyperparam_str = f"prior_{args.feature_prior_type}_weight_{args.weight_mode}_{alpha_str}"
    if args.gcn_n_layers != 1:
        hyperparam_str += f"_gcn{args.gcn_n_layers}"
    if args.meanfield:
        hyperparam_str += "_meanfield"
    if args.no_spatial:
        hyperparam_str += "_nospatial"
    output_dir = os.path.join(args.output_dir, hyperparam_str)

    print("=" * 70)
    print("Mouse Brain COSMOS - GaussianLDA Training")
    print("=" * 70)
    print(f"Feature prior: {args.feature_prior_type}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Spatial: {'disabled' if args.no_spatial else 'enabled'}")
    if not args.no_spatial:
        print(f"GCN layers: {args.gcn_n_layers}, type: {args.gcn_layers_type}")
        print(f"GCN alpha: {args.gcn_alpha} ({'fixed' if args.fixed_alpha else 'learned'})")
    print(f"Loss: {'TraceMeanField_ELBO' if args.meanfield else 'Trace_ELBO'}")
    print(f"N topics: {args.n_topics}, Max epochs: {args.max_epochs}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    print("\nLoading data...")
    mdata = load_data(args.data_path, use_spatial=not args.no_spatial)

    print("\nCreating model...")
    model = create_model(mdata, args)
    print(f"Spatial mode: {model.spatial}")

    print("\nTraining...")
    train_model(model, args)

    print_diagnostics(model)

    print("\nSaving results...")
    save_results(model, mdata, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
