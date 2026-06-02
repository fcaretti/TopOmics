"""
Train topomics on retina dataset with batch correction.

This script trains up to three variants:
1. encode_covariates=True (default): Batch correction in encoder + decoder
2. encode_covariates=False: Batch correction in decoder only (scVI-style)
3. No batch correction: No covariates passed to the model
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import scvi
from anndata import AnnData

from topomics.models import MultimodalAmortizedLDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_retina_data(save_path: str = "/data/retina_dataset") -> AnnData:
    """
    Load and prepare the retina dataset.

    The dataset should be downloaded first using:
        python paper/scripts/download_retina.py --output_dir /data/retina_dataset

    Returns
    -------
    adata : AnnData
        Preprocessed retina dataset with batch and labels annotations
    """
    from pathlib import Path

    retina_file = Path(save_path) / "retina.h5ad"

    if not retina_file.exists():
        logger.error(f"Retina dataset not found at {retina_file}")
        logger.error("Please download it first using:")
        logger.error(f"  python paper/scripts/download_retina.py --output_dir {save_path}")
        raise FileNotFoundError(f"Dataset not found: {retina_file}")

    logger.info(f"Loading retina dataset from {retina_file}...")
    adata = sc.read_h5ad(retina_file)

    logger.info(f"Dataset shape: {adata.shape}")
    logger.info(f"Batches: {adata.obs['batch'].unique()}")
    logger.info(f"Cell types: {adata.obs['labels'].nunique()}")

    # Basic preprocessing
    logger.info("Preprocessing...")
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_genes=200)

    # Normalize and log-transform for visualization (keep raw counts for model)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    # Find highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        batch_key="batch",
        flavor="seurat_v3"
    )

    logger.info(f"After filtering: {adata.shape}")
    logger.info(f"Highly variable genes: {adata.var['highly_variable'].sum()}")

    return adata


def train_omics_topic(
    adata: AnnData,
    encode_covariates: bool,
    batch_correction: bool = True,
    n_topics: int = 30,
    n_hidden: int = 128,
    max_epochs: int = 100,
    output_dir: str = "paper/models",
    seed: int = 42
) -> MultimodalAmortizedLDA:
    """
    Train topomics model with or without batch correction.

    Parameters
    ----------
    adata : AnnData
        Input dataset
    encode_covariates : bool
        If True, apply batch correction in encoder + decoder (default)
        If False, apply batch correction in decoder only (scVI-style)
        Ignored when batch_correction=False.
    batch_correction : bool
        If True, use batch as a covariate for correction.
        If False, train without any batch correction.
    n_topics : int
        Number of topics
    n_hidden : int
        Hidden layer size
    max_epochs : int
        Maximum training epochs
    output_dir : str
        Directory to save model
    seed : int
        Random seed

    Returns
    -------
    model : MultimodalAmortizedLDA
        Trained model
    """
    np.random.seed(seed)

    # Use HVGs and raw counts
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    adata_hvg.X = adata_hvg.layers["counts"]

    if batch_correction:
        logger.info(f"Training topomics (encode_covariates={encode_covariates})...")
    else:
        logger.info("Training topomics (no batch correction)...")
    logger.info(f"Data shape: {adata_hvg.shape}")

    # Setup data
    if batch_correction:
        MultimodalAmortizedLDA.setup_anndata(
            adata_hvg,
            layer=None,  # Use .X which has counts
            categorical_covariate_keys=["batch"]
        )
    else:
        MultimodalAmortizedLDA.setup_anndata(
            adata_hvg,
            layer=None,  # Use .X which has counts
        )

    # Initialize model
    model = MultimodalAmortizedLDA(
        adata_hvg,
        n_topics=n_topics,
        n_inputs_modalities=[adata_hvg.n_vars],
        likelihoods=["gamma_poisson"],
        n_hidden=n_hidden,
        encode_covariates=encode_covariates if batch_correction else False,
    )

    # Train
    logger.info("Starting training...")
    model.train(
        max_epochs=max_epochs,
        batch_size=128,
        validation_size=0.1,
        early_stopping=True,
        early_stopping_monitor="elbo_val",  # Fix: use the correct metric name
        early_stopping_patience=10,
    )

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if batch_correction:
        model_name = f"retina_omics_topic_encode_cov_{encode_covariates}"
    else:
        model_name = "retina_omics_topic_no_batch"
    save_path = output_path / model_name
    model.save(save_path, overwrite=True)
    logger.info(f"Model saved to {save_path}")

    # Get latent representation
    logger.info("Computing latent representation...")
    theta = model.get_latent_representation()

    # Add to adata (theta is already (n_cells, n_topics), directly assign it)
    if batch_correction:
        adata.obsm[f"X_omics_topic_encode_{encode_covariates}"] = theta
    else:
        adata.obsm["X_omics_topic_no_batch"] = theta

    logger.info(f"Latent representation shape: {theta.shape}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train topomics on retina dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/retina_dataset",
        help="Path to save/load retina dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/retina",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=30,
        help="Number of topics"
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=128,
        help="Hidden layer size"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--encode_covariates",
        type=str,
        default="both",
        choices=["true", "false", "both", "none"],
        help="Train with encode_covariates=True, False, both, or none (no batch correction)"
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    preprocessed_file = data_path / "retina_preprocessed.h5ad"

    # Load preprocessed data if available, otherwise prepare from scratch
    if preprocessed_file.exists():
        logger.info(f"Loading existing preprocessed data from {preprocessed_file}")
        adata = sc.read_h5ad(preprocessed_file)
    else:
        adata = prepare_retina_data(args.data_path)
        adata.write(preprocessed_file)
        logger.info(f"Preprocessed data saved to {preprocessed_file}")

    # Create output directory for models
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train model(s)
    if args.encode_covariates in ["true", "both"]:
        logger.info("\n" + "="*80)
        logger.info("Training topomics with encode_covariates=True")
        logger.info("="*80 + "\n")
        model_true = train_omics_topic(
            adata,
            encode_covariates=True,
            n_topics=args.n_topics,
            n_hidden=args.n_hidden,
            max_epochs=args.max_epochs,
            output_dir=args.output_dir,
            seed=args.seed
        )

    if args.encode_covariates in ["false", "both"]:
        logger.info("\n" + "="*80)
        logger.info("Training topomics with encode_covariates=False")
        logger.info("="*80 + "\n")
        model_false = train_omics_topic(
            adata,
            encode_covariates=False,
            n_topics=args.n_topics,
            n_hidden=args.n_hidden,
            max_epochs=args.max_epochs,
            output_dir=args.output_dir,
            seed=args.seed
        )

    if args.encode_covariates == "none":
        logger.info("\n" + "="*80)
        logger.info("Training topomics without batch correction")
        logger.info("="*80 + "\n")
        model_no_batch = train_omics_topic(
            adata,
            encode_covariates=False,
            batch_correction=False,
            n_topics=args.n_topics,
            n_hidden=args.n_hidden,
            max_epochs=args.max_epochs,
            output_dir=args.output_dir,
            seed=args.seed
        )

    # Save final adata with all representations to /data
    if args.encode_covariates == "none":
        out_file = "retina_with_omics_topic_no_batch.h5ad"
    else:
        out_file = "retina_with_omics_topic.h5ad"
    adata.write(data_path / out_file)
    logger.info(f"Final adata saved to {data_path / out_file}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
