"""
Train LinearSCVI on retina dataset with or without batch correction.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import scvi
from anndata import AnnData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path: str) -> AnnData:
    """Load preprocessed retina data."""
    logger.info(f"Loading preprocessed data from {data_path}")
    adata = sc.read_h5ad(data_path)
    return adata


def train_linear_scvi(
    adata: AnnData,
    n_latent: int = 30,
    max_epochs: int = 100,
    output_dir: str = "paper/models",
    seed: int = 42,
    batch_correction: bool = True,
) -> scvi.model.LinearSCVI:
    """
    Train LinearSCVI model with or without batch correction.

    LinearSCVI uses a linear decoder, making it more interpretable
    and comparable to topic models.

    Parameters
    ----------
    adata : AnnData
        Input dataset
    n_latent : int
        Latent dimension (comparable to n_topics)
    max_epochs : int
        Maximum training epochs
    output_dir : str
        Directory to save model
    seed : int
        Random seed
    batch_correction : bool
        If True, use batch as a covariate for correction.
        If False, train without any batch correction.

    Returns
    -------
    model : scvi.model.LinearSCVI
        Trained model
    """
    scvi.settings.seed = seed
    np.random.seed(seed)

    # Use HVGs and raw counts
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    adata_hvg.X = adata_hvg.layers["counts"]

    if batch_correction:
        logger.info("Training LinearSCVI (with batch correction)...")
    else:
        logger.info("Training LinearSCVI (no batch correction)...")
    logger.info(f"Data shape: {adata_hvg.shape}")

    # Setup AnnData for LinearSCVI
    # Note: LinearSCVI uses batch_key instead of categorical_covariate_keys
    setup_kwargs = dict(layer=None)  # Use .X which has counts
    if batch_correction:
        setup_kwargs["batch_key"] = "batch"
    scvi.model.LinearSCVI.setup_anndata(adata_hvg, **setup_kwargs)

    # Initialize model
    model = scvi.model.LinearSCVI(
        adata_hvg,
        n_latent=n_latent,
    )

    # Train
    logger.info("Starting training...")
    model.train(
        max_epochs=max_epochs,
        batch_size=128,
        early_stopping=True,
        early_stopping_monitor="elbo_validation",  # LinearSCVI uses elbo_validation
        early_stopping_patience=10,
    )

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = "retina_linear_scvi" if batch_correction else "retina_linear_scvi_no_batch"
    save_path = output_path / model_name
    model.save(save_path, overwrite=True)
    logger.info(f"Model saved to {save_path}")

    # Get latent representation
    logger.info("Computing latent representation...")
    latent = model.get_latent_representation()

    # Add to adata (latent is already (n_cells, n_latent), directly assign it)
    key = "X_linear_scvi" if batch_correction else "X_linear_scvi_no_batch"
    adata.obsm[key] = latent

    logger.info(f"Latent representation shape: {latent.shape}")

    # Get loadings (gene weights) - similar to topic-gene distributions
    loadings = model.get_loadings()
    logger.info(f"Loadings shape: {loadings.shape}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train LinearSCVI on retina dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="paper/models/retina_preprocessed.h5ad",
        help="Path to preprocessed retina dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/topomics_models/retina", help="Directory to save trained model"
    )
    parser.add_argument("--n_latent", type=int, default=30, help="Latent dimension")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_batch", action="store_true", help="Train without batch correction")

    args = parser.parse_args()

    # Load preprocessed data
    adata = load_preprocessed_data(args.data_path)

    batch_correction = not args.no_batch

    # Train model
    logger.info("\n" + "=" * 80)
    if batch_correction:
        logger.info("Training LinearSCVI (with batch correction)")
    else:
        logger.info("Training LinearSCVI (no batch correction)")
    logger.info("=" * 80 + "\n")

    model = train_linear_scvi(
        adata,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_correction=batch_correction,
    )

    # Save final adata with representation to /data (to save space)
    from pathlib import Path as P

    data_dir = P(args.data_path).parent
    suffix = "retina_with_linear_scvi.h5ad" if batch_correction else "retina_with_linear_scvi_no_batch.h5ad"
    adata.write(data_dir / suffix)
    logger.info(f"Final adata saved to {data_dir / suffix}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
