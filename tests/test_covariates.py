"""
Tests for batch effect / covariate support in MultimodalAmortizedLDA.

This module tests the STAMP-style batch correction implementation including:
- Backward compatibility (no covariates)
- Single categorical covariate (batch key) -- STAMP-style
- encode_covariates flag (encoder+decoder vs decoder-only)
- Integration with spatial (GCN) and non-spatial models
- Error cases (continuous covariates, multiple categoricals)
"""

from __future__ import annotations

import anndata as ad
import mudata as mu
import numpy as np
import pytest
import torch
import scipy.sparse as sp

from topomics.models import MultimodalAmortizedLDA


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def adata_with_categorical():
    """
    Synthetic AnnData with a single categorical covariate (batch).

    Returns
    -------
    adata : AnnData
        Count data with categorical batch column
    N : int
        Number of cells
    F : int
        Number of features
    """
    N, F = 100, 50
    rng = np.random.default_rng(seed=42)

    # Generate count data
    X = rng.poisson(5, size=(N, F)).astype(np.float32)

    adata = ad.AnnData(X=X)

    # Add categorical covariate (single batch key only for STAMP-style)
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B", "batch_C"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    return adata, N, F


@pytest.fixture(scope="function")
def mudata_with_covariates():
    """
    Synthetic MuData with a single categorical covariate for multimodal testing.

    Returns
    -------
    mdata : MuData
        MuData with rna and protein modalities
    N : int
        Number of cells
    G : int
        Number of genes
    P : int
        Number of proteins
    """
    N, G, P = 100, 50, 20
    rng = np.random.default_rng(seed=45)

    # RNA modality
    rna_counts = rng.poisson(5, size=(N, G))
    adata_rna = ad.AnnData(
        X=rna_counts.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"gene_names": [f"gene_{i}" for i in range(G)]},
    )

    # Protein modality
    protein_counts = rng.poisson(3, size=(N, P))
    adata_protein = ad.AnnData(
        X=protein_counts.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"protein_names": [f"protein_{i}" for i in range(P)]},
    )

    # Create MuData
    mdata = mu.MuData({"rna": adata_rna, "protein": adata_protein})

    # Add single categorical covariate
    mdata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    mdata.obs["batch"] = mdata.obs["batch"].astype("category")

    return mdata, N, G, P


@pytest.fixture(scope="function")
def adata_spatial_with_covariates():
    """
    Synthetic spatial AnnData with a single categorical covariate and adjacency graph.

    Returns
    -------
    adata : AnnData
        Spatial data with covariates and precomputed graph
    N : int
        Number of cells
    F : int
        Number of features
    """
    N, F = 100, 50
    rng = np.random.default_rng(seed=46)

    # Generate count data
    X = rng.poisson(5, size=(N, F)).astype(np.float32)

    adata = ad.AnnData(X=X)

    # Add categorical covariate
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Add spatial coordinates
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(N, 2))

    # Create a simple k-NN adjacency matrix (each cell connected to ~5 neighbors)
    from scipy.spatial import KDTree

    tree = KDTree(adata.obsm["spatial"])
    adjacency = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        _, indices = tree.query(adata.obsm["spatial"][i], k=6)  # self + 5 neighbors
        for j in indices:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

    adata.obsp["spatial_connectivities"] = sp.csr_matrix(adjacency)

    return adata, N, F


# ============================================================================
# Test Backward Compatibility (No Covariates)
# ============================================================================


def test_no_covariates_backward_compatible():
    """Test that model works without any covariates (backward compatibility)."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=100)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Setup without covariates
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Model should initialize without error
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    assert model is not None
    assert model.n_cats_per_cov is None
    assert model.n_continuous_cov == 0


def test_no_covariates_trains():
    """Test that model trains successfully without covariates."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=101)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    # Train for a few epochs
    model.train(max_epochs=5, batch_size=32, validation_size=None)

    assert model.is_trained_ is True
    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)


# ============================================================================
# Test Categorical Covariates (STAMP-style batch correction)
# ============================================================================


def test_categorical_covariates_setup(adata_with_categorical):
    """Test that a single categorical covariate is properly registered."""
    adata, N, F = adata_with_categorical
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    # Should have detected 1 categorical covariate
    assert model.n_cats_per_cov is not None
    assert len(model.n_cats_per_cov) == 1
    assert model.n_cats_per_cov[0] == 3  # batch has 3 levels


def test_categorical_covariates_train(adata_with_categorical):
    """Test that model trains with a single categorical covariate (STAMP-style)."""
    adata, N, F = adata_with_categorical
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    # Train
    model.train(max_epochs=5, batch_size=32, validation_size=None)

    assert model.is_trained_ is True
    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)
    assert np.allclose(theta.sum(axis=1), 1.0, rtol=1e-5)


# ============================================================================
# Test STAMP-style Error Cases
# ============================================================================


def test_continuous_covariates_raises_error():
    """Test that continuous covariates raise ValueError (STAMP only supports categorical)."""
    N, F = 100, 50
    rng = np.random.default_rng(seed=43)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["percent_mito"] = rng.uniform(0, 0.2, size=N)

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        continuous_covariate_keys=["percent_mito"],
    )

    with pytest.raises(ValueError, match="Continuous covariates are not supported"):
        MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
        )


def test_multiple_categorical_covariates_raises_error():
    """Test that multiple categorical covariates raise ValueError (STAMP needs exactly one)."""
    N, F = 100, 50
    rng = np.random.default_rng(seed=42)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["sample"] = rng.choice(["s1", "s2"], size=N)
    adata.obs["sample"] = adata.obs["sample"].astype("category")

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch", "sample"],
    )

    with pytest.raises(ValueError, match="Exactly one categorical covariate"):
        MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
        )


# ============================================================================
# Test encode_covariates Flag
# ============================================================================


def test_encode_covariates_true(adata_with_categorical):
    """Test encode_covariates=True (encoder + decoder batch correction)."""
    adata, N, F = adata_with_categorical
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
        encode_covariates=True,  # Default
    )

    assert model.encode_covariates is True

    # Train
    model.train(max_epochs=5, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


def test_encode_covariates_false(adata_with_categorical):
    """Test encode_covariates=False (decoder-only batch correction, STAMP-style)."""
    adata, N, F = adata_with_categorical
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
        encode_covariates=False,  # Decoder-only
    )

    assert model.encode_covariates is False

    # Train
    model.train(max_epochs=5, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Multimodal with Covariates (MuData)
# ============================================================================


def test_multimodal_with_covariates(mudata_with_covariates):
    """Test multimodal model (RNA + protein) with a single batch covariate."""
    mdata, N, G, P = mudata_with_covariates

    # Use from_mudata which should pass covariates through
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=5,
        likelihoods=["gamma_poisson", "multinomial"],
        categorical_covariate_keys=["batch"],
    )

    assert model.n_modalities == 2
    # Check covariates were registered
    assert model.n_cats_per_cov is not None

    # Train
    model.train(max_epochs=5, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Spatial (GCN) with Covariates
# ============================================================================


def test_spatial_with_covariates(adata_spatial_with_covariates):
    """Test spatial model (with GCN encoder) with covariates."""
    adata, N, F = adata_spatial_with_covariates
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        spatial_key="spatial_connectivities",
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
        gcn_n_layers=1,
    )

    # Should have spatial enabled
    assert model.spatial is True
    # Should have covariates
    assert model.n_cats_per_cov is not None

    # Train (just verify it doesn't crash)
    model.train(max_epochs=3, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Different Likelihoods with Covariates
# ============================================================================


def test_multinomial_with_covariates(adata_with_categorical):
    """Test multinomial likelihood with covariates."""
    adata, N, F = adata_with_categorical
    adata = adata.copy()

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["multinomial"],
    )

    model.train(max_epochs=5, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


def test_bernoulli_with_covariates():
    """Test Bernoulli likelihood with covariates."""
    N, F = 100, 50
    rng = np.random.default_rng(seed=200)

    # Binary data
    X = rng.binomial(1, 0.3, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Add covariates
    adata.obs["batch"] = rng.choice(["A", "B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    model.train(max_epochs=5, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test setup_data with Covariates
# ============================================================================


def test_setup_data_with_covariates():
    """Test the unified setup_data API with a single categorical covariate."""
    N, F = 100, 50
    rng = np.random.default_rng(seed=44)
    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Use new unified API
    MultimodalAmortizedLDA.setup_data(
        adata,
        modalities=["rna"],
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    assert model.n_cats_per_cov is not None

    model.train(max_epochs=3, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test from_data with Covariates
# ============================================================================


def test_from_data_with_covariates():
    """Test from_data convenience method with a single categorical covariate."""
    N, F = 100, 50
    rng = np.random.default_rng(seed=44)
    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    model = MultimodalAmortizedLDA.from_data(
        adata,
        modalities=["rna"],
        n_topics=5,
        categorical_covariate_keys=["batch"],
    )

    assert model is not None
    assert model.n_cats_per_cov is not None

    model.train(max_epochs=3, batch_size=32, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_single_category_covariate():
    """Test with a categorical covariate that has only one level."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=300)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Single category
    adata.obs["batch"] = "batch_A"
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    # Should still work
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    assert model.n_cats_per_cov is not None
    assert model.n_cats_per_cov[0] == 1


def test_many_categories():
    """Test with a categorical covariate that has many levels."""
    N, F = 100, 30
    rng = np.random.default_rng(seed=301)

    X = rng.poisson(5, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Many categories
    adata.obs["batch"] = [f"batch_{i % 20}" for i in range(N)]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )

    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
    )

    assert model.n_cats_per_cov is not None
    assert model.n_cats_per_cov[0] == 20

    # Should still train
    model.train(max_epochs=3, batch_size=32, validation_size=None)
    assert model.is_trained_ is True
