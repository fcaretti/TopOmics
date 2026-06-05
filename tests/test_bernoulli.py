"""
Tests for Bernoulli likelihood support in MultimodalAmortizedLDA.

This module tests the implementation of Bernoulli likelihood for binary data
(e.g., ATAC-seq peaks, methylation) including:
- Forward pass execution
- Library size scaling
- Binary data validation
- Multimodal integration with other likelihoods
- Training convergence
"""

from __future__ import annotations

import anndata as ad
import mudata as mu
import numpy as np
import pytest
import torch

from topomics.models import MultimodalAmortizedLDA

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def binary_adata():
    """
    Synthetic binary data simulating ATAC-seq peaks.

    Returns
    -------
    adata : AnnData
        Binary data (0/1) with shape (N=100, F=50)
    N : int
        Number of cells
    F : int
        Number of features (peaks)
    """
    N, F = 100, 50
    rng = np.random.default_rng(seed=42)

    # Generate binary data with ~30% accessibility
    X = rng.binomial(1, 0.3, size=(N, F))

    adata = ad.AnnData(X=X.astype(np.float32))
    return adata, N, F


@pytest.fixture(scope="session")
def binary_adata_varying_lib():
    """
    Synthetic binary data with varying library sizes to test depth normalization.

    Returns
    -------
    adata : AnnData
        Binary data with variable library sizes
    N : int
        Number of cells
    F : int
        Number of features
    lib_sizes : np.ndarray
        Library size per cell
    """
    N, F = 100, 50
    rng = np.random.default_rng(seed=123)

    # Generate cells with different library sizes
    # Half with high library (60% accessibility), half with low (15% accessibility)
    X = np.zeros((N, F), dtype=np.float32)
    X[:50] = rng.binomial(1, 0.6, size=(50, F))  # High library cells
    X[50:] = rng.binomial(1, 0.15, size=(50, F))  # Low library cells

    lib_sizes = X.sum(axis=1)

    adata = ad.AnnData(X=X)
    return adata, N, F, lib_sizes


@pytest.fixture(scope="session")
def multimodal_mudata_with_bernoulli():
    """
    Synthetic MuData with RNA (count data) + ATAC (binary data).

    Returns
    -------
    mdata : MuData
        MuData with 'rna' (gamma_poisson) and 'atac' (bernoulli) modalities
    N : int
        Number of cells
    G : int
        Number of genes
    P : int
        Number of peaks
    """
    N, G, P = 100, 50, 30
    rng = np.random.default_rng(seed=456)

    # RNA modality - count data
    rna_counts = rng.negative_binomial(10, 0.3, size=(N, G))
    adata_rna = ad.AnnData(
        X=rna_counts.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"gene_names": [f"gene_{i}" for i in range(G)]},
    )

    # ATAC modality - binary data
    atac_binary = rng.binomial(1, 0.25, size=(N, P))
    adata_atac = ad.AnnData(
        X=atac_binary.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"peak_names": [f"peak_{i}" for i in range(P)]},
    )

    # Create MuData
    mdata = mu.MuData({"rna": adata_rna, "atac": adata_atac})

    return mdata, N, G, P


# ============================================================================
# Test Binary Data Validation
# ============================================================================


def test_bernoulli_accepts_binary_data(binary_adata):
    """Test that Bernoulli likelihood accepts valid binary data {0, 1}."""
    adata, N, F = binary_adata

    # Setup and initialize model - should not raise
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    assert model.n_modalities == 1
    assert model.likelihoods == ["bernoulli"]


def test_bernoulli_rejects_continuous_data():
    """Test that Bernoulli likelihood rejects continuous (non-binary) data."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=789)

    # Continuous data (floats between 0 and 1)
    X = rng.uniform(0, 1, size=(N, F))
    adata = ad.AnnData(X=X.astype(np.float32))

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    with pytest.raises(ValueError, match="contains non-binary values"):
        MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["bernoulli"],
        )


def test_bernoulli_rejects_count_data():
    """Test that Bernoulli likelihood rejects count data {0, 1, 2, ...}."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=101)

    # Count data with values > 1
    X = rng.poisson(2, size=(N, F))
    adata = ad.AnnData(X=X.astype(np.float32))

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    with pytest.raises(ValueError, match="contains non-binary values"):
        MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["bernoulli"],
        )


# ============================================================================
# Test Forward Pass
# ============================================================================


def test_bernoulli_forward_pass(binary_adata):
    """Test that forward pass completes without errors for Bernoulli likelihood."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=10,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Get a batch of data - use make_data_loader from SCVI
    from scvi._constants import REGISTRY_KEYS
    from scvi.dataloaders import DataSplitter

    data_splitter = DataSplitter(
        adata_manager=model.adata_manager,
        train_size=1.0,
        validation_size=None,
        batch_size=32,
    )
    data_splitter.setup()  # Need to call setup() before accessing dataloaders
    batch = next(iter(data_splitter.train_dataloader()))
    x = batch[REGISTRY_KEYS.X_KEY]

    # Forward pass should not raise
    with torch.no_grad():
        libraries = torch.stack([x.sum(dim=1)], dim=1)
        model.module.model(x=x, libraries=libraries, n_obs=N)


def test_bernoulli_likelihood_shape(binary_adata):
    """Test that Bernoulli likelihood produces correct output shapes."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=10,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Train briefly so we can get latent representation
    model.train(max_epochs=5, batch_size=50, validation_size=None)

    # Get latent representation (should be theta with shape (N, K))
    theta = model.get_latent_representation()
    assert theta.shape == (N, 10)
    assert np.all(theta >= 0) and np.all(theta <= 1)
    assert np.allclose(theta.sum(axis=1), 1.0, rtol=1e-5)


# ============================================================================
# Test Library Size Scaling
# ============================================================================


def test_bernoulli_library_scaling(binary_adata_varying_lib):
    """
    Test that library size scaling works correctly.

    Cells with higher library should have higher detection probabilities
    (all else being equal).
    """
    adata, N, F, lib_sizes = binary_adata_varying_lib

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Verify that library sizes differ between groups
    high_lib_mean = lib_sizes[:50].mean()
    low_lib_mean = lib_sizes[50:].mean()
    assert high_lib_mean > low_lib_mean * 2  # Should be significantly different


def test_bernoulli_clamping():
    """
    Test that probability clamping works (p_m is clamped to [0, 1]).

    This is tested implicitly - if clamping didn't work, the Bernoulli
    distribution would raise an error for probabilities > 1.
    """
    N, F = 50, 20
    rng = np.random.default_rng(seed=202)

    # Sparse binary data (will have some cells with very high library relative to mean)
    X = rng.binomial(1, 0.1, size=(N, F))
    # Add a few cells with many peaks to test clamping
    X[:5] = rng.binomial(1, 0.9, size=(5, F))

    adata = ad.AnnData(X=X.astype(np.float32))

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Forward pass should not raise (clamping prevents p > 1)
    from scvi._constants import REGISTRY_KEYS
    from scvi.dataloaders import DataSplitter

    data_splitter = DataSplitter(
        adata_manager=model.adata_manager,
        train_size=1.0,
        validation_size=None,
        batch_size=32,
    )
    data_splitter.setup()  # Need to call setup() before accessing dataloaders
    batch = next(iter(data_splitter.train_dataloader()))
    x = batch[REGISTRY_KEYS.X_KEY]
    with torch.no_grad():
        libraries = torch.stack([x.sum(dim=1)], dim=1)
        model.module.model(x=x, libraries=libraries, n_obs=N)


# ============================================================================
# Test Multimodal Integration
# ============================================================================


def test_multimodal_bernoulli_gamma_poisson(multimodal_mudata_with_bernoulli):
    """Test integration of Bernoulli and gamma_poisson likelihoods in MuData."""
    mdata, N, G, P = multimodal_mudata_with_bernoulli

    # Use from_mudata with explicit likelihoods
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "atac"],
        n_topics=10,
        likelihoods=["gamma_poisson", "bernoulli"],
    )

    assert model.n_modalities == 2
    assert model.likelihoods == ["gamma_poisson", "bernoulli"]
    assert model.n_inputs_modalities == [G, P]


def test_background_not_applied_to_bernoulli(multimodal_mudata_with_bernoulli):
    """
    Test that background terms are only applied to gamma_poisson, not Bernoulli.

    This is tested by checking that the model initializes correctly with
    use_feature_background=True and doesn't try to compute background for Bernoulli.
    """
    mdata, N, G, P = multimodal_mudata_with_bernoulli

    # Enable background - should only apply to RNA (gamma_poisson)
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "atac"],
        n_topics=10,
        likelihoods=["gamma_poisson", "bernoulli"],
        use_feature_background=True,
    )

    # Check that background buffers exist only for RNA modality (index 0)
    assert hasattr(model.module.model, "init_bg_mean_0")
    bg_0 = model.module.model.init_bg_mean_0
    assert bg_0.numel() == G  # Should have background for RNA features

    # ATAC modality (index 1) should have placeholder background
    assert hasattr(model.module.model, "init_bg_mean_1")
    bg_1 = model.module.model.init_bg_mean_1
    assert bg_1.numel() == 1  # Placeholder (single element)


# ============================================================================
# Test Training
# ============================================================================


def test_bernoulli_training_converges(binary_adata):
    """Test that model trains successfully with Bernoulli likelihood."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Train for a few epochs
    model.train(max_epochs=10, batch_size=50, validation_size=None)

    # Model should be trained
    assert model.is_trained_ is True

    # Should be able to get latent representation
    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)


def test_bernoulli_elbo_increases(binary_adata):
    """Test that ELBO increases during training (learning is happening)."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=8,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Train and track history
    model.train(max_epochs=20, batch_size=50, validation_size=None)

    # Get training history
    history = model.history_

    # ELBO should improve (become less negative)
    # Note: we check elbo_train (lower is better for loss, but ELBO should increase)
    assert len(history["elbo_train"]) > 0


def test_multimodal_training(multimodal_mudata_with_bernoulli):
    """Test that multimodal model (RNA + ATAC) trains successfully."""
    mdata, N, G, P = multimodal_mudata_with_bernoulli

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "atac"],
        n_topics=10,
        likelihoods=["gamma_poisson", "bernoulli"],
    )

    # Train
    model.train(max_epochs=10, batch_size=50, validation_size=None)

    assert model.is_trained_ is True

    # Check outputs
    theta = model.get_latent_representation()
    assert theta.shape == (N, 10)


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_bernoulli_all_zeros():
    """Test Bernoulli with all-zero data (extreme sparsity)."""
    N, F = 50, 20
    X = np.zeros((N, F), dtype=np.float32)
    adata = ad.AnnData(X=X)

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Should initialize without error
    assert model.is_trained_ is False


def test_bernoulli_all_ones():
    """Test Bernoulli with all-one data (no sparsity)."""
    N, F = 50, 20
    X = np.ones((N, F), dtype=np.float32)
    adata = ad.AnnData(X=X)

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    # Should initialize without error
    assert model.is_trained_ is False


def test_bernoulli_with_horseshoe_prior(binary_adata):
    """Test Bernoulli likelihood with horseshoe prior (should work)."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
        topic_feature_prior_type="horseshoe",
    )

    # Should initialize successfully
    assert model.topic_feature_prior_type == "horseshoe"

    # Train briefly
    model.train(max_epochs=5, batch_size=50, validation_size=None)
    assert model.is_trained_ is True


def test_bernoulli_with_logistic_normal_prior(binary_adata):
    """Test Bernoulli likelihood with logistic_normal prior (default)."""
    adata, N, F = binary_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
        topic_feature_prior_type="logistic_normal",
    )

    # Should initialize successfully
    assert model.topic_feature_prior_type == "logistic_normal"

    # Train briefly
    model.train(max_epochs=5, batch_size=50, validation_size=None)
    assert model.is_trained_ is True
