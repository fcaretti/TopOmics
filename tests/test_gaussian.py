"""
Tests for Gaussian likelihood support in MultimodalAmortizedLDA.

This module tests the implementation of Gaussian likelihood for continuous data
(e.g., Harmony/scVI embeddings, CLR-normalized protein) including:
- Forward pass execution
- No softmax on topic-feature distribution
- No library size scaling
- Multimodal integration with count likelihoods
- Training convergence
- Batch correction with Gaussian
- Horseshoe prior compatibility
- Regression: existing likelihoods unchanged
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
def continuous_adata():
    """
    Synthetic continuous data simulating Harmony embeddings.

    Returns
    -------
    adata : AnnData
        Continuous float data with shape (N=100, F=30), can be negative
    N : int
        Number of cells
    F : int
        Number of features (embedding dimensions)
    """
    N, F = 100, 30
    rng = np.random.default_rng(seed=42)

    # Simulate Harmony embeddings: real-valued, potentially negative
    X = rng.normal(0, 1, size=(N, F)).astype(np.float32)

    adata = ad.AnnData(X=X)
    return adata, N, F


@pytest.fixture(scope="session")
def count_adata():
    """
    Synthetic count data for regression testing.

    Returns
    -------
    adata : AnnData
        Count data with shape (N=100, G=50)
    N : int
        Number of cells
    G : int
        Number of genes
    """
    N, G = 100, 50
    rng = np.random.default_rng(seed=43)

    X = rng.negative_binomial(10, 0.3, size=(N, G)).astype(np.float32)
    adata = ad.AnnData(X=X)
    return adata, N, G


@pytest.fixture(scope="session")
def multimodal_mudata_rna_harmony():
    """
    Synthetic MuData with RNA (count data) + Harmony embeddings (continuous).

    Returns
    -------
    mdata : MuData
        MuData with 'rna' (gamma_poisson) and 'harmony' (gaussian) modalities
    N : int
        Number of cells
    G : int
        Number of genes
    D : int
        Number of embedding dimensions
    """
    N, G, D = 100, 50, 20
    rng = np.random.default_rng(seed=456)

    # RNA modality - count data
    rna_counts = rng.negative_binomial(10, 0.3, size=(N, G))
    adata_rna = ad.AnnData(
        X=rna_counts.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"gene_names": [f"gene_{i}" for i in range(G)]},
    )

    # Harmony modality - continuous embeddings
    harmony_emb = rng.normal(0, 1, size=(N, D))
    adata_harmony = ad.AnnData(
        X=harmony_emb.astype(np.float32),
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"dim_names": [f"harmony_{i}" for i in range(D)]},
    )

    mdata = mu.MuData({"rna": adata_rna, "harmony": adata_harmony})
    return mdata, N, G, D


@pytest.fixture(scope="function")
def continuous_adata_with_batch():
    """
    Synthetic continuous data with a batch covariate.

    Returns
    -------
    adata : AnnData
        Continuous data with batch column
    N : int
        Number of cells
    F : int
        Number of features
    """
    N, F = 100, 30
    rng = np.random.default_rng(seed=44)

    X = rng.normal(0, 1, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["batch"] = rng.choice(["batch_A", "batch_B"], size=N)
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    return adata, N, F


# ============================================================================
# Test Initialization
# ============================================================================


def test_gaussian_init(continuous_adata):
    """Test that Gaussian likelihood model initializes correctly."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    assert model.n_modalities == 1
    assert model.likelihoods == ["gaussian"]


def test_gaussian_guide_has_sigma_params(continuous_adata):
    """Test that guide creates sigma variational parameters for Gaussian modality."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    guide = model.module.guide
    assert guide.gaussian_sigma_loc is not None
    assert guide.gaussian_sigma_loc[0] is not None
    assert guide.gaussian_sigma_loc[0].shape == (F,)
    assert guide.gaussian_sigma_scale[0].shape == (F,)


# ============================================================================
# Test Forward Pass
# ============================================================================


def test_gaussian_forward_pass(continuous_adata):
    """Test that forward pass completes without errors for Gaussian likelihood."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    from scvi._constants import REGISTRY_KEYS
    from scvi.dataloaders import DataSplitter

    data_splitter = DataSplitter(
        adata_manager=model.adata_manager,
        train_size=1.0,
        validation_size=None,
        batch_size=32,
    )
    data_splitter.setup()
    batch = next(iter(data_splitter.train_dataloader()))
    x = batch[REGISTRY_KEYS.X_KEY]

    with torch.no_grad():
        libraries = torch.stack([x.sum(dim=1)], dim=1)
        model.module.model(x=x, libraries=libraries, n_obs=N)


def test_gaussian_accepts_negative_data():
    """Test that Gaussian likelihood handles negative values (unlike count likelihoods)."""
    N, F = 50, 20
    rng = np.random.default_rng(seed=100)

    # Data with negative values (like CLR-normalized protein or Harmony)
    X = rng.normal(-2, 3, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    # Train briefly - should not crash on negative data
    model.train(max_epochs=3, batch_size=50, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Training
# ============================================================================


def test_gaussian_training_converges(continuous_adata):
    """Test that model trains successfully with Gaussian likelihood."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    model.train(max_epochs=10, batch_size=50, validation_size=None)

    assert model.is_trained_ is True

    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)
    assert np.all(theta >= 0) and np.all(theta <= 1)
    assert np.allclose(theta.sum(axis=1), 1.0, rtol=1e-5)


def test_gaussian_topic_by_feature_no_softmax(continuous_adata):
    """Test that topic_by_feature returns raw means (not softmaxed) for Gaussian."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    model.train(max_epochs=5, batch_size=50, validation_size=None)

    tbf = model.get_feature_topic_dist()
    # get_feature_topic_dist returns DataFrame with features as rows, topics as columns
    assert tbf.shape == (F, 5)
    # For Gaussian, topic-feature values should NOT sum to 1 per topic (not on simplex)
    # and can be negative. Column sums (per-topic) should NOT be ~1.
    col_sums = tbf.values.sum(axis=0)
    assert not np.allclose(col_sums, 1.0, atol=0.1), "Gaussian topic-feature dist should NOT be on the simplex"


# ============================================================================
# Test Batch Correction with Gaussian
# ============================================================================


def test_gaussian_with_batch_correction(continuous_adata_with_batch):
    """Test Gaussian likelihood with STAMP-style batch correction."""
    adata, N, F = continuous_adata_with_batch

    MultimodalAmortizedLDA.setup_anndata(
        adata,
        layer=None,
        categorical_covariate_keys=["batch"],
    )
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
    )

    model.train(max_epochs=10, batch_size=50, validation_size=None)

    assert model.is_trained_ is True
    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)


# ============================================================================
# Test Prior Types
# ============================================================================


def test_gaussian_with_horseshoe_prior(continuous_adata):
    """Test Gaussian likelihood with horseshoe prior."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
        topic_feature_prior_type="horseshoe",
    )

    assert model.topic_feature_prior_type == "horseshoe"
    model.train(max_epochs=5, batch_size=50, validation_size=None)
    assert model.is_trained_ is True


def test_gaussian_with_logistic_normal_prior(continuous_adata):
    """Test Gaussian likelihood with logistic_normal prior (default)."""
    adata, N, F = continuous_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gaussian"],
        topic_feature_prior_type="logistic_normal",
    )

    assert model.topic_feature_prior_type == "logistic_normal"
    model.train(max_epochs=5, batch_size=50, validation_size=None)
    assert model.is_trained_ is True


# ============================================================================
# Test Multimodal Integration
# ============================================================================


def test_multimodal_gaussian_gamma_poisson(multimodal_mudata_rna_harmony):
    """Test multimodal model mixing RNA (gamma_poisson) + Harmony (gaussian)."""
    mdata, N, G, D = multimodal_mudata_rna_harmony

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "harmony"],
        n_topics=10,
        likelihoods=["gamma_poisson", "gaussian"],
    )

    assert model.n_modalities == 2
    assert model.likelihoods == ["gamma_poisson", "gaussian"]
    assert model.n_inputs_modalities == [G, D]


def test_multimodal_gaussian_training(multimodal_mudata_rna_harmony):
    """Test that mixed RNA + Gaussian model trains successfully."""
    mdata, N, G, D = multimodal_mudata_rna_harmony

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "harmony"],
        n_topics=5,
        likelihoods=["gamma_poisson", "gaussian"],
    )

    model.train(max_epochs=10, batch_size=50, validation_size=None)

    assert model.is_trained_ is True
    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)


def test_multimodal_gaussian_topic_by_feature(multimodal_mudata_rna_harmony):
    """Test that topic_by_feature works for mixed modalities (softmax for RNA, raw for Gaussian)."""
    mdata, N, G, D = multimodal_mudata_rna_harmony

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "harmony"],
        n_topics=5,
        likelihoods=["gamma_poisson", "gaussian"],
    )

    model.train(max_epochs=5, batch_size=50, validation_size=None)

    tbf = model.get_feature_topic_dist(as_dict=True)
    # RNA (modality 0): should be on simplex — columns (topics) sum to ~1
    rna_tbf = tbf[0]
    rna_col_sums = rna_tbf.sum(axis=0)
    assert np.allclose(rna_col_sums, 1.0, atol=0.05), f"RNA topic-feature columns should sum to ~1, got {rna_col_sums}"

    # Harmony (modality 1): should NOT be on simplex
    harmony_tbf = tbf[1]
    harmony_col_sums = harmony_tbf.sum(axis=0)
    assert not np.allclose(harmony_col_sums, 1.0, atol=0.1), "Gaussian topic-feature columns should NOT sum to 1"


# ============================================================================
# Regression: existing likelihoods unchanged
# ============================================================================


def test_regression_gamma_poisson_unchanged(count_adata):
    """Regression test: gamma_poisson still works exactly as before."""
    adata, N, G = count_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
    )

    model.train(max_epochs=10, batch_size=50, validation_size=None)
    assert model.is_trained_ is True

    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)
    assert np.allclose(theta.sum(axis=1), 1.0, rtol=1e-5)

    # Topic-feature should be on the simplex for count data
    # DataFrame has features as rows, topics as columns — columns should sum to ~1
    tbf = model.get_feature_topic_dist()
    col_sums = tbf.values.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=0.05)


def test_regression_bernoulli_unchanged():
    """Regression test: bernoulli still works exactly as before."""
    N, F = 80, 40
    rng = np.random.default_rng(seed=99)
    X = rng.binomial(1, 0.3, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["bernoulli"],
    )

    model.train(max_epochs=10, batch_size=50, validation_size=None)
    assert model.is_trained_ is True

    theta = model.get_latent_representation()
    assert theta.shape == (N, 5)
    assert np.allclose(theta.sum(axis=1), 1.0, rtol=1e-5)


def test_regression_default_likelihood_is_not_gaussian(count_adata):
    """Ensure the default likelihood via from_data remains gamma_poisson, not gaussian."""
    adata, N, G = count_adata

    # from_data auto-infers likelihoods based on modality names (rna -> gamma_poisson)
    model = MultimodalAmortizedLDA.from_data(
        adata,
        modalities=["rna"],
        n_topics=5,
    )

    assert model.likelihoods == ["gamma_poisson"]


def test_regression_no_gaussian_sigma_params_for_count_model(count_adata):
    """Ensure guide doesn't allocate gaussian_sigma params for non-Gaussian models."""
    adata, N, G = count_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
    )

    guide = model.module.guide
    assert guide.gaussian_sigma_loc is None
    assert guide.gaussian_sigma_scale is None
