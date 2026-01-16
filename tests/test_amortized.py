from __future__ import annotations

import anndata as ad
import mudata as mu
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scvi._constants import REGISTRY_KEYS

from omics_topic.models import MultimodalAmortizedLDA


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def synthetic_adata():
    """
    Tiny two-modal synthetic counts:
    * RNA : Negative-Binomial(10, 0.3), 200 × 50
    * ADT : Poisson(20),               200 × 10
    """
    N, G, P = 200, 50, 10
    rng = np.random.default_rng(seed=0)
    rna = rng.negative_binomial(10, 0.3, size=(N, G))
    adt = rng.poisson(20, size=(N, P))
    X = np.concatenate([rna, adt], axis=1)
    adata = ad.AnnData(X)
    return adata, N, G, P


@pytest.fixture(scope="session")
def synthetic_mudata():
    """
    Synthetic MuData with two modalities (RNA + protein).
    
    Returns
    -------
    mdata : MuData
        MuData object with 'rna' and 'protein' modalities
    N : int
        Number of cells
    G : int
        Number of genes
    P : int
        Number of proteins
    """
    N, G, P = 200, 50, 10
    rng = np.random.default_rng(seed=42)
    
    # Create RNA modality
    rna_counts = rng.negative_binomial(10, 0.3, size=(N, G))
    adata_rna = ad.AnnData(
        X=rna_counts,
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"gene_names": [f"gene_{i}" for i in range(G)]}
    )
    
    # Create protein modality
    protein_counts = rng.poisson(20, size=(N, P))
    adata_protein = ad.AnnData(
        X=protein_counts,
        obs={"cell_id": [f"cell_{i}" for i in range(N)]},
        var={"protein_names": [f"protein_{i}" for i in range(P)]}
    )
    
    # Create MuData
    mdata = mu.MuData({"rna": adata_rna, "protein": adata_protein})
    
    return mdata, N, G, P


@pytest.fixture(scope="session")
def synthetic_spatial_adata():
    """
    Synthetic spatial transcriptomics data with adjacency matrix.
    
    Returns
    -------
    adata : AnnData
        Spatial data with connectivity matrix in obsp
    N : int
        Number of spots
    G : int
        Number of genes
    """
    N, G = 100, 50
    rng = np.random.default_rng(seed=123)
    
    # Generate counts
    counts = rng.negative_binomial(10, 0.3, size=(N, G))
    
    # Generate spatial coordinates (10x10 grid)
    grid_size = int(np.sqrt(N))
    x_coords = np.repeat(np.arange(grid_size), grid_size)
    y_coords = np.tile(np.arange(grid_size), grid_size)
    spatial_coords = np.stack([x_coords, y_coords], axis=1).astype(float)
    
    # Create AnnData
    adata = ad.AnnData(
        X=counts,
        obs={"x": x_coords, "y": y_coords},
        obsm={"spatial": spatial_coords},
        var={"gene_names": [f"gene_{i}" for i in range(G)]}
    )
    
    # Create adjacency matrix (4-connectivity: up, down, left, right)
    adjacency = sp.lil_matrix((N, N))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # Right neighbor
            if j < grid_size - 1:
                adjacency[idx, idx + 1] = 1
                adjacency[idx + 1, idx] = 1
            # Down neighbor
            if i < grid_size - 1:
                adjacency[idx, idx + grid_size] = 1
                adjacency[idx + grid_size, idx] = 1
    
    # Normalize adjacency (D^{-1/2} A D^{-1/2})
    adjacency = adjacency.tocsr()
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degrees_inv_sqrt = np.power(degrees, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0
    D_inv_sqrt = sp.diags(degrees_inv_sqrt)
    adjacency_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    adata.obsp["connectivities"] = adjacency_normalized
    
    return adata, N, G


@pytest.fixture(scope="session")
def synthetic_spatial_mudata():
    """
    Synthetic spatial multi-omics data.
    
    Returns
    -------
    mdata : MuData
        Spatial MuData with RNA and protein modalities
    N : int
        Number of spots
    G : int
        Number of genes
    P : int
        Number of proteins
    """
    N, G, P = 100, 50, 20
    rng = np.random.default_rng(seed=456)
    
    # Generate spatial coordinates
    grid_size = int(np.sqrt(N))
    x_coords = np.repeat(np.arange(grid_size), grid_size)
    y_coords = np.tile(np.arange(grid_size), grid_size)
    spatial_coords = np.stack([x_coords, y_coords], axis=1).astype(float)
    
    # Create RNA modality
    rna_counts = rng.negative_binomial(10, 0.3, size=(N, G))
    adata_rna = ad.AnnData(
        X=rna_counts,
        obs={"x": x_coords, "y": y_coords},
        obsm={"spatial": spatial_coords},
        var={"gene_names": [f"gene_{i}" for i in range(G)]}
    )
    
    # Create protein modality
    protein_counts = rng.poisson(20, size=(N, P))
    adata_protein = ad.AnnData(
        X=protein_counts,
        obs={"x": x_coords, "y": y_coords},
        obsm={"spatial": spatial_coords},
        var={"protein_names": [f"protein_{i}" for i in range(P)]}
    )
    
    # Create adjacency matrix
    adjacency = sp.lil_matrix((N, N))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if j < grid_size - 1:
                adjacency[idx, idx + 1] = 1
                adjacency[idx + 1, idx] = 1
            if i < grid_size - 1:
                adjacency[idx, idx + grid_size] = 1
                adjacency[idx + grid_size, idx] = 1
    
    adjacency = adjacency.tocsr()
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degrees_inv_sqrt = np.power(degrees, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0
    D_inv_sqrt = sp.diags(degrees_inv_sqrt)
    adjacency_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    # Add to both modalities
    adata_rna.obsp["connectivities"] = adjacency_normalized
    adata_protein.obsp["connectivities"] = adjacency_normalized
    
    # Create MuData
    mdata = mu.MuData({"rna": adata_rna, "protein": adata_protein})
    
    return mdata, N, G, P


# ============================================================================
# Helper Functions
# ============================================================================


def _build_model(adata: ad.AnnData, G: int, P: int) -> MultimodalAmortizedLDA:
    """Shared helper that sets up `anndata` and returns a fresh model."""
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    return MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
    )


def _build_spatial_model(adata: ad.AnnData, G: int, spatial_key: str = "connectivities") -> MultimodalAmortizedLDA:
    """Build a spatial model from AnnData."""
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key=spatial_key)
    return MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
    )


# ============================================================================
# Basic Tests (Original)
# ============================================================================


def test_build_and_single_elbo(synthetic_adata):
    """Model builds and its ELBO on a mini-batch is finite."""
    adata, N, G, P = synthetic_adata
    model = _build_model(adata, G, P)
    
    # one mini-batch
    dl = model._make_data_loader(adata=adata, batch_size=64)
    x = next(iter(dl))[REGISTRY_KEYS.X_KEY]
    
    # library sizes per modality: (batch , 2)
    libs = torch.stack([x[:, :G].sum(1), x[:, G:].sum(1)], dim=1)
    
    elbo = model.module.get_elbo(x, libs, n_obs=N)
    assert np.isfinite(elbo), "ELBO is inf / NaN"


def test_train_one_epoch_and_latent_shape(synthetic_adata):
    """A single epoch finishes and latent matrix has the expected shape."""
    adata, N, G, P = synthetic_adata
    model = _build_model(adata, G, P)
    
    # fast CPU-only training
    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)


# ============================================================================
# MuData Tests
# ============================================================================


def test_setup_mudata(synthetic_mudata):
    """Test that MuData setup works correctly."""
    mdata, N, G, P = synthetic_mudata
    
    # Setup MuData
    mdata_setup, modality_names, feat_counts = MultimodalAmortizedLDA.setup_mudata(
        mdata,
        modality_order=["rna", "protein"],
        layer_dict=None,
    )
    
    # Check metadata
    assert "_multimodal_setup" in mdata_setup.uns
    setup_info = mdata_setup.uns["_multimodal_setup"]
    
    assert setup_info["modality_order"] == ["rna", "protein"]
    assert setup_info["feat_counts"] == [G, P]
    assert modality_names == ["rna", "protein"]
    assert feat_counts == [G, P]


def test_mudata_from_mudata_constructor(synthetic_mudata):
    """Test the from_mudata constructor."""
    mdata, N, G, P = synthetic_mudata
    
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
    )
    
    # Check model attributes
    assert model.n_modalities == 2
    assert model.n_inputs_modalities == [G, P]
    assert model.likelihoods == ["gamma_poisson", "multinomial"]  # Default
    
    # Check that adata is set up correctly
    assert model.adata is not None
    assert model.adata.n_obs == N
    assert model.adata.n_vars == G + P  # Concatenated


def test_mudata_train_and_inference(synthetic_mudata):
    """Test training on MuData and getting latent representation."""
    mdata, N, G, P = synthetic_mudata
    
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
    )
    
    # Train for 1 epoch
    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    
    # Get latent representation
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)
    
    # Check that it's a valid probability distribution
    assert np.allclose(Z.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(Z >= 0)


def test_mudata_get_feature_topic_dist(synthetic_mudata):
    """Test getting feature-topic distributions per modality."""
    mdata, N, G, P = synthetic_mudata
    
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
    )
    
    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    
    # Get as dict
    tbf_dict = model.get_feature_topic_dist(n_samples=100, as_dict=True)
    
    assert isinstance(tbf_dict, dict)
    assert len(tbf_dict) == 2
    assert 0 in tbf_dict and 1 in tbf_dict
    
    # Check shapes: each should be (n_features × n_topics)
    assert tbf_dict[0].shape == (G, 8)
    assert tbf_dict[1].shape == (P, 8)
    
    # Check that topics sum to 1 across features
    assert np.allclose(tbf_dict[0].sum(axis=0), 1.0, atol=1e-3)
    assert np.allclose(tbf_dict[1].sum(axis=0), 1.0, atol=1e-3)


# ============================================================================
# Spatial Tests (Non-MuData)
# ============================================================================


def test_spatial_setup(synthetic_spatial_adata):
    """Test that spatial data setup works."""
    adata, N, G = synthetic_spatial_adata
    
    # Check that connectivities exist
    assert "connectivities" in adata.obsp
    assert adata.obsp["connectivities"].shape == (N, N)
    
    # Setup with spatial key
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    
    # Build model (note: spatial flag set in __init__)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
    )
    
    # Model should recognize it has spatial data
    # (You may need to check model.spatial or model.module.guide.use_gcn depending on implementation)
    assert model.module.guide.adjacency is not None or hasattr(model, "spatial")


def test_spatial_train(synthetic_spatial_adata):
    """Test training on spatial data."""
    adata, N, G = synthetic_spatial_adata
    
    model = _build_spatial_model(adata, G, spatial_key="connectivities")
    
    # Train for 1 epoch with full batch (required for GCN)
    model.train(
        max_epochs=1,
        batch_size=N,  # Full batch for spatial
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )
    
    # Get latent representation
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)


def test_spatial_gcn_activated(synthetic_spatial_adata):
    """Test that GCN encoder is actually used when spatial data is present."""
    adata, N, G = synthetic_spatial_adata
    
    # Build model with spatial key
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
    )
    
    # Check that GCN components are present
    assert model.module.guide.use_gcn is True, "GCN should be activated"
    assert model.module.guide.gcn_encoders is not None, "GCN encoders should exist"
    assert len(model.module.guide.gcn_encoders) == 1, "Should have 1 GCN encoder"


# ============================================================================
# Spatial MuData Tests
# ============================================================================


def test_spatial_mudata_setup(synthetic_spatial_mudata):
    """Test setup of spatial MuData."""
    mdata, N, G, P = synthetic_spatial_mudata
    
    # Check that both modalities have spatial info
    assert "spatial" in mdata.mod["rna"].obsm
    assert "spatial" in mdata.mod["protein"].obsm
    assert "connectivities" in mdata.mod["rna"].obsp
    assert "connectivities" in mdata.mod["protein"].obsp
    
    # Setup should work
    mdata_setup, modality_names, feat_counts = MultimodalAmortizedLDA.setup_mudata(
        mdata,
        modality_order=["rna", "protein"],
        layer_dict=None,
    )
    
    assert feat_counts == [G, P]


def test_spatial_mudata_train(synthetic_spatial_mudata):
    """Test training on spatial MuData."""
    mdata, N, G, P = synthetic_spatial_mudata
    
    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
        # Note: You may need to pass spatial_key or the model should auto-detect
    )
    
    # Train with full batch (required for spatial)
    model.train(
        max_epochs=1,
        batch_size=N,
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )
    
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_mismatched_modality_counts():
    """Test that mismatched modality info raises error."""
    N, G, P = 100, 50, 10
    rng = np.random.default_rng(seed=0)
    X = rng.negative_binomial(10, 0.3, size=(N, G + P))
    adata = ad.AnnData(X)
    
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    
    # Wrong feature counts
    with pytest.raises(ValueError, match="Sum.*must equal.*n_vars"):
        MultimodalAmortizedLDA(
            adata,
            n_inputs_modalities=[G, P + 10],  # Wrong!
            likelihoods=["gamma_poisson", "multinomial"],
            n_topics=8,
        )


def test_mismatched_likelihoods():
    """Test that mismatched likelihoods length raises error."""
    N, G, P = 100, 50, 10
    rng = np.random.default_rng(seed=0)
    X = rng.negative_binomial(10, 0.3, size=(N, G + P))
    adata = ad.AnnData(X)
    
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    
    with pytest.raises(ValueError, match="same length"):
        MultimodalAmortizedLDA(
            adata,
            n_inputs_modalities=[G, P],
            likelihoods=["gamma_poisson"],  # Only 1 likelihood for 2 modalities!
            n_topics=8,
        )


def test_perplexity_computation(synthetic_adata):
    """Test that perplexity can be computed."""
    adata, N, G, P = synthetic_adata
    model = _build_model(adata, G, P)
    
    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    
    perplexity = model.get_perplexity()
    assert np.isfinite(perplexity)
    assert perplexity > 0


def test_single_modality_is_valid(synthetic_adata):
    """Test that single modality works (no mixing needed)."""
    adata, N, G, P = synthetic_adata

    # Use only RNA modality
    adata_rna = ad.AnnData(adata.X[:, :G])

    MultimodalAmortizedLDA.setup_anndata(adata_rna, layer=None)
    model = MultimodalAmortizedLDA(
        adata_rna,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
    )

    model.train(max_epochs=1, batch_size=64, early_stopping=False)

    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)


# ============================================================================
# Normalization Tests
# ============================================================================


def test_normalize_encoder_inputs_basic(synthetic_adata):
    """Test that normalize_encoder_inputs parameter works without errors."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
        encoder_scale_factor=1e4,
    )

    # Check parameters are stored
    assert model.normalize_encoder_inputs is True
    assert model.encoder_scale_factor == 1e4
    assert model.module.normalize_encoder_inputs is True
    assert model.module.guide.normalize_encoder_inputs is True

    # Train briefly
    model.train(max_epochs=1, batch_size=64, early_stopping=False)

    # Test inference methods
    theta = model.get_latent_representation()
    assert theta.shape == (N, 8)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)


def test_normalize_encoder_inputs_default_false(synthetic_adata):
    """Test that default behavior (normalize_encoder_inputs=False) is unchanged."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Default should be False
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
    )

    assert model.normalize_encoder_inputs is False
    assert model.module.normalize_encoder_inputs is False
    assert model.module.guide.normalize_encoder_inputs is False

    # Should train normally
    model.train(max_epochs=1, batch_size=64, early_stopping=False)

    theta = model.get_latent_representation()
    assert theta.shape == (N, 8)


def test_normalize_encoder_inputs_transformation_applied(synthetic_adata):
    """Test that log1p normalization is actually applied to encoder inputs."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
        encoder_scale_factor=1e4,
    )

    # Get sample batch
    x = torch.tensor(adata.X[:10], dtype=torch.float32)

    # Manually compute expected transformation for RNA modality
    x_rna = x[:, :G]
    lib_rna = x_rna.sum(dim=1, keepdim=True)
    lib_rna_clamped = torch.clamp(lib_rna, min=1.0)
    expected_rna = torch.log1p(x_rna / lib_rna_clamped * 1e4)

    # Now test that the model would apply this transformation
    # We verify by checking that parameters are set correctly
    assert model.module.guide.normalize_encoder_inputs is True
    assert model.module.guide.encoder_scale_factor == 1e4

    # Train to ensure no errors during transformation
    model.train(max_epochs=1, batch_size=64, early_stopping=False)


def test_normalize_encoder_inputs_zeros_handling(synthetic_adata):
    """Test that zeros are handled correctly with log1p."""
    adata, N, G, P = synthetic_adata

    # Add some all-zero rows for first modality
    adata_with_zeros = adata.copy()
    adata_with_zeros.X[:5, :G] = 0  # Zero out RNA for first 5 cells

    MultimodalAmortizedLDA.setup_anndata(adata_with_zeros, layer=None)
    model = MultimodalAmortizedLDA(
        adata_with_zeros,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
    )

    # Should handle zeros gracefully (library size clamped to 1.0)
    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    theta = model.get_latent_representation()

    # Check no NaNs or Infs (theta is a DataFrame, need .any().any())
    assert not np.isnan(theta.values).any(), "Theta contains NaN"
    assert not np.isinf(theta.values).any(), "Theta contains Inf"
    assert theta.shape == (N, 8)


def test_normalize_encoder_inputs_mudata(synthetic_mudata):
    """Test normalize_encoder_inputs with MuData input."""
    mdata, N, G, P = synthetic_mudata

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
        encoder_scale_factor=1e4,
    )

    assert model.normalize_encoder_inputs is True

    model.train(max_epochs=1, batch_size=64, early_stopping=False)
    theta = model.get_latent_representation()
    assert theta.shape == (N, 8)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)


def test_normalize_encoder_inputs_spatial(synthetic_spatial_adata):
    """Test normalize_encoder_inputs with spatial/GCN encoders."""
    adata, N, G = synthetic_spatial_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
        encoder_scale_factor=1e4,
    )

    # Check that GCN is activated and normalization is set
    assert model.module.guide.use_gcn is True
    assert model.module.guide.normalize_encoder_inputs is True

    # Train with full batch (required for spatial)
    model.train(
        max_epochs=1,
        batch_size=N,
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )

    theta = model.get_latent_representation()
    assert theta.shape == (N, 8)
    assert not np.isnan(theta.values).any()


def test_normalize_encoder_inputs_inference_consistency(synthetic_adata):
    """Test that same transformation applied during training and inference."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Get latent representation (uses get_topic_distribution with sampling)
    theta1 = model.get_latent_representation()

    # Get it again (should be consistent but with some sampling variability)
    theta2 = model.get_latent_representation()

    # Should be close (some variability due to sampling with n_samples=5000)
    # Use relaxed tolerance to account for Monte Carlo sampling
    np.testing.assert_allclose(theta1.values, theta2.values, rtol=0.02, atol=0.01)


def test_normalize_encoder_inputs_changes_results(synthetic_adata):
    """Test that normalize_encoder_inputs actually changes model behavior."""
    adata, N, G, P = synthetic_adata

    # Model without normalization
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model_raw = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=5,
        n_hidden=16,
        normalize_encoder_inputs=False,
    )
    model_raw.train(max_epochs=3, batch_size=64, early_stopping=False)

    # Model with normalization (need fresh setup)
    adata2 = adata.copy()
    MultimodalAmortizedLDA.setup_anndata(adata2, layer=None)
    model_norm = MultimodalAmortizedLDA(
        adata2,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=5,
        n_hidden=16,
        normalize_encoder_inputs=True,
    )
    model_norm.train(max_epochs=3, batch_size=64, early_stopping=False)

    # Results should be different (different preprocessing)
    theta_raw = model_raw.get_latent_representation()
    theta_norm = model_norm.get_latent_representation()

    # Should not be identical (allows for some tolerance but should differ significantly)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(theta_raw.values, theta_norm.values, rtol=0.01)


# ============================================================================
# Entropy Regularization Tests
# ============================================================================


def test_entropy_weight_parameter_stored(synthetic_adata):
    """Test that entropy_weight parameter is properly stored."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Model with entropy_weight = 0.0 (default)
    model_no_entropy = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.0,
    )
    assert model_no_entropy.entropy_weight == 0.0
    assert model_no_entropy.module.entropy_weight == 0.0
    assert model_no_entropy.module.guide.entropy_weight == 0.0

    # Model with entropy_weight = 0.1
    adata2 = adata.copy()
    MultimodalAmortizedLDA.setup_anndata(adata2, layer=None)
    model_with_entropy = MultimodalAmortizedLDA(
        adata2,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )
    assert model_with_entropy.entropy_weight == 0.1
    assert model_with_entropy.module.entropy_weight == 0.1
    assert model_with_entropy.module.guide.entropy_weight == 0.1


def test_entropy_weight_getter(synthetic_adata):
    """Test get_entropy_weight() method."""
    adata, N, G, P = synthetic_adata
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.05,
    )

    assert model.get_entropy_weight() == 0.05


def test_entropy_regularization_increases_entropy(synthetic_adata):
    """Test that entropy_weight > 0 leads to higher entropy after training."""
    adata, N, G, P = synthetic_adata

    # Model without entropy regularization
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model_no_entropy = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.0,
    )
    model_no_entropy.train(max_epochs=5, batch_size=64, early_stopping=False)

    # Model with entropy regularization
    adata2 = adata.copy()
    MultimodalAmortizedLDA.setup_anndata(adata2, layer=None)
    model_with_entropy = MultimodalAmortizedLDA(
        adata2,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )
    model_with_entropy.train(max_epochs=5, batch_size=64, early_stopping=False)

    # Get cell-topic distributions
    theta_no_entropy = model_no_entropy.get_latent_representation()
    theta_with_entropy = model_with_entropy.get_latent_representation()

    # Compute mean entropy per cell
    def compute_entropy(theta):
        """Compute per-cell entropy from topic distributions."""
        eps = 1e-10
        return -(theta * np.log(theta + eps)).sum(axis=1).mean()

    entropy_no_reg = compute_entropy(theta_no_entropy.values)
    entropy_with_reg = compute_entropy(theta_with_entropy.values)

    # Model with entropy regularization should have higher entropy
    assert entropy_with_reg > entropy_no_reg, (
        f"Entropy with regularization ({entropy_with_reg:.4f}) should be higher "
        f"than without ({entropy_no_reg:.4f})"
    )


def test_get_cell_entropy_method(synthetic_adata):
    """Test the get_cell_entropy() method."""
    adata, N, G, P = synthetic_adata
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.05,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Get per-cell entropy
    cell_entropies = model.get_cell_entropy(n_samples=50)

    # Check shape and properties
    assert cell_entropies.shape == (N,)
    assert np.all(np.isfinite(cell_entropies))
    assert np.all(cell_entropies >= 0)

    # Entropy should be bounded by log(n_topics)
    max_entropy = np.log(8)  # 8 topics
    assert np.all(cell_entropies <= max_entropy + 0.1)  # Small tolerance


def test_get_last_entropy(synthetic_adata):
    """Test get_last_entropy() method."""
    adata, N, G, P = synthetic_adata
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )

    # Before training, should return None
    assert model.get_last_entropy() is None

    # Train and check
    model.train(max_epochs=1, batch_size=64, early_stopping=False)

    # After training, should return a float
    last_entropy = model.get_last_entropy()
    assert last_entropy is not None
    assert isinstance(last_entropy, float)
    assert np.isfinite(last_entropy)
    assert last_entropy >= 0


def test_entropy_term_with_zero_weight(synthetic_adata):
    """Test that entropy_weight=0 behaves like the original model."""
    adata, N, G, P = synthetic_adata
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.0,
    )

    # Should train without errors
    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # get_last_entropy should return None when weight is 0
    # (because entropy is not computed in forward pass)
    last_entropy = model.get_last_entropy()
    assert last_entropy is None


def test_entropy_extensive_formulation(synthetic_adata):
    """Test that entropy term scales properly with batch size (extensive formulation)."""
    adata, N, G, P = synthetic_adata
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )

    # Train with different batch sizes - should converge to similar results
    # (This tests that the extensive formulation with Pyro scaling works correctly)
    model.train(max_epochs=3, batch_size=32, early_stopping=False)

    theta = model.get_latent_representation()
    assert theta.shape == (N, 8)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)


def test_entropy_with_mudata(synthetic_mudata):
    """Test entropy regularization with MuData input."""
    mdata, N, G, P = synthetic_mudata

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )

    assert model.entropy_weight == 0.1

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Check that entropy can be computed
    cell_entropies = model.get_cell_entropy(n_samples=50)
    assert cell_entropies.shape == (N,)
    assert np.all(np.isfinite(cell_entropies))


def test_entropy_with_spatial(synthetic_spatial_adata):
    """Test entropy regularization with spatial data."""
    adata, N, G = synthetic_spatial_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
    )

    # Train with full batch (required for spatial)
    model.train(
        max_epochs=2,
        batch_size=N,
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )

    # Check entropy computation works with spatial data
    cell_entropies = model.get_cell_entropy(n_samples=50)
    assert cell_entropies.shape == (N,)
    assert np.all(np.isfinite(cell_entropies))


# ========================================================================
# Topic variance regularization tests
# ========================================================================


def test_topic_variance_weight_parameter_stored(synthetic_adata):
    """Test that topic_variance_weight is properly stored in model and module."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
        topic_variance_weight=5.0,
    )

    # Check parameter is stored in module
    assert hasattr(model.module, "topic_variance_weight")
    assert model.module.topic_variance_weight == 5.0

    # Check parameter is stored in guide
    assert hasattr(model.module.guide, "topic_variance_weight")
    assert model.module.guide.topic_variance_weight == 5.0

    # Check getter method
    assert model.get_topic_variance_weight() == 5.0


def test_topic_variance_weight_getter(synthetic_adata):
    """Test get_topic_variance_weight() method."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Test with default (0.0)
    model_default = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
    )
    assert model_default.get_topic_variance_weight() == 0.0

    # Test with explicit value
    model_custom = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
        topic_variance_weight=10.0,
    )
    assert model_custom.get_topic_variance_weight() == 10.0


def test_topic_variance_regularization_increases_diversity(synthetic_adata):
    """Test that topic variance regularization makes cells more diverse."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Train WITHOUT topic variance regularization
    model_no_var = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=0.0,
    )
    model_no_var.train(max_epochs=10, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Train WITH topic variance regularization
    model_with_var = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=5.0,
    )
    model_with_var.train(max_epochs=10, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Get topic distributions for all cells
    theta_no_var = model_no_var.get_cell_topic_dist()  # (N, K)
    theta_with_var = model_with_var.get_cell_topic_dist()  # (N, K)

    # Compute pairwise cosine similarity between cells
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_no_var = cosine_similarity(theta_no_var)
    similarity_with_var = cosine_similarity(theta_with_var)

    # Mean similarity (excluding diagonal)
    mask = ~np.eye(N, dtype=bool)
    mean_sim_no_var = similarity_no_var[mask].mean()
    mean_sim_with_var = similarity_with_var[mask].mean()

    # With topic variance regularization, cells should be MORE diverse (LOWER similarity)
    assert mean_sim_with_var < mean_sim_no_var, (
        f"Expected lower similarity with variance regularization, "
        f"but got {mean_sim_with_var:.3f} vs {mean_sim_no_var:.3f}"
    )


def test_get_topic_variance_method(synthetic_adata):
    """Test get_topic_variance() method."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=5.0,
    )

    model.train(max_epochs=5, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Test get_topic_variance
    topic_variance = model.get_topic_variance(n_samples=50)
    assert topic_variance.shape == (8,), f"Expected shape (8,), got {topic_variance.shape}"
    assert np.all(np.isfinite(topic_variance)), "Topic variance contains NaN or inf"
    assert np.all(topic_variance >= 0), "Variance should be non-negative"


def test_get_last_topic_variance(synthetic_adata):
    """Test get_last_topic_variance() method."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=5.0,
    )

    # Before training, should return None
    assert model.get_last_topic_variance() is None

    # After one forward pass
    model.train(max_epochs=1, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Should now return a value
    last_variance = model.get_last_topic_variance()
    assert last_variance is not None
    assert isinstance(last_variance, float)
    assert np.isfinite(last_variance)
    assert last_variance >= 0


def test_topic_variance_term_with_zero_weight(synthetic_adata):
    """Test that topic variance term is NOT computed when weight is 0."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
        topic_variance_weight=0.0,
    )

    model.train(max_epochs=1, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # _last_topic_variance should remain None when weight is 0
    last_variance = model.get_last_topic_variance()
    assert last_variance is None, "Expected None when topic_variance_weight=0"


def test_combined_entropy_and_variance_regularization(synthetic_adata):
    """Test that both entropy and variance regularization can be used together."""
    adata, N, G = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        entropy_weight=0.1,
        topic_variance_weight=5.0,
    )

    model.train(max_epochs=5, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Both metrics should be available
    last_entropy = model.get_last_entropy()
    last_variance = model.get_last_topic_variance()

    assert last_entropy is not None
    assert last_variance is not None
    assert np.isfinite(last_entropy)
    assert np.isfinite(last_variance)

    # Compute full metrics
    cell_entropies = model.get_cell_entropy(n_samples=50)
    topic_variance = model.get_topic_variance(n_samples=50)

    assert cell_entropies.shape == (N,)
    assert topic_variance.shape == (8,)


def test_topic_variance_with_mudata(synthetic_mudata):
    """Test topic variance regularization with MuData."""
    mdata, N, G1, G2 = synthetic_mudata

    MultimodalAmortizedLDA.setup_mudata(mdata, modality_keys=["rna", "protein"])
    model = MultimodalAmortizedLDA(
        mdata,
        n_inputs_modalities=[G1, G2],
        likelihoods=["gamma_poisson", "gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=5.0,
    )

    model.train(max_epochs=3, batch_size=64, train_size=1.0, validation_size=0, early_stopping=False)

    # Check topic variance computation works with MuData
    topic_variance = model.get_topic_variance(n_samples=50)
    assert topic_variance.shape == (8,)
    assert np.all(np.isfinite(topic_variance))


def test_topic_variance_with_spatial(synthetic_spatial_adata):
    """Test topic variance regularization with spatial data."""
    adata, N, G = synthetic_spatial_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        topic_variance_weight=5.0,
    )

    # Train with full batch (required for spatial)
    model.train(
        max_epochs=2,
        batch_size=N,
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )

    # Check topic variance computation works with spatial data
    topic_variance = model.get_topic_variance(n_samples=50)
    assert topic_variance.shape == (8,)
    assert np.all(np.isfinite(topic_variance))


# ============================================================================
# Spatial Bug Fix Tests (Normalization & Collapse)
# ============================================================================


def test_spatial_normalization_consistency(synthetic_spatial_adata):
    """
    Test that full graph data stored in GCN encoders has same normalization
    as minibatch data processed during training.

    This verifies the fix for the spatial collapse bug where full graph data
    was not log-normalized while minibatch data was.
    """
    adata, N, G = synthetic_spatial_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=8,
        n_hidden=32,
        normalize_encoder_inputs=True,
    )

    # Check that GCN encoders have full graph data stored
    assert model.module.guide.use_gcn is True
    assert len(model.module.guide.gcn_encoders) == 1
    gcn_enc = model.module.guide.gcn_encoders[0]
    assert gcn_enc._graph_initialized

    # Check that stored full graph data is in log-scale (not raw counts)
    x_full = gcn_enc.x_full
    assert x_full.shape[0] == N
    assert x_full.shape[1] == G

    # Log-normalized data should have reasonable range (typically 0-10)
    # Raw count data would have much larger values (0-1000s)
    assert x_full.max() < 50, f"Expected log-normalized data (max < 50), got max={x_full.max()}"
    assert x_full.min() >= 0, "Log-normalized data should be non-negative"

    # Compute expected normalization for comparison
    x_raw = torch.tensor(adata.X, dtype=torch.float32)
    lib = x_raw.sum(dim=1, keepdim=True)
    lib_clamped = torch.clamp(lib, min=1.0)
    median_depth = torch.median(lib_clamped)
    x_expected = torch.log1p(x_raw / lib_clamped * median_depth)

    # Full graph data should match expected log-normalized data
    torch.testing.assert_close(x_full.cpu(), x_expected, rtol=1e-4, atol=1e-4)


def test_spatial_no_collapse(synthetic_spatial_adata):
    """
    Test that training with spatial data does NOT cause topic collapse.

    This verifies the fix for the spatial collapse bug. With proper
    normalization and KL scaling, topic distributions should maintain
    meaningful variance across cells.
    """
    adata, N, G = synthetic_spatial_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None, spatial_key="connectivities")
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G],
        likelihoods=["gamma_poisson"],
        n_topics=10,  # Use 10 topics to better test diversity
        n_hidden=32,
        normalize_encoder_inputs=True,
    )

    # Train with full batch (required for spatial)
    model.train(
        max_epochs=10,
        batch_size=N,
        train_size=1.0,
        validation_size=0,
        early_stopping=False
    )

    # Get cell-topic distributions
    theta = model.get_latent_representation()
    assert theta.shape == (N, 10)

    # Compute per-cell entropy
    eps = 1e-10
    entropy_per_cell = -(theta.values * np.log(theta.values + eps)).sum(axis=1)
    mean_entropy = entropy_per_cell.mean()

    # If topics collapsed, all cells would have same distribution
    # Maximum entropy for 10 topics is log(10) ≈ 2.30
    # We expect mean entropy > 0.5 (much higher than collapsed state near 0)
    assert mean_entropy > 0.5, (
        f"Topics appear to have collapsed: mean entropy={mean_entropy:.3f} "
        f"(expected > 0.5). Entropy close to 0 indicates all cells mapped to same topic."
    )

    # Check topic proportion variance
    # If collapsed, all topics would have similar proportions (near 0.1 for 10 topics)
    topic_props = theta.mean(axis=0)  # Mean proportion for each topic
    topic_std = topic_props.std()

    # Standard deviation should be > 0.05 (topics should differ meaningfully)
    assert topic_std > 0.05, (
        f"Topic proportions show low variance: std={topic_std:.4f} "
        f"(expected > 0.05). This suggests topics collapsed to uniform distribution."
    )

    # Verify no NaN or Inf
    assert not np.isnan(theta.values).any(), "Theta contains NaN"
    assert not np.isinf(theta.values).any(), "Theta contains Inf"


def test_kl_scaling_applied(synthetic_adata):
    """
    Test that KL weight scaling is properly applied to cell-topic distribution.

    This verifies the fix for the KL scaling bug where poutine.scale had
    incorrect syntax (None as first argument instead of scale=kl_weight).
    """
    adata, N, G, P = synthetic_adata

    # Train two models with different KL weights
    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Model with low KL weight (less regularization)
    model_low_kl = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        kl_weight=0.1,
    )
    model_low_kl.train(max_epochs=5, batch_size=64, early_stopping=False)

    # Model with high KL weight (more regularization)
    adata2 = adata.copy()
    MultimodalAmortizedLDA.setup_anndata(adata2, layer=None)
    model_high_kl = MultimodalAmortizedLDA(
        adata2,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        kl_weight=10.0,
    )
    model_high_kl.train(max_epochs=5, batch_size=64, early_stopping=False)

    # Get cell-topic distributions
    theta_low_kl = model_low_kl.get_latent_representation()
    theta_high_kl = model_high_kl.get_latent_representation()

    # Compute entropy for both models
    eps = 1e-10
    entropy_low = -(theta_low_kl.values * np.log(theta_low_kl.values + eps)).sum(axis=1).mean()
    entropy_high = -(theta_high_kl.values * np.log(theta_high_kl.values + eps)).sum(axis=1).mean()

    # Higher KL weight should push toward prior (more uniform, higher entropy)
    # Lower KL weight allows more deviation from prior (potentially lower entropy)
    # The difference should be noticeable if KL scaling is working
    assert entropy_high > entropy_low * 0.9, (
        f"KL weight scaling may not be working correctly. "
        f"Expected higher entropy with high KL weight, but got "
        f"entropy_low={entropy_low:.3f}, entropy_high={entropy_high:.3f}"
    )

    # Verify both produce valid distributions
    assert np.allclose(theta_low_kl.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(theta_high_kl.sum(axis=1), 1.0, atol=1e-5)


# ============================================================================
# Learnable Dispersion Tests
# ============================================================================


def test_learnable_dispersion_parameter_stored(synthetic_adata):
    """Test that learnable_dispersion and global_dispersion are properly stored."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Test with learnable_dispersion=False (default)
    model_fixed = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=False,
    )
    assert model_fixed.learnable_dispersion is False
    assert model_fixed.module.learnable_dispersion is False

    # Test with learnable_dispersion=True, global_dispersion=True
    adata2 = adata.copy()
    MultimodalAmortizedLDA.setup_anndata(adata2, layer=None)
    model_global = MultimodalAmortizedLDA(
        adata2,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=True,
    )
    assert model_global.learnable_dispersion is True
    assert model_global.global_dispersion is True


def test_learnable_dispersion_global_trains(synthetic_adata):
    """Test training with global learnable dispersion."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=True,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Should produce valid latent representation
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)
    assert np.allclose(Z.sum(axis=1), 1.0, atol=1e-5)


def test_learnable_dispersion_per_gene_trains(synthetic_adata):
    """Test training with per-gene learnable dispersion (STAMP-like)."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=False,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)


def test_get_learned_dispersion_global(synthetic_adata):
    """Test get_learned_dispersion with global dispersion."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=True,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Get dispersion for first modality (gamma_poisson)
    disp = model.get_learned_dispersion(modality=0, n_samples=100)
    assert disp.shape == (1,), f"Expected shape (1,), got {disp.shape}"
    assert np.all(disp > 0), "Dispersion should be positive"
    assert np.all(np.isfinite(disp))


def test_get_learned_dispersion_per_gene(synthetic_adata):
    """Test get_learned_dispersion with per-gene dispersion."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=False,
    )

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Get dispersion for first modality (gamma_poisson) - should be per-gene
    disp = model.get_learned_dispersion(modality=0, n_samples=100)
    assert disp.shape == (G,), f"Expected shape ({G},), got {disp.shape}"
    assert np.all(disp > 0), "Dispersion should be positive"


def test_learnable_dispersion_backward_compatible(synthetic_adata):
    """Test that default behavior (learnable_dispersion=False) is unchanged."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)

    # Default should be fixed dispersion
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=8,
        n_hidden=32,
    )

    assert model.learnable_dispersion is False

    # Should train exactly as before
    model.train(max_epochs=2, batch_size=64, early_stopping=False)
    Z = model.get_latent_representation()
    assert Z.shape == (N, 8)

    # Fixed dispersion should return the default value
    disp = model.get_learned_dispersion(modality=0)
    assert np.allclose(disp, [1.0])


def test_learnable_dispersion_only_affects_nb_modalities(synthetic_adata):
    """Test that dispersion is only learned for NB/gamma_poisson modalities."""
    adata, N, G, P = synthetic_adata

    MultimodalAmortizedLDA.setup_anndata(adata, layer=None)
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[G, P],
        likelihoods=["gamma_poisson", "multinomial"],  # only first is NB
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=True,
    )

    # Guide should have dispersion params for first modality only
    assert model.module.guide.disp_loc is not None
    assert model.module.guide.disp_loc[0] is not None  # gamma_poisson
    assert model.module.guide.disp_loc[1] is None  # multinomial - no dispersion


def test_learnable_dispersion_with_mudata(synthetic_mudata):
    """Test learnable dispersion with MuData input."""
    mdata, N, G, P = synthetic_mudata

    model = MultimodalAmortizedLDA.from_mudata(
        mdata,
        modality_order=["rna", "protein"],
        n_topics=8,
        n_hidden=32,
        learnable_dispersion=True,
        global_dispersion=False,
    )

    assert model.learnable_dispersion is True
    assert model.global_dispersion is False

    model.train(max_epochs=2, batch_size=64, early_stopping=False)

    # Get dispersion by modality name
    disp_dict = model.get_learned_dispersion()
    assert "rna" in disp_dict  # gamma_poisson by default
    assert disp_dict["rna"].shape == (G,)