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