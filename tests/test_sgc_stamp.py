"""Tests for STAMP-style SGC spatial mode in TopOmics."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from omics_topic.module._amortizedLDA import precompute_sgc, SGCEncoder


# ---------------------------------------------------------------------------
# precompute_sgc
# ---------------------------------------------------------------------------

def _make_ring_adj(n: int) -> sp.csr_matrix:
    """Simple ring graph: each node connected to its two neighbours."""
    row = list(range(n)) + list(range(n))
    col = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    data = [1.0] * (2 * n)
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


def test_precompute_sgc_sign_shape():
    n, g = 50, 10
    x = torch.rand(n, g) * 100
    adj = _make_ring_adj(n)
    sgc_x = precompute_sgc(x, adj, n_layers=1, mode="sign")
    assert sgc_x.shape == (n, g * 2), f"Expected ({n}, {g*2}), got {sgc_x.shape}"


def test_precompute_sgc_multilayer():
    n, g = 50, 10
    x = torch.rand(n, g) * 100
    adj = _make_ring_adj(n)
    sgc_x = precompute_sgc(x, adj, n_layers=3, mode="sign")
    assert sgc_x.shape == (n, g * 4)


def test_precompute_sgc_mode():
    n, g = 50, 10
    x = torch.rand(n, g) * 100
    adj = _make_ring_adj(n)
    sgc_x = precompute_sgc(x, adj, n_layers=2, mode="sgc")
    # "sgc" mode returns only the final layer
    assert sgc_x.shape == (n, g)


def test_precompute_sgc_values_non_negative():
    x = torch.rand(30, 5) * 100
    adj = _make_ring_adj(30)
    sgc_x = precompute_sgc(x, adj, n_layers=1)
    assert (sgc_x >= 0).all(), "SGC features should be non-negative (log(x+1))"


def test_precompute_sgc_smoothing_reduces_variance():
    """Smoothed features should have lower variance than original."""
    n, g = 100, 5
    x = torch.rand(n, g) * 100
    adj = _make_ring_adj(n)
    sgc_x = precompute_sgc(x, adj, n_layers=1, mode="sign")
    original = sgc_x[:, :g]
    smoothed = sgc_x[:, g:]
    assert smoothed.var(dim=0).mean() < original.var(dim=0).mean()


# ---------------------------------------------------------------------------
# SGCEncoder
# ---------------------------------------------------------------------------

def test_sgcencoder_forward():
    n, g, k, h = 64, 20, 10, 32
    enc = SGCEncoder(n_genes=g, n_sgc_layers=1, n_hidden=h, n_topics=k)
    sgc_x = torch.randn(n, g * 2)  # sign mode: g * (1+1)
    enc.set_sgc_data(sgc_x)
    batch_idx = torch.arange(16)
    q, _ = enc(None, None, batch_indices=batch_idx)
    assert q.loc.shape == (16, k)
    assert q.scale.shape == (16, k)


def test_sgcencoder_per_batch_bn():
    n, g, k, h = 64, 20, 10, 32
    enc = SGCEncoder(n_genes=g, n_sgc_layers=1, n_hidden=h, n_topics=k, n_batches=3)
    sgc_x = torch.randn(n, g * 2)
    enc.set_sgc_data(sgc_x)
    batch_idx = torch.arange(16)
    cat_list = [torch.randint(0, 3, (16, 1))]
    q, _ = enc(None, None, batch_indices=batch_idx, cat_list=cat_list)
    assert q.loc.shape == (16, k)


def test_sgcencoder_no_batch_bn():
    """Without per-batch BN (n_batches<=1), should work without cat_list."""
    n, g, k, h = 64, 20, 10, 32
    enc = SGCEncoder(n_genes=g, n_sgc_layers=1, n_hidden=h, n_topics=k, n_batches=0)
    sgc_x = torch.randn(n, g * 2)
    enc.set_sgc_data(sgc_x)
    batch_idx = torch.arange(16)
    q, _ = enc(None, None, batch_indices=batch_idx)
    assert q.loc.shape == (16, k)


def test_sgcencoder_not_initialized_raises():
    enc = SGCEncoder(n_genes=10, n_sgc_layers=1, n_hidden=32, n_topics=5)
    with pytest.raises(ValueError, match="not initialized"):
        enc(None, None, batch_indices=torch.arange(4))


# ---------------------------------------------------------------------------
# Integration: full model with spatial_mode="sgc"
# ---------------------------------------------------------------------------

def test_sgc_model_creation():
    """Test that the full model can be created with spatial_mode='sgc'."""
    from omics_topic import MultimodalAmortizedLDA
    import anndata
    import squidpy as sq

    np.random.seed(42)
    n_obs, n_genes = 100, 50
    X = np.random.poisson(5, (n_obs, n_genes)).astype(np.float32)
    coords = np.random.rand(n_obs, 2) * 100

    adata = anndata.AnnData(X=sp.csr_matrix(X))
    adata.obsm["spatial"] = coords
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

    MultimodalAmortizedLDA.setup_anndata(
        adata, spatial_keys="spatial_connectivities"
    )
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[n_genes],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
        spatial_mode="sgc",
        sgc_n_layers=1,
        weight_mode="equal",
        use_feature_background=False,
        entropy_weight=0.0,
        topic_variance_weight=0.0,
    )
    assert model.module.spatial_mode == "sgc"


def test_sgc_model_train_smoke():
    """Smoke test: train for a few steps with spatial_mode='sgc'."""
    from omics_topic import MultimodalAmortizedLDA
    import anndata
    import squidpy as sq

    np.random.seed(42)
    n_obs, n_genes = 100, 50
    X = np.random.poisson(5, (n_obs, n_genes)).astype(np.float32)
    coords = np.random.rand(n_obs, 2) * 100

    adata = anndata.AnnData(X=sp.csr_matrix(X))
    adata.obsm["spatial"] = coords
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

    MultimodalAmortizedLDA.setup_anndata(
        adata, spatial_keys="spatial_connectivities"
    )
    model = MultimodalAmortizedLDA(
        adata,
        n_inputs_modalities=[n_genes],
        likelihoods=["gamma_poisson"],
        n_topics=5,
        n_hidden=32,
        spatial_mode="sgc",
        sgc_n_layers=1,
        weight_mode="equal",
        use_feature_background=False,
        entropy_weight=0.0,
        topic_variance_weight=0.0,
    )
    model.train(max_epochs=3, batch_size=50, train_size=0.9)
    theta = model.get_latent_representation(adata, batch_size=n_obs)
    assert theta.shape == (n_obs, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
