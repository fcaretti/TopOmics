"""
Tests for two new optional architecture features:

1. Attention aggregation (aggregation_type="attention")
   - SpatialGlue-style single-head attention for mixing modality encoder outputs
   - Input-dependent per-cell mixing weights rather than global MoE weights

2. Pre-GCN FC layers (gcn_n_pre_layers > 0)
   - Optional FC projection applied to raw inputs BEFORE graph convolution
   - Lets the GCN operate in a learned feature space rather than raw count space
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import anndata as ad
import mudata as mu

from omics_topic.models import MultimodalAmortizedLDA


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rna_adata():
    N, F = 120, 50
    rng = np.random.default_rng(seed=7)
    X = rng.negative_binomial(5, 0.4, size=(N, F)).astype(np.float32)
    return ad.AnnData(X=X), N, F


@pytest.fixture(scope="module")
def multimodal_adata():
    N, F1, F2 = 120, 40, 30
    rng = np.random.default_rng(seed=8)
    X = np.hstack([
        rng.negative_binomial(5, 0.4, size=(N, F1)),
        rng.negative_binomial(3, 0.5, size=(N, F2)),
    ]).astype(np.float32)
    return ad.AnnData(X=X), N, F1, F2


@pytest.fixture(scope="module")
def spatial_adata():
    """Small AnnData with a toy spatial graph stored in uns for GCN tests."""
    N, F = 80, 40
    rng = np.random.default_rng(seed=9)
    X = rng.negative_binomial(5, 0.4, size=(N, F)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Build a simple ring graph adjacency and store directly in uns
    # (the model __init__ reads adata.uns["_spatial_graph"])
    row = list(range(N)) + list(range(1, N)) + [0]
    col = list(range(1, N)) + [0] + list(range(N))
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    adata.uns["_spatial_graph"] = {"adjacency": adj, "key": "_spatial_graph"}
    return adata, N, F


# ===========================================================================
# Feature 1: Attention aggregation
# ===========================================================================

class TestAttentionAggregation:

    def test_attention_single_modality_runs(self, rna_adata):
        """Attention aggregation should work with a single modality."""
        adata, _, F = rna_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
            aggregation_type="attention",
        )
        model.train(max_epochs=5, batch_size=32)
        theta = model.get_cell_topic_dist()
        assert theta.shape == (adata.n_obs, 5)

    def test_attention_multimodal_runs(self, multimodal_adata):
        """Attention aggregation should work with multiple modalities."""
        adata, N, F1, F2 = multimodal_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F1, F2],
            likelihoods=["gamma_poisson", "gamma_poisson"],
            aggregation_type="attention",
            att_dim=16,
        )
        model.train(max_epochs=5, batch_size=32)
        theta = model.get_cell_topic_dist()
        assert theta.shape == (N, 5)

    def test_attention_weights_sum_to_one(self, multimodal_adata):
        """Attention weights should sum to 1 per cell across modalities."""
        from omics_topic.module._amortizedLDA import AttentionAggregator
        adata, N, F1, F2 = multimodal_adata
        agg = AttentionAggregator(n_topics=5, att_dim=16)

        M, B, K = 2, 10, 5
        mus = torch.randn(M, B, K)
        masks = torch.ones(M, B)  # all modalities present

        w = agg(mus, masks)  # (M, B, 1)
        assert w.shape == (M, B, 1)
        # Weights should sum to ~1 over M dimension
        weight_sum = w.squeeze(-1).sum(dim=0)  # (B,)
        assert torch.allclose(weight_sum, torch.ones(B), atol=1e-5)

    def test_attention_masks_absent_modality(self):
        """Absent modalities (mask=0) should receive zero attention weight."""
        from omics_topic.module._amortizedLDA import AttentionAggregator
        agg = AttentionAggregator(n_topics=5, att_dim=16)

        M, B, K = 2, 4, 5
        mus = torch.randn(M, B, K)
        # First modality absent for all cells
        masks = torch.ones(M, B)
        masks[0, :] = 0.0

        w = agg(mus, masks)  # (M, B, 1)
        # Absent modality should have near-zero weight
        assert (w[0] < 1e-3).all(), f"Absent modality has non-zero weight: {w[0]}"
        # Present modality should receive all the weight
        assert torch.allclose(w[1].squeeze(-1), torch.ones(B), atol=1e-3)

    def test_attention_invalid_type_raises(self, rna_adata):
        adata, _, F = rna_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        with pytest.raises(ValueError, match="aggregation_type"):
            MultimodalAmortizedLDA(
                adata,
                n_topics=5,
                n_inputs_modalities=[F],
                likelihoods=["gamma_poisson"],
                aggregation_type="invalid_type",
            )

    def test_attention_vs_moe_different_params(self, multimodal_adata):
        """Attention and MoE models should have different parameter counts."""
        adata, N, F1, F2 = multimodal_adata
        MultimodalAmortizedLDA.setup_anndata(adata)

        model_moe = MultimodalAmortizedLDA(
            adata, n_topics=5, n_inputs_modalities=[F1, F2],
            likelihoods=["gamma_poisson", "gamma_poisson"], aggregation_type="moe",
        )
        model_att = MultimodalAmortizedLDA(
            adata, n_topics=5, n_inputs_modalities=[F1, F2],
            likelihoods=["gamma_poisson", "gamma_poisson"],
            aggregation_type="attention", att_dim=16,
        )
        n_params_moe = sum(p.numel() for p in model_moe.module.parameters())
        n_params_att = sum(p.numel() for p in model_att.module.parameters())
        # Attention model has extra W_omega and u_omega parameters
        assert n_params_att > n_params_moe


# ===========================================================================
# Feature 2: Pre-GCN FC layers
# ===========================================================================

class TestPreGCNLayers:

    def test_pre_gcn_layers_spatial_runs(self, spatial_adata):
        """Spatial model with gcn_n_pre_layers=1 should train without error."""
        adata, _, F = spatial_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
            gcn_n_pre_layers=1,
        )
        model.train(max_epochs=5, batch_size=32)
        theta = model.get_cell_topic_dist()
        assert theta.shape == (adata.n_obs, 5)

    def test_pre_gcn_layers_zero_baseline(self, spatial_adata):
        """gcn_n_pre_layers=0 should have no pre_gcn_fc (None)."""
        adata, _, F = spatial_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
            gcn_n_pre_layers=0,
        )
        model.train(max_epochs=5, batch_size=32)
        assert model.module._guide.gcn_encoders[0].pre_gcn_fc is None

    def test_pre_gcn_layers_creates_module(self, spatial_adata):
        """gcn_n_pre_layers=1 should create a pre_gcn_fc module in GCNEncoder."""
        adata, _, F = spatial_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
            gcn_n_pre_layers=1,
        )
        gcn_enc = model.module._guide.gcn_encoders[0]
        assert gcn_enc.pre_gcn_fc is not None

    def test_pre_gcn_layers_more_params_than_baseline(self, spatial_adata):
        """Model with pre-GCN layers should have more parameters."""
        adata, _, F = spatial_adata
        MultimodalAmortizedLDA.setup_anndata(adata)

        model_base = MultimodalAmortizedLDA(
            adata, n_topics=5, n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"], gcn_n_pre_layers=0,
        )
        model_pre = MultimodalAmortizedLDA(
            adata, n_topics=5, n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"], gcn_n_pre_layers=1,
        )
        n_base = sum(p.numel() for p in model_base.module.parameters())
        n_pre = sum(p.numel() for p in model_pre.module.parameters())
        assert n_pre > n_base

    def test_pre_gcn_not_spatial_has_no_effect(self, rna_adata):
        """gcn_n_pre_layers is accepted but has no effect in non-spatial models."""
        adata, _, F = rna_adata
        MultimodalAmortizedLDA.setup_anndata(adata)
        model = MultimodalAmortizedLDA(
            adata,
            n_topics=5,
            n_inputs_modalities=[F],
            likelihoods=["gamma_poisson"],
            gcn_n_pre_layers=2,   # ignored when spatial=False
        )
        model.train(max_epochs=3, batch_size=32)
        assert model.module._guide.gcn_encoders is None
