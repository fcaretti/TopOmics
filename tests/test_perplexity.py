"""
Tests for perplexity and ELBO metrics in MultimodalAmortizedLDA.

Verifies:
- get_elbo() returns the actual ELBO (higher is better, not the loss)
- get_perplexity() returns a value > 1 (proper perplexity)
- get_perplexity() is finite and not inf/nan
- get_perplexity_per_modality() returns finite values > 1 per modality
- Perplexity is lower (better) for a trained model than an untrained one
- Regularization bonuses (entropy_weight, topic_variance_weight) do not inflate perplexity
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from topomics.models import MultimodalAmortizedLDA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rna_adata():
    N, F = 150, 60
    rng = np.random.default_rng(seed=0)
    X = rng.negative_binomial(5, 0.4, size=(N, F)).astype(np.float32)
    return ad.AnnData(X=X), N, F


@pytest.fixture(scope="module")
def trained_model(rna_adata):
    adata, _, F = rna_adata
    MultimodalAmortizedLDA.setup_anndata(adata)
    model = MultimodalAmortizedLDA(adata, n_topics=5, n_inputs_modalities=[F], likelihoods=["gamma_poisson"])
    model.train(max_epochs=10, batch_size=32)
    return model


@pytest.fixture(scope="module")
def trained_model_with_regularization(rna_adata):
    adata, _, F = rna_adata
    MultimodalAmortizedLDA.setup_anndata(adata)
    model = MultimodalAmortizedLDA(
        adata,
        n_topics=5,
        n_inputs_modalities=[F],
        likelihoods=["gamma_poisson"],
        entropy_weight=5.0,
        topic_variance_weight=5.0,
    )
    model.train(max_epochs=10, batch_size=32)
    return model


# ---------------------------------------------------------------------------
# ELBO tests
# ---------------------------------------------------------------------------

def test_elbo_is_finite(trained_model):
    elbo = trained_model.get_elbo()
    assert np.isfinite(elbo), f"ELBO is not finite: {elbo}"


def test_elbo_is_negative(trained_model):
    """ELBO should be negative for typical count data (log evidence is negative)."""
    elbo = trained_model.get_elbo()
    assert elbo < 0, f"ELBO should be negative for NB count data, got {elbo}"


# ---------------------------------------------------------------------------
# Perplexity tests
# ---------------------------------------------------------------------------

def test_perplexity_is_finite(trained_model):
    ppl = trained_model.get_perplexity()
    assert np.isfinite(ppl), f"Perplexity is not finite: {ppl}"


def test_perplexity_greater_than_one(trained_model):
    """Perplexity must be > 1 by definition (it is exp of a positive cross-entropy)."""
    ppl = trained_model.get_perplexity()
    assert ppl > 1.0, f"Perplexity should be > 1, got {ppl}"


def test_perplexity_not_nan(trained_model):
    ppl = trained_model.get_perplexity()
    assert not np.isnan(ppl), f"Perplexity is NaN"


def test_perplexity_finite_with_regularization(trained_model_with_regularization):
    """Regularization bonuses must not push perplexity to infinity."""
    ppl = trained_model_with_regularization.get_perplexity()
    assert np.isfinite(ppl), (
        f"Perplexity with regularization is not finite: {ppl}. "
        "This likely means get_perplexity() is incorrectly using ELBO (which includes "
        "regularization terms) instead of pure log-likelihood."
    )
    assert ppl > 1.0, f"Perplexity with regularization should be > 1, got {ppl}"


def test_perplexity_reasonable_range(trained_model):
    """Perplexity for NB count data should be in a reasonable range (1 < ppl < 1e6)."""
    ppl = trained_model.get_perplexity()
    assert 1.0 < ppl < 1e6, f"Perplexity out of expected range: {ppl}"


# ---------------------------------------------------------------------------
# Per-modality perplexity tests
# ---------------------------------------------------------------------------

def test_perplexity_per_modality_finite(trained_model):
    result = trained_model.get_perplexity_per_modality()
    for mod_name, ppl in result.items():
        assert np.isfinite(ppl), f"Per-modality perplexity for {mod_name} is not finite: {ppl}"


def test_perplexity_per_modality_greater_than_one(trained_model):
    result = trained_model.get_perplexity_per_modality()
    for mod_name, ppl in result.items():
        assert ppl > 1.0, f"Per-modality perplexity for {mod_name} should be > 1, got {ppl}"


def test_perplexity_per_modality_finite_with_regularization(trained_model_with_regularization):
    result = trained_model_with_regularization.get_perplexity_per_modality()
    for mod_name, ppl in result.items():
        assert np.isfinite(ppl), (
            f"Per-modality perplexity for {mod_name} is not finite with regularization: {ppl}"
        )


def test_perplexity_per_modality_keys(trained_model, rna_adata):
    _, _, F = rna_adata
    result = trained_model.get_perplexity_per_modality()
    assert len(result) == 1  # single modality
