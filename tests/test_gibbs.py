from __future__ import annotations

import numpy as np
import pytest
import torch

# ------------------------------------------------------------------ #
# optional deps – skip tests if the module is missing
# ------------------------------------------------------------------ #
try:
    # Adjust this import to your actual package layout if necessary
    from topomics.models.gibbs_model import Gibbs_LDA_Multi
except ModuleNotFoundError:
    Gibbs_LDA_Multi = None  # type: ignore

pytest.importorskip("torch")


# ------------------------------------------------------------------ #
# fixtures
# ------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def synthetic_mdata() -> dict[str, torch.Tensor]:
    """
    200 cells × (40 RNA + 8 protein) counts.

    RNA ~ Poisson(λ=6)   |   ADT ~ Poisson(λ=3)
    """
    torch.manual_seed(0)
    N, G_rna, G_adt = 200, 40, 8
    rna = torch.poisson(torch.full((N, G_rna), 6.0)).long()
    adt = torch.poisson(torch.full((N, G_adt), 3.0)).long()
    return {"rna": rna, "protein": adt}


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _build_model(data, K: int = 6):
    """Return a model initialised *without* the NMF smart-init (keeps tests fast)."""
    return Gibbs_LDA_Multi(
        data,
        n_topics=K,
        device="cpu",
        smart_init=False,  # avoid sklearn/NMF dependency inside CI
    )


# ------------------------------------------------------------------ #
# tests
# ------------------------------------------------------------------ #
@pytest.mark.skipif(Gibbs_LDA_Multi is None, reason="gibbs_model module missing")
def test_initialisation_shapes(synthetic_mdata):
    """Shapes of θ and Λ/Φ match expectations right after construction."""
    K = 5
    model = _build_model(synthetic_mdata, K)

    assert model.C == 200
    assert model.K == K
    assert model.theta.shape == (200, K)
    for m in ("rna", "protein"):
        assert m in model.lambda_
        # rna has 40 genes, protein has 8 features
        expected_G = 40 if m == "rna" else 8
        assert model.lambda_[m].shape == (K, expected_G)


@pytest.mark.skipif(Gibbs_LDA_Multi is None, reason="gibbs_model module missing")
def test_log_likelihood_is_finite(synthetic_mdata):
    """The joint log-likelihood is finite on the initial parameters."""
    model = _build_model(synthetic_mdata)
    ll = model._log_likelihood()
    assert torch.isfinite(ll)


@pytest.mark.skipif(Gibbs_LDA_Multi is None, reason="gibbs_model module missing")
def test_fit_runs_and_accessors(synthetic_mdata, monkeypatch):
    """
    Run a *very* short Gibbs chain and check that cached samples
    and public accessors behave.
    """
    model = _build_model(synthetic_mdata, K=4)

    # silence tqdm in test logs
    monkeypatch.setattr("topomics.models.gibbs_model.tqdm", lambda x, *a, **k: x)

    model.fit(
        batch_size=64,
        n_samples=1,  # keep one posterior sample
        thin=1,
        initial_burnin=10,
        burnin=0,
        progress=False,
        ll_every=5,
    )

    # cached arrays have the right shape
    assert model.theta_samples.shape == (1, 200, 4)
    assert model.lambda_samples["rna"].shape == (1, 4, 40)

    # accessor returns row-normalised Θ
    theta_mean = model.get_cell_topic_dist()
    assert theta_mean.shape == (200, 4)
    row_sums = theta_mean.sum(1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)

    # feature-topic accessor works for each modality
    for mod in ("rna", "protein"):
        mat = model.get_feature_topic_dist(mod)
        G = 40 if mod == "rna" else 8
        assert mat.shape == (4, G)
