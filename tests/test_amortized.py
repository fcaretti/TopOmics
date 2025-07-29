from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import torch
from scvi._constants import REGISTRY_KEYS

from omics_topic.models import MultimodalAmortizedLDA


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


# ------------------------------------------------------------------ #
# tests
# ------------------------------------------------------------------ #
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
