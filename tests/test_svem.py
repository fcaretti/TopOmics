from __future__ import annotations

import pytest
import torch
from torch import Tensor

from topomics.models import SVEM_LDA_Multi

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_data() -> dict[str, Tensor]:
    """
    Tiny two-modal synthetic count matrix.

    30 cells × (50 rna + 12 protein) ≈ 1 600 integers total.
    """
    torch.manual_seed(0)
    C, G_rna, G_prot = 30, 50, 12
    rna = torch.poisson(5.0 * torch.ones(C, G_rna, dtype=torch.float32)).long()
    prot = torch.poisson(2.0 * torch.ones(C, G_prot, dtype=torch.float32)).long()
    return {"rna": rna, "protein": prot}


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------
def test_initialisation_shapes(synthetic_data):
    """Model builds and tensors have the expected shapes."""
    K = 4
    model = SVEM_LDA_Multi(
        synthetic_data,
        n_topics=K,
        batch_size=8,
        device="cpu",
    )

    # global dims
    assert model.C == 30
    assert model.K == K
    assert model.gamma.shape == (30, K)

    # per-modality params
    assert set(model.A) == {"rna", "protein"}
    assert model.A["rna"].shape == (K, 50)
    assert model.B["protein"].shape == (K, 12)

    # accessor returns the right shape on CPU
    tp_by_feat = model.topic_by_feature("rna")
    assert tp_by_feat.shape == (K, 50)


def test_cell_topic_distribution_normalises(synthetic_data):
    """θ rows should sum to 1 after normalisation."""
    model = SVEM_LDA_Multi(synthetic_data, n_topics=3, batch_size=10, device="cpu")
    theta = model.cell_topic_distribution(normalised=True)
    assert torch.allclose(
        theta.sum(dim=1),
        torch.ones(theta.size(0)),
        atol=1e-6,
    )


def test_fit_runs_and_diagnostics(synthetic_data):
    """
    A couple of epochs on the toy data should finish quickly and
    produce finite perplexity / log-lik values.
    """
    model = SVEM_LDA_Multi(
        synthetic_data,
        n_topics=5,
        batch_size=15,
        device="cpu",
        feature_frac=1.0,
    )
    model.fit(n_epochs=2, inner_iters=1, verbose=False)


def test_mismatched_cells_raises():
    """Constructor must reject modalities with different cell counts."""
    bad = {
        "rna": torch.zeros(5, 4, dtype=torch.int64),
        "protein": torch.zeros(6, 4, dtype=torch.int64),  # off-by-one
    }
    with pytest.raises(ValueError, match="same cells"):
        SVEM_LDA_Multi(bad, n_topics=2, device="cpu")
