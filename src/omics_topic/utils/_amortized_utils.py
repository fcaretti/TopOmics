"""Low-level tensor / graph utilities for the amortized LDA Pyro module.

These helpers were factored out of ``omics_topic.module._amortizedLDA`` so they
can be reused without importing the (heavy) Pyro module classes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

CLAMP_EPS = 10e-6
CLAMP_MAX = 1.0 / CLAMP_EPS


def clamp_symmetric(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        t, nan=0.0, posinf=CLAMP_MAX, neginf=-CLAMP_MAX
    ).clamp(min=-CLAMP_MAX, max=CLAMP_MAX)


def clamp_positive(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        t, nan=CLAMP_EPS, posinf=CLAMP_MAX, neginf=CLAMP_EPS
    ).clamp(min=CLAMP_EPS, max=CLAMP_MAX)


def logistic_normal_approximation(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Laplace approximation parameters (μ, σ) of Logistic-Normal ≈ Dirichlet(α)."""
    K = alpha.shape[-1]
    mu = torch.log(alpha) - torch.log(alpha).sum() / K
    sigma = torch.sqrt((1 - 2 / K) / alpha + torch.sum(1 / alpha) / K**2)
    return mu, sigma


def horseshoe_shrinkage(
    caux: torch.Tensor,     # scalar
    tau: torch.Tensor,      # (K, 1)
    delta: torch.Tensor,    # (F,)
    lambda_: torch.Tensor,  # (K, F)
) -> torch.Tensor:
    """
    Compute regularized horseshoe shrinkage multiplier.

    Based on Carvalho et al. (2010) and Piironen & Vehtari (2017).
    This implements the Finnish horseshoe prior which adds a regularization
    component (caux) to prevent over-shrinkage.

    Parameters
    ----------
    caux : scalar
        Global auxiliary variable for regularization (prevents over-shrinkage)
    tau : (K, 1)
        Per-topic local shrinkage parameter
    delta : (F,)
        Per-feature local shrinkage parameter
    lambda_ : (K, F)
        Per-topic-feature interaction shrinkage parameter

    Returns
    -------
    lambda_tilde : (K, F)
        Effective shrinkage multipliers in [0, 1]

    Notes
    -----
    The formula combines hierarchical shrinkage with regularization:
    λ̃² = (c² * τ² * δ² * λ²) / (c² + τ² * δ² * λ²)

    When τ²δ²λ² >> c²: λ̃ → 1 (no shrinkage, signal preserved)
    When τ²δ²λ² << c²: λ̃ → 0 (strong shrinkage, noise removed)
    """
    caux_sq = caux ** 2
    tau_sq = tau ** 2              # (K, 1)
    delta_sq = delta.unsqueeze(0) ** 2   # (1, F)
    lambda_sq = lambda_ ** 2       # (K, F)

    numerator = caux_sq * tau_sq * delta_sq * lambda_sq
    denominator = caux_sq + tau_sq * delta_sq * lambda_sq

    # Add epsilon for numerical stability
    lambda_tilde = torch.sqrt(numerator / (denominator + 1e-8))
    return lambda_tilde  # (K, F)


def masked_softmax(weights: torch.Tensor, mask: torch.Tensor, dim: int = 0):
    """Softmax **ignoring** masked entries (mask == 0)."""
    weights = clamp_symmetric(weights)
    weights = weights.masked_fill(~mask.bool(), -CLAMP_MAX)
    return torch.softmax(weights, dim=dim)


def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """
    Convert adjacency matrix to PyG edge_index format.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix (dense or sparse)

    Returns
    -------
    edge_index : torch.Tensor
        Edge indices in COO format (2, num_edges)
    """
    if adj.is_sparse:
        adj = adj.coalesce()
        return adj.indices()
    else:
        return adj.nonzero().t().contiguous()


def precompute_sgc(
    x: torch.Tensor,
    adj: sp.spmatrix,
    n_layers: int = 1,
    mode: str = "sign",
) -> torch.Tensor:
    """
    Precompute Simplified Graph Convolution features (STAMP-style).

    Applies fixed, parameter-free neighborhood averaging to produce spatially
    smoothed features that are concatenated with the original.

    Parameters
    ----------
    x : torch.Tensor
        Raw count matrix (n_obs, n_features).
    adj : scipy sparse matrix
        Spatial adjacency matrix (n_obs, n_obs).
    n_layers : int
        Number of SGC hops (default: 1).
    mode : str
        ``"sign"`` — concatenate [X, ÃX, Ã²X, ...] (default).
        ``"sgc"``  — use only the final smoothed version Ã^L X.

    Returns
    -------
    sgc_x : torch.Tensor
        SGC-preprocessed features: (n_obs, n_features * (n_layers + 1)) for "sign" mode.
    """
    # Symmetric normalisation matching STAMP exactly:
    # deg = original_degree + 2, adj = adj + I
    adj_coo = adj.tocoo().astype(np.float64)
    deg = np.asarray(adj_coo.sum(axis=1)).flatten()
    deg = deg + 2  # STAMP adds 2 to degree (not 1)
    adj_with_self = adj_coo + sp.eye(adj_coo.shape[0], format="coo")
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_with_self @ D_inv_sqrt

    # Convert to torch sparse
    adj_norm_coo = adj_norm.astype(np.float32).tocoo()
    indices = torch.tensor(np.vstack([adj_norm_coo.row, adj_norm_coo.col]), dtype=torch.long)
    values = torch.tensor(adj_norm_coo.data, dtype=torch.float32)
    adj_t = torch.sparse_coo_tensor(indices, values, torch.Size(adj_norm_coo.shape))

    # Library-size normalisation to median depth
    ls = x.sum(dim=1, keepdim=True)
    ms = torch.median(x.sum(dim=1))
    xs = x / ls * ms

    # Iterative smoothing
    sgc = [xs]
    for _ in range(n_layers):
        xs = torch.sparse.mm(adj_t, sgc[-1])
        sgc.append(xs)

    if mode == "sgc":
        sgc = [sgc[-1]]

    sgc_x = torch.cat(sgc, dim=1)
    sgc_x = torch.log(sgc_x + 1)
    return sgc_x


class AttentionAggregator(nn.Module):
    """
    SpatialGlue-style single-head attention for mixing M modality encoder outputs.

    Computes per-cell, input-dependent mixing weights alpha (B, M) from the
    encoder means, then applies them as weights in the Gaussian mixture. This
    is more expressive than MoE because the weight of each modality depends on
    what each encoder actually produced for that cell.

    Parameters
    ----------
    n_topics
        Dimensionality of each modality embedding (= n_topics in latent space).
    att_dim
        Projection dimension for the attention key/value (default: 32).

    Notes
    -----
    Missing modalities are handled by masking attention logits to -inf before
    softmax, the same strategy used by masked_softmax in the MoE path. However,
    if all modalities are missing for a cell the result is undefined (shouldn't
    happen in practice). Unlike MoE, this method requires all attending modalities
    to share the same latent dimensionality (n_topics), which is already the case.

    References
    ----------
    Tang et al. (2023), "SpatialGlue: Deciphering multi-modal spatial omics
    integration" – between-modality attention mechanism.
    """

    def __init__(self, n_topics: int, att_dim: int = 32) -> None:
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(n_topics, att_dim))
        self.u_omega = nn.Parameter(torch.empty(att_dim, 1))
        nn.init.xavier_uniform_(self.w_omega.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_omega.unsqueeze(0))

    def forward(self, mus: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute attention mixing weights.

        Parameters
        ----------
        mus : (M, B, K)
            Per-modality encoder means.
        masks : (M, B)
            1 if modality present for that cell, 0 if absent.

        Returns
        -------
        w : (M, B, 1)
            Per-cell mixing weights summing to 1 over M (absent modalities = 0).
        """
        emb = mus.permute(1, 0, 2)              # (B, M, K)
        v = torch.tanh(emb @ self.w_omega)       # (B, M, att_dim)
        vu = (v @ self.u_omega).squeeze(-1)      # (B, M)

        # Mask absent modalities before softmax
        masks_BM = masks.T                       # (B, M)
        vu = vu.masked_fill(~masks_BM.bool(), -CLAMP_MAX)
        alpha = torch.softmax(vu, dim=1)         # (B, M)

        return alpha.T.unsqueeze(-1)             # (M, B, 1)
