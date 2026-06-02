# multimodal_lda_module.py
"""
The file declares three public objects
-------------------------------------
* `MultimodalLDAPyroModel`   – generative process with modality plate & mixed likelihoods.
* `MultimodalLDAPyroGuide`   – per-modality encoders + combined θₙ posterior + per-modality ϕₖ,ₘ posterior.
* `MultimodalAmortizedLDAPyroModule` – wrapper pairing the two above and exposing
  helper utilities (`topic_by_feature`, `get_topic_distribution`, `get_elbo`).

A higher-level `ModelClass` wrapper (*MultimodalAmortizedLDA*) is provided at the bottom so that
users get the usual `train()`/`get_latent_representation()`/etc. API.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import Trace_ELBO
from pyro.nn import PyroModule
from scvi._constants import REGISTRY_KEYS
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import k_hop_subgraph

logger = logging.getLogger(__name__)

from omics_topic.utils._amortized_utils import (
    CLAMP_EPS,
    CLAMP_MAX,
    adjacency_to_edge_index,
    clamp_positive,
    clamp_symmetric,
    horseshoe_shrinkage,
    logistic_normal_approximation,
    masked_softmax,
    precompute_sgc,
)


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


class GCNEncoder(nn.Module):
    """
    GCN/GAT encoder for spatial data with optional multi-layer graph
    convolution and an explicit skip-connection that preserves node identity.

    The skip path uses the scvi Encoder structure (FCLayers -> mean/var). When
    the skip weight is exactly 1, the graph branch is ignored and the encoder
    reduces to the scvi implementation.

        h = alpha * FCLayers(x) + (1 - alpha) * GNN(x, A)

    where alpha is a learnable scalar initialized so that alpha≈0.7.
    """

    def __init__(
        self,
        n_in: int,
        n_topics: int,
        n_hidden: int,
        gcn_n_layers: int = 1,
        gcn_hidden_dims: list[int] | None = None,
        gcn_n_pre_layers: int = 0,     # FC layers before graph convolution (0 = disabled)
        dropout: float = 0.1,
        add_self_loops: bool = True,
        conv_type: str = "GATv2Conv",  # or 'GATv2Conv'
        heads: int = 4,              # used only by GAT
        normalize: bool = True,      # used only by GCN
        concat: bool = True,         # multi-head strategy for GAT
        alpha_init: float = 0.7,     # initial weight for self signal
        use_learned_alpha: bool = True,  # if False, uses fixed alpha_init
        var_eps: float = 1e-4,
        var_activation=None,
        sampling: str = "approx",    # "approx" (stochastic neighbor sampling) or "exact" (k_hop_subgraph)
        fan_out: list[int] | None = None,  # per-layer fan-out for neighbor sampling
        conv_first: bool = False,    # if True, run GCN on raw features then FC on seeds only
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_in = n_in
        self.n_topics = n_topics
        self.n_hidden = n_hidden
        self.var_eps = var_eps
        self.gcn_n_layers = gcn_n_layers
        self.gcn_hidden_dims = gcn_hidden_dims
        self.gcn_dropout = dropout
        # Use softplus instead of exp for variance - bounded gradients prevent explosion
        self.var_activation = F.softplus if var_activation is None else var_activation

        if gcn_n_layers < 1:
            raise ValueError("gcn_n_layers must be >= 1.")

        # Optional FC projection applied to x_full BEFORE graph convolution.
        # When gcn_n_pre_layers > 0 the GCN operates in learned feature space
        # rather than raw count space — beneficial for high-dimensional inputs.
        # Note: these share the same n_hidden width as the self-path encoder.
        fc_n_layers_kwarg = kwargs.pop("n_layers", 1)
        fc_dropout_rate = kwargs.pop("dropout_rate", dropout)
        n_cats_per_cov_kwarg = kwargs.pop("n_cats_per_cov", None)
        n_continuous_cov_kwarg = kwargs.pop("n_continuous_cov", 0)

        if gcn_n_pre_layers > 0:
            self.pre_gcn_fc = FCLayers(
                n_in=n_in,
                n_out=n_hidden,
                n_layers=gcn_n_pre_layers,
                n_hidden=n_hidden,
                dropout_rate=fc_dropout_rate,
            )
            gcn_input_dim = n_hidden  # GCN receives projected features
        else:
            self.pre_gcn_fc = None
            gcn_input_dim = n_in     # GCN receives raw (normalized) counts

        # Neighbor aggregation (graph convolution)
        self.conv_type = conv_type
        if gcn_hidden_dims is None:
            gcn_hidden_dims = [n_hidden] * gcn_n_layers
        elif len(gcn_hidden_dims) != gcn_n_layers:
            raise ValueError("gcn_hidden_dims must have length gcn_n_layers.")

        def _make_conv(in_dim: int, out_dim: int):
            if conv_type == "GATv2Conv":
                out_channels = out_dim if not concat else max(1, out_dim // heads)
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    heads=heads,
                    add_self_loops=add_self_loops,
                    dropout=dropout,
                    concat=concat,
                )
                conv_out_dim = (heads * out_channels) if concat else out_dim
            else:
                conv = GCNConv(
                    in_dim,
                    out_dim,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                )
                conv_out_dim = out_dim
            return conv, conv_out_dim

        self.convs = nn.ModuleList()
        self.conv_bns = nn.ModuleList()
        self.conv_out_dims = []

        prev_dim = gcn_input_dim  # may be n_in or n_hidden depending on pre_gcn_fc
        for hidden_dim in gcn_hidden_dims:
            conv, conv_out_dim = _make_conv(prev_dim, hidden_dim)
            self.convs.append(conv)
            self.conv_bns.append(nn.BatchNorm1d(conv_out_dim))
            self.conv_out_dims.append(conv_out_dim)
            prev_dim = conv_out_dim

        # Covariate parameters (already popped from kwargs above)
        self.n_cats_per_cov = n_cats_per_cov_kwarg
        self.n_continuous_cov = n_continuous_cov_kwarg
        self.use_covariates = (self.n_cats_per_cov is not None and len(self.n_cats_per_cov) > 0) or self.n_continuous_cov > 0

        # scvi-style encoder for the self signal (n_layers / dropout_rate already popped)
        self.encoder = FCLayers(
            n_in=n_in + self.n_continuous_cov,
            n_out=n_hidden,
            n_cat_list=self.n_cats_per_cov,
            n_layers=fc_n_layers_kwarg,
            n_hidden=n_hidden,
            dropout_rate=fc_dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_topics)
        self.var_encoder = nn.Linear(n_hidden, n_topics)

        # If GAT concat produced a different hidden size, align it
        final_conv_dim = self.conv_out_dims[-1]
        if final_conv_dim != n_hidden:
            self.nei_proj = nn.Linear(final_conv_dim, n_hidden, bias=False)
        else:
            self.nei_proj = nn.Identity()

        # Alpha mixing in [0, 1]
        # Use sigmoid transformation for proper gradient flow (avoids gradient=0 at boundaries)
        self.use_learned_alpha = use_learned_alpha
        alpha_init = float(alpha_init)
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError("alpha_init must be in [0, 1].")
        # Store alpha in logit space, apply sigmoid during forward pass
        # Use large finite values for boundary cases (sigmoid(±20) ≈ 0/1)
        if alpha_init == 0.0:
            alpha_logit = torch.tensor(-20.0, dtype=torch.float32)
        elif alpha_init == 1.0:
            alpha_logit = torch.tensor(20.0, dtype=torch.float32)
        else:
            alpha_logit = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
        if use_learned_alpha:
            self._alpha_logit = nn.Parameter(alpha_logit.clone().detach())
        else:
            self.register_buffer("_alpha_logit", alpha_logit.clone().detach())

        # Full-graph data stored on CPU (NOT as buffers) to avoid GPU OOM
        # on large graphs (e.g. VisiumHD).  Subgraph extraction via
        # k_hop_subgraph moves only the needed neighbourhood to GPU each step.
        self.x_full: torch.Tensor | None = None
        self.edge_index_full: torch.Tensor | None = None
        self.num_nodes: int = 0
        self.num_hops: int = gcn_n_layers
        self._graph_initialized = False

        # Neighbor sampling mode
        self.sampling = sampling
        self.fan_out = fan_out
        self._neighbor_sampler = None  # built lazily in set_full_graph_data

        # Conv-first mode: run GCN on raw features, then FC only on seeds
        self.conv_first = conv_first

    @property
    def alpha(self) -> float:
        """Get the current alpha value (skip connection weight for self features)."""
        return torch.sigmoid(self._alpha_logit).item()

    @alpha.setter
    def alpha(self, value: float):
        """
        Set the alpha value (skip connection weight for self features).

        Parameters
        ----------
        value : float
            Alpha value in [0, 1].
        """
        value = float(value)
        if value <= 0.0:
            logit_value = torch.tensor(-20.0, dtype=self._alpha_logit.dtype, device=self._alpha_logit.device)
        elif value >= 1.0:
            logit_value = torch.tensor(20.0, dtype=self._alpha_logit.dtype, device=self._alpha_logit.device)
        else:
            logit_value = torch.logit(torch.tensor(value, dtype=self._alpha_logit.dtype, device=self._alpha_logit.device))
        self._alpha_logit.data.copy_(logit_value)

    def set_full_graph_data(self, x_full: torch.Tensor, edge_index_full: torch.Tensor):
        """
        Initialize full graph data for minibatch-subgraph training.

        Both tensors are kept on **CPU**; only the k-hop neighbourhood of
        each minibatch is moved to the model device during ``forward()``.

        Parameters
        ----------
        x_full : torch.Tensor
            Full feature matrix [n_obs, n_in]
        edge_index_full : torch.Tensor
            COO edge index [2, n_edges]
        """
        if x_full.dim() != 2 or x_full.size(1) != self.n_in:
            raise ValueError(f"x_full must have shape [n_obs, {self.n_in}], got {tuple(x_full.shape)}")
        if edge_index_full.dim() != 2 or edge_index_full.size(0) != 2:
            raise ValueError(f"edge_index_full must have shape [2, n_edges], got {tuple(edge_index_full.shape)}")

        # Store on CPU to avoid GPU memory pressure on large graphs
        self.x_full = x_full.detach().cpu()
        self.edge_index_full = edge_index_full.detach().cpu().contiguous()
        self.num_nodes = x_full.shape[0]
        self._graph_initialized = True

        # Build neighbor sampler if requested
        if self.sampling == "approx":
            from omics_topic.utils.neighbor_sampler import NeighborSampler as _NS
            fan_out = self.fan_out if self.fan_out is not None else [10] * self.num_hops
            self._neighbor_sampler = _NS(
                self.edge_index_full, self.num_nodes, fan_out=fan_out,
            )
            logger.info(
                f"GCN graph initialized (CPU): {x_full.shape[0]} cells, "
                f"{edge_index_full.shape[1]} edges, neighbor sampling fan_out={fan_out}"
            )
        else:
            logger.info(
                f"GCN graph initialized (CPU): {x_full.shape[0]} cells, "
                f"{edge_index_full.shape[1]} edges, {self.num_hops}-hop exact subgraph sampling"
            )

    def _mix_self_and_neighbors(self, h_self: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """
        Mix identity/self and neighbor-aggregated features.
        """
        h_nei = self.nei_proj(h_nei)

        alpha = torch.sigmoid(self._alpha_logit)  # scalar in (0,1)
        return alpha * h_self + (1.0 - alpha) * h_nei

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        cat_list: list[torch.Tensor] | None = None,
    ):
        """
        Forward pass using k-hop subgraph sampling.

        Instead of computing on the full graph and subsetting, we extract the
        k-hop neighbourhood around ``batch_indices`` (global cell ids), move
        only that subgraph to the model device, run the GCN, and return the
        seed-node representations.

        Parameters
        ----------
        x : torch.Tensor
            Ignored (features come from stored ``x_full``).
        edge_index : torch.Tensor
            Ignored (edges come from stored ``edge_index_full``).
        batch_indices : torch.Tensor
            **Global** cell indices for the current minibatch.
        cat_list : list[torch.Tensor], optional
            Categorical covariate tensors (not used in current graph path).

        Returns
        -------
        pyro.distributions.Normal
            q(z_topic_logits | x, A) as a Normal over logits
        None
            Placeholder for compatibility
        """
        if not self._graph_initialized:
            raise ValueError("Graph should be precomputed: call set_full_graph_data(...) before training.")

        if batch_indices is None:
            raise ValueError(
                "batch_indices (global cell indices) required. "
                "This should be provided automatically by the guide."
            )

        device = self.mean_encoder.weight.device  # model's compute device

        # ---- STEP 1: extract subgraph on CPU ----
        batch_indices_cpu = batch_indices.detach().cpu().long()
        if self._neighbor_sampler is not None:
            subset, sub_edge_index, mapping = self._neighbor_sampler.sample(
                batch_indices_cpu
            )
        else:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                batch_indices_cpu,
                num_hops=self.num_hops,
                edge_index=self.edge_index_full,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
            )

        # Move only the subgraph to the compute device
        x_sub = clamp_symmetric(self.x_full[subset].to(device))
        sub_edge_index = sub_edge_index.to(device)
        mapping = mapping.to(device)

        # ---- STEP 2: compute on subgraph ----
        if self.conv_first:
            # Conv-first: GCN on raw features (all subgraph nodes), then FC on seeds only.
            # This is much faster because FC runs on batch_size nodes instead of
            # the full subgraph (~30x savings on large graphs).
            if self.pre_gcn_fc is not None:
                h_nei = clamp_symmetric(self.pre_gcn_fc(x_sub))
            else:
                h_nei = x_sub
            for i, (conv, bn) in enumerate(zip(self.convs, self.conv_bns, strict=False)):
                h_nei = conv(h_nei, sub_edge_index)
                h_nei = clamp_symmetric(bn(h_nei))
                if i < len(self.convs) - 1:
                    h_nei = clamp_symmetric(F.relu(h_nei))
                    if self.gcn_dropout > 0:
                        h_nei = F.dropout(
                            h_nei, p=self.gcn_dropout, training=self.training
                        )

            # Extract seeds BEFORE FC — FC only runs on batch_size nodes
            h_nei_seeds = clamp_symmetric(self.nei_proj(h_nei[mapping]))
            x_seeds = x_sub[mapping]
            h_self_seeds = clamp_symmetric(self.encoder(x_seeds))

            alpha = torch.sigmoid(self._alpha_logit)
            h = clamp_symmetric(alpha * h_self_seeds + (1.0 - alpha) * h_nei_seeds)
        else:
            # Default: FC and GCN both run on full subgraph, then extract seeds.
            # Self path (scvi-style FCLayers)
            h_self = clamp_symmetric(self.encoder(x_sub))

            # Neighbor aggregation path (stacked graph convs with BatchNorm)
            if self.pre_gcn_fc is not None:
                h_nei = clamp_symmetric(self.pre_gcn_fc(x_sub))
            else:
                h_nei = x_sub
            for i, (conv, bn) in enumerate(zip(self.convs, self.conv_bns, strict=False)):
                h_nei = conv(h_nei, sub_edge_index)
                h_nei = clamp_symmetric(bn(h_nei))
                if i < len(self.convs) - 1:
                    h_nei = clamp_symmetric(F.relu(h_nei))
                    if self.gcn_dropout > 0:
                        h_nei = F.dropout(
                            h_nei, p=self.gcn_dropout, training=self.training
                        )

            # Mix self + neighbors (skip connection)
            h_sub = clamp_symmetric(self._mix_self_and_neighbors(h_self, h_nei))

            # Extract seed-node representations
            h = clamp_symmetric(h_sub[mapping])  # [batch_size, n_hidden]

        # scvi-style mean/variance heads
        q_m = clamp_symmetric(self.mean_encoder(h))
        raw_q_v = clamp_positive(self.var_activation(self.var_encoder(h)))
        q_v = clamp_positive(raw_q_v + self.var_eps)

        # NaN detection warning
        if torch.isnan(q_m).any() or torch.isnan(q_v).any():
            nan_info = []
            if torch.isnan(h_self).any():
                nan_info.append("h_self")
            if torch.isnan(h_nei).any():
                nan_info.append("h_nei")
            if torch.isnan(h_sub).any():
                nan_info.append("h_sub")
            if torch.isnan(q_m).any():
                nan_info.append("q_m (mean)")
            if torch.isnan(q_v).any():
                nan_info.append("q_v (var)")
            logger.warning(
                f"NaN detected in GCNEncoder forward pass! "
                f"Affected tensors: {', '.join(nan_info)}. "
                f"This usually indicates gradient explosion. Try reducing learning rate."
            )

        return dist.Normal(q_m, q_v.sqrt()), None


# --------------------------------------------------------------------------------------------------
# SGC (Simplified Graph Convolution) preprocessing and encoder — STAMP-style spatial handling
# --------------------------------------------------------------------------------------------------


class SGCEncoder(nn.Module):
    """
    STAMP-style encoder: MLP on precomputed SGC features with optional
    per-batch BatchNorm.

    The SGC features are precomputed once before training (fixed, parameter-free).
    The encoder is a simple MLP that maps [X, ÃX, ...] → (μ, σ) for topic logits.

    Parameters
    ----------
    n_genes : int
        Number of features per modality.
    n_sgc_layers : int
        Number of SGC hops (determines input width: n_genes * (n_sgc_layers + 1) for "sign" mode).
    n_hidden : int
        Hidden layer size.
    n_topics : int
        Output dimension (number of topics).
    dropout : float
        Dropout rate.
    n_batches : int
        Number of batch categories for per-batch BatchNorm. If <= 1, uses standard BatchNorm.
    var_eps : float
        Minimum variance.
    """

    def __init__(
        self,
        n_genes: int,
        n_sgc_layers: int,
        n_hidden: int,
        n_topics: int,
        dropout: float = 0.0,
        n_batches: int = 0,
        var_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_topics = n_topics
        self.var_eps = var_eps
        self.n_batches = n_batches

        sgc_input_dim = n_genes * (n_sgc_layers + 1)

        # Base network — exact replica of STAMP's MLPEncoderMVN:
        # Dropout → Linear → BN → ReLU → Dropout
        self.base = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sgc_input_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.mu_head = nn.Linear(n_hidden, n_topics)
        self.var_head = nn.Linear(n_hidden, n_topics)

        # STAMP-style: norm_topic BN on mu (always applied, even with 1 batch)
        # With frozen affine weights (weight.requires_grad = False)
        n_bn = max(n_batches, 1)
        self.norm_topic = nn.ModuleList([
            nn.BatchNorm1d(n_topics) for _ in range(n_bn)
        ])
        for norm in self.norm_topic:
            norm.weight.requires_grad = False

        # Xavier initialization (STAMP-style)
        nn.init.xavier_normal_(self.base[1].weight)
        nn.init.xavier_normal_(self.mu_head.weight)
        nn.init.xavier_normal_(self.var_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.var_head.bias)

        # Buffer for precomputed SGC features (full dataset)
        self.register_buffer("sgc_x_full", torch.empty(0, sgc_input_dim))
        self._sgc_initialized = False

    def set_sgc_data(self, sgc_x: torch.Tensor):
        """Store precomputed SGC features for the full dataset."""
        self.sgc_x_full = sgc_x
        self._sgc_initialized = True
        logger.info(f"SGCEncoder initialized: {sgc_x.shape[0]} cells, {sgc_x.shape[1]} features")

    def forward(
        self,
        x: torch.Tensor,  # ignored — we use sgc_x_full
        edge_index: torch.Tensor,  # ignored — spatial info is in SGC features
        batch_indices: torch.Tensor | None = None,
        cat_list: list[torch.Tensor] | None = None,
    ):
        """
        Forward pass using precomputed SGC features.

        Parameters
        ----------
        x : ignored (kept for API compatibility with GCNEncoder)
        edge_index : ignored
        batch_indices : indices into sgc_x_full for the current minibatch
        cat_list : categorical covariates (used for per-batch BatchNorm)

        Returns
        -------
        dist.Normal, None
        """
        if not self._sgc_initialized:
            raise ValueError("SGC features not initialized. Call set_sgc_data() first.")
        if batch_indices is None:
            raise ValueError("batch_indices required for SGCEncoder.")

        # Subset precomputed SGC features to minibatch
        h = self.sgc_x_full[batch_indices]  # (B, sgc_input_dim)

        # Base network
        h = self.base(h)

        # Output heads
        mu = self.mu_head(h)
        raw_var = self.var_head(h)

        # STAMP-style: norm_topic BN on mu (always applied)
        if self.n_batches > 1 and cat_list is not None and len(cat_list) > 0:
            batch_codes = cat_list[0].squeeze(-1).long()
            mu_out = torch.zeros_like(mu)
            for b in range(self.n_batches):
                mask = batch_codes == b
                if mask.sum() > 1:
                    mu_out[mask] = self.norm_topic[b](mu[mask])
                elif mask.sum() == 1:
                    mu_out[mask] = mu[mask]
            mu = mu_out
        else:
            # Single batch: still apply BN (STAMP always does this)
            if mu.shape[0] > 1:
                mu = self.norm_topic[0](mu)

        var = F.softplus(raw_var) + self.var_eps

        return dist.Normal(mu, var.sqrt()), None


class CategoricalBoW(dist.Multinomial):
    """Multinomial that tolerates −inf logits for zero counts (identical to scvi implementation)."""

    def log_prob(self, value):  # type: ignore[override]
        if self._validate_args:
            self._validate_sample(value)
        logits, value = dist.util.broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        logits[(value == 0) & (logits == -math.inf)] = 0
        return (logits * value).sum(-1)


# --------------------------------------------------------------------------------------------------
# Generative model with modality plate
# --------------------------------------------------------------------------------------------------


class MultimodalLDAPyroModel(PyroModule):
    """Generative process

    For k in {1..K}, m in {1..M}::
        ϕₖ,ₘ  ~ LogisticNormal(α)          # topic by feature for modality m
    For cell n::
        θₙ ~ LogisticNormal(β)
        For modality m::
            xₙ,ₘ ~   Multinomial(Nₙ, θₙ·ϕₘ)    if likelihood[m] == "multinomial"
                    GammaPoisson(μ, rₘ)        if likelihood[m] == "gamma_poisson"
    """

    def __init__(
        self,
        n_inputs_modalities: list[int],
        likelihoods: list[str],
        n_topics: int,
        cell_topic_prior: torch.Tensor,
        topic_feature_priors: list[torch.Tensor],  # one tensor per modality (len = M)
        dispersion_rna: float = 1.,  # global NB dispersion for Gamma-Poisson modalities
        learnable_dispersion: bool = False,  # whether to learn dispersion (STAMP-like)
        global_dispersion: bool = True,  # if learnable: one global vs per-gene dispersion
        topic_feature_prior_type: str = "logistic_normal",
        use_feature_background: bool = True,
        learnable_bg: bool = True,
        likelihood_weight_mode: str = "none",
        likelihood_weight_ref: str = "mean",
        init_bg_mean: list[torch.Tensor | None] | None = None,
        # Covariate parameters for decoder-side batch correction
        n_cats_per_cov: list[int] | None = None,
        n_continuous_cov: int = 0,
    ) -> None:
        super().__init__("multimodal_lda")

        assert len(n_inputs_modalities) == len(likelihoods) == len(topic_feature_priors)
        self.n_modalities = len(n_inputs_modalities)
        self.n_topics = n_topics
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.dispersion_rna = dispersion_rna  # used for Gamma-Poisson / NB (when not learnable)
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.learnable_bg = learnable_bg
        self.likelihood_weight_mode = likelihood_weight_mode
        self.likelihood_weight_ref = likelihood_weight_ref

        valid_weight_modes = {"none", "inverse_features", "sqrt_inverse_features"}
        if likelihood_weight_mode not in valid_weight_modes:
            raise ValueError(
                f"likelihood_weight_mode must be one of {valid_weight_modes}, "
                f"got '{likelihood_weight_mode}'"
            )

        valid_weight_refs = {"mean", "median", "max"}
        if likelihood_weight_ref not in valid_weight_refs:
            raise ValueError(
                f"likelihood_weight_ref must be one of {valid_weight_refs}, "
                f"got '{likelihood_weight_ref}'"
            )

        feature_counts = torch.as_tensor(n_inputs_modalities, dtype=torch.float32)
        if likelihood_weight_mode == "none":
            weights = torch.ones_like(feature_counts)
        else:
            if likelihood_weight_ref == "mean":
                ref_value = feature_counts.mean()
            elif likelihood_weight_ref == "median":
                ref_value = feature_counts.median()
            else:
                ref_value = feature_counts.max()

            weights = ref_value / feature_counts
            if likelihood_weight_mode == "sqrt_inverse_features":
                weights = torch.sqrt(weights)
        self.register_buffer("likelihood_weights", weights)

        # Normalise/expand cell_topic_prior to length-K tensor
        cell_topic_prior = torch.as_tensor(cell_topic_prior)
        if cell_topic_prior.ndim == 0:  # scalar -> repeat for each topic
            cell_topic_prior = cell_topic_prior.expand(n_topics)
        elif cell_topic_prior.numel() == 1 and cell_topic_prior.ndim == 1:
            cell_topic_prior = cell_topic_prior.repeat(n_topics)
        elif cell_topic_prior.numel() != n_topics:
            raise ValueError(f"cell_topic_prior must have length {n_topics} (got {cell_topic_prior.numel()})")

        # Cell-topic prior always uses Logistic-Normal
        cell_mu, cell_sigma = logistic_normal_approximation(cell_topic_prior)
        self.register_buffer("cell_mu", cell_mu)
        self.register_buffer("cell_sigma", cell_sigma)

        # Covariate handling for decoder-side batch correction (STAMP-style)
        # Batch effects are modelled as latent offsets on topic-gene logits:
        #   batch_tau   ~ Beta(0.5, 0.5)        per-topic gate       (K, 1)
        #   batch_delta ~ StudentT(10, 0, 0.01)  per-batch, per-gene  (n_batches, F_m)
        # Applied as: phi_b = softmax(log_phi + batch_tau * batch_delta[b])
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        # Batch correction only uses categorical covariates; continuous covariates
        # are encoder-only (concatenated to input) and don't need batch correction.
        has_cat_covs = n_cats_per_cov is not None and len(n_cats_per_cov) > 0
        self.use_covariates = has_cat_covs or n_continuous_cov > 0

        if has_cat_covs:
            if n_cats_per_cov is not None and len(n_cats_per_cov) != 1:
                raise ValueError(
                    f"Exactly one categorical covariate (batch key) is required, "
                    f"got {len(n_cats_per_cov)}. STAMP-style batch correction supports "
                    f"a single batch key only."
                )
            self.n_batches = n_cats_per_cov[0]

        # Topic-feature priors: Logistic-Normal or Horseshoe
        if topic_feature_prior_type == "logistic_normal":
            # Pre-compute Logistic-Normal approximations for priors
            self.topic_prior_mus = torch.nn.ParameterList()
            self.topic_prior_sigmas = torch.nn.ParameterList()
            for t_prior in topic_feature_priors:
                t_prior = torch.as_tensor(t_prior)
                if t_prior.ndim == 0:
                    t_prior = t_prior.expand(1)  # fallback; should be length = n_features_m
                mu_m, sig_m = logistic_normal_approximation(t_prior)
                self.topic_prior_mus.append(torch.nn.Parameter(mu_m, requires_grad=False))
                self.topic_prior_sigmas.append(torch.nn.Parameter(sig_m, requires_grad=False))

        elif topic_feature_prior_type == "horseshoe":
            # No pre-computed topic-feature priors for horseshoe
            # (sampling happens dynamically in forward())
            self.topic_prior_mus = None
            self.topic_prior_sigmas = None

        else:
            raise ValueError(f"Unknown topic_feature_prior_type: {topic_feature_prior_type}")

        # Feature background initialization (like in STAMP)
        # Register as buffers (not trainable, just for initialization)
        if use_feature_background and init_bg_mean is not None:
            for m, bg_mean_m in enumerate(init_bg_mean):
                if bg_mean_m is not None:
                    self.register_buffer(f"init_bg_mean_{m}", bg_mean_m)
                else:
                    self.register_buffer(f"init_bg_mean_{m}", torch.zeros(1))  # Placeholder
        else:
            # No background
            for m in range(self.n_modalities):
                self.register_buffer(f"init_bg_mean_{m}", torch.zeros(1))

        # Populated by training plan for full-batch ELBO scaling
        self.n_obs = None
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # for device

    # ---------- data-loader helper ----------
    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]  # concatenated features
        libraries = []
        cursor = 0
        for n_f in tensor_dict["n_inputs_modalities"]:  # injected by wrapper
            libraries.append(x[:, cursor : cursor + n_f].sum(dim=1))
            cursor += n_f
        return (x, torch.stack(libraries, dim=1)), {}

    # ---------- forward ----------
    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,  # (B, ΣF)
        libraries: torch.Tensor,  # (B, M) – per-modality lib sizes
        n_obs: int | None = None,
        kl_weight: float = 1.0,
        batch_indices: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,  # (B, n_cat_covs)
        cont_covs: torch.Tensor | None = None,  # (B, n_continuous_cov)
        encoder_extra: torch.Tensor | None = None,  # passed to guide, ignored by model
    ):
        # ----- topic-feature distributions (per modality) -----
        topic_feature_dists = []  # will store φₖ,ₘ tensors

        if self.topic_feature_prior_type == "logistic_normal":
            # First, sample backgrounds outside of topics plate (per-feature, not per-topic)
            # Bernoulli and multinomial likelihoods do not use feature backgrounds.
            bg_samples = []
            for m in range(self.n_modalities):
                if self.use_feature_background and self.likelihoods[m] in {"gamma_poisson", "nb"}:
                    init_bg = getattr(self, f"init_bg_mean_{m}")
                    if init_bg.numel() > 1:  # Not a placeholder
                        if self.learnable_bg:
                            with poutine.scale(scale=kl_weight):
                                bg_m = pyro.sample(
                                    f"bg_{m}",
                                    dist.Normal(torch.zeros_like(init_bg), torch.ones_like(init_bg)).to_event(1)
                                )
                            bg_m = bg_m + init_bg  # Add empirical baseline
                        else:
                            bg_m = init_bg  # Fixed background (STAMP-style)
                        bg_samples.append(bg_m)
                    else:
                        bg_samples.append(None)
                else:
                    bg_samples.append(None)

            # Now sample topic-feature distributions
            with pyro.plate("topics", self.n_topics):
                for m in range(self.n_modalities):
                    mu_m = self.topic_prior_mus[m].unsqueeze(0).expand(self.n_topics, -1)
                    sig_m = self.topic_prior_sigmas[m].unsqueeze(0).expand(self.n_topics, -1)

                    with poutine.scale(scale=kl_weight):
                        log_phi = pyro.sample(f"log_topic_feature_dist_{m}", dist.Normal(mu_m, sig_m).to_event(1))

                    # Add background if available
                    if bg_samples[m] is not None:
                        log_phi = log_phi + bg_samples[m].unsqueeze(0)  # (K, F_m) + (1, F_m) -> (K, F_m)

                    topic_feature_dists.append(log_phi)  # (K, Fₘ) -- log-space, softmax applied later

        elif self.topic_feature_prior_type == "horseshoe":
            # Horseshoe prior (again introduced in STAMP)
            for m in range(self.n_modalities):
                F_m = self.n_inputs_modalities[m]

                # 1. Global auxiliary variable (regularization)
                with poutine.scale(scale=kl_weight):
                    caux_m = pyro.sample(
                        f"caux_{m}",
                        dist.InverseGamma(
                            torch.ones(1, device=self._dummy.device) * 0.5,
                            torch.ones(1, device=self._dummy.device) * 0.5
                        )
                    )

                # 2. Per-topic local shrinkage
                with pyro.plate(f"topics_tau_{m}", self.n_topics):
                    with poutine.scale(scale=kl_weight):
                        tau_m = pyro.sample(
                            f"tau_{m}",
                            dist.HalfCauchy(torch.ones(1, device=self._dummy.device))
                        )
                tau_m = tau_m.unsqueeze(-1)  # (K, 1)

                # 3. Per-feature local shrinkage
                with pyro.plate(f"features_delta_{m}", F_m):
                    with poutine.scale(scale=kl_weight):
                        delta_m = pyro.sample(
                            f"delta_{m}",
                            dist.HalfCauchy(torch.ones(1, device=self._dummy.device))
                        )
                # delta_m is (F_m,)

                # 4. Per-topic-feature interaction shrinkage
                # Note: Nested plates add dimensions from left, so outer plate -> inner dimension
                with pyro.plate(f"features_lambda_{m}", F_m):
                    with pyro.plate(f"topics_lambda_{m}", self.n_topics):
                        with poutine.scale(scale=kl_weight):
                            lambda_m = pyro.sample(
                                f"lambda_{m}",
                                dist.HalfCauchy(torch.ones(1, device=self._dummy.device))
                            )
                # lambda_m is now (K, F_m) after plate ordering

                # 5. Compute horseshoe shrinkage multiplier
                lambda_tilde_m = horseshoe_shrinkage(caux_m, tau_m, delta_m, lambda_m)  # (K, F_m)

                # 6. Sample standard normal coefficients
                with pyro.plate(f"features_beta_{m}", F_m):
                    with pyro.plate(f"topics_beta_{m}", self.n_topics):
                        with poutine.scale(scale=kl_weight):
                            beta_m = pyro.sample(
                                f"beta_{m}",
                                dist.Normal(torch.zeros(1, device=self._dummy.device),
                                           torch.ones(1, device=self._dummy.device))
                            )
                # beta_m is now (K, F_m) after plate ordering

                # 7. Apply shrinkage and convert to log-probabilities
                beta_shrunk_m = beta_m * lambda_tilde_m  # (K, F_m)

                # Feature background (scTM-style) - only for count-based likelihoods
                if self.use_feature_background and self.likelihoods[m] in {"gamma_poisson", "nb"}:
                    init_bg = getattr(self, f"init_bg_mean_{m}")
                    if init_bg.numel() > 1:  # Not a placeholder
                        if self.learnable_bg:
                            with poutine.scale(scale=kl_weight):
                                bg_m = pyro.sample(
                                    f"bg_{m}",
                                    dist.Normal(torch.zeros_like(init_bg), torch.ones_like(init_bg)).to_event(1)
                                )
                            bg_m = bg_m + init_bg  # Add empirical baseline
                        else:
                            # Fixed background (STAMP-style): no sampling, no KL cost
                            bg_m = init_bg
                        beta_shrunk_m = beta_shrunk_m + bg_m.unsqueeze(0)  # (K, F_m) + (1, F_m) -> (K, F_m)

                # Register as named sample for guide to match
                pyro.deterministic(f"log_topic_feature_dist_{m}", beta_shrunk_m)

                # 8. Store in log-space (softmax applied later)
                topic_feature_dists.append(beta_shrunk_m)  # (K, F_m) -- log-space

        # ----- dispersion sampling (for learnable dispersion) -----
        dispersion_samples = {}
        if self.learnable_dispersion:
            for m, (F_m, L_m) in enumerate(zip(self.n_inputs_modalities, self.likelihoods, strict=False)):
                if L_m in {"gamma_poisson", "nb"}:
                    if self.global_dispersion:
                        # Single global dispersion for this modality
                        with poutine.scale(scale=kl_weight):
                            disp = pyro.sample(
                                f"disp_{m}",
                                dist.HalfCauchy(torch.ones(1, device=self._dummy.device))
                            )
                        dispersion_samples[m] = disp
                    else:
                        # Per-gene dispersion (STAMP-like)
                        with pyro.plate(f"genes_disp_{m}", F_m):
                            with poutine.scale(scale=kl_weight):
                                disp = pyro.sample(
                                    f"disp_{m}",
                                    dist.HalfCauchy(torch.ones(1, device=self._dummy.device))
                                )
                        dispersion_samples[m] = disp  # shape: (F_m,)

        # ----- STAMP-style batch effect parameters (per modality) -----
        batch_effects = []  # list of (batch_tau_m, batch_delta_m) tuples
        if self.use_covariates:
            for m in range(self.n_modalities):
                F_m = self.n_inputs_modalities[m]

                # Per-topic gate: Beta(0.5, 0.5), shape (K, 1)
                with pyro.plate(f"topics_batch_tau_{m}", self.n_topics):
                    with poutine.scale(scale=kl_weight):
                        batch_tau_m = pyro.sample(
                            f"batch_tau_{m}",
                            dist.Beta(
                                torch.tensor(0.5, device=self._dummy.device),
                                torch.tensor(0.5, device=self._dummy.device),
                            ),
                        )
                batch_tau_m = batch_tau_m.unsqueeze(-1)  # (K, 1)

                # Per-batch, per-gene offset: StudentT(10, 0, 0.01), shape (n_batches, F_m)
                # Plate nesting: genes outer (dim=-1), batches inner (dim=-2)
                # -> sample shape (n_batches, F_m) with n_batches at dim=-2, F_m at dim=-1
                with pyro.plate(f"genes_batch_delta_{m}", F_m):
                    with pyro.plate(f"batches_batch_delta_{m}", self.n_batches):
                        with poutine.scale(scale=kl_weight):
                            batch_delta_m = pyro.sample(
                                f"batch_delta_{m}",
                                dist.StudentT(
                                    torch.tensor(10.0, device=self._dummy.device),
                                    torch.zeros(1, device=self._dummy.device),
                                    torch.tensor(0.01, device=self._dummy.device),
                                ),
                            )
                # batch_delta_m shape: (n_batches, F_m)

                batch_effects.append((batch_tau_m, batch_delta_m))

        # ----- Gaussian sigma (per-feature variance, shared across cells) -----
        gaussian_sigma_samples = {}
        for m, (F_m, L_m) in enumerate(zip(self.n_inputs_modalities, self.likelihoods, strict=False)):
            if L_m == "gaussian":
                with poutine.scale(scale=kl_weight):
                    gaussian_sigma_samples[m] = pyro.sample(
                        f"gaussian_sigma_{m}",
                        dist.HalfCauchy(torch.ones(F_m, device=self._dummy.device)).to_event(1)
                    )

        # ----- cells plate -----
        with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=x.shape[0]):
            # shared θₙ
            with poutine.scale(scale=kl_weight):
                log_theta = pyro.sample("log_cell_topic_dist", dist.Normal(self.cell_mu, self.cell_sigma).to_event(1))

            theta = F.softmax(log_theta, dim=-1)

            # likelihood per modality
            cursor = 0
            for m, (F_m, L_m) in enumerate(zip(self.n_inputs_modalities, self.likelihoods, strict=False)):
                x_m = x[:, cursor : cursor + F_m]
                lib_m = torch.clamp(libraries[:, m], min=0.0)
                log_phi_m = topic_feature_dists[m]  # (K, F_m) -- log-space

                if L_m == "gaussian":
                    # Gaussian likelihood: topic means are unconstrained (NO softmax)
                    # NO library size scaling
                    if self.use_covariates and batch_effects:
                        batch_tau_m, batch_delta_m = batch_effects[m]
                        if cat_covs is None:
                            raise ValueError(
                                "Covariates were enabled but `cat_covs` is missing. "
                                "Ensure CAT_COVS_KEY is provided to the model/guide call."
                            )
                        if cat_covs.dim() == 1:
                            cat_covs = cat_covs.unsqueeze(1)
                        batch_idx = cat_covs[:, 0].long()
                        mean_m = torch.zeros_like(x_m)
                        for b in range(self.n_batches):
                            mask_b = (batch_idx == b)
                            if mask_b.any():
                                offset = batch_tau_m * batch_delta_m[b]
                                mu_b = log_phi_m + offset  # additive offset, no softmax
                                mean_m[mask_b] = theta[mask_b] @ mu_b
                    else:
                        mean_m = theta @ log_phi_m  # (B, F_m)

                    # Per-feature variance (sampled outside cells plate)
                    sigma_m = gaussian_sigma_samples[m]

                    likelihood_scale = self.likelihood_weights[m]
                    with poutine.scale(scale=likelihood_scale):
                        pyro.sample(
                            f"feature_counts_{m}",
                            dist.Normal(mean_m, sigma_m).to_event(1),
                            obs=x_m,
                        )
                else:
                    # Count-based likelihoods: softmax on phi, library size scaling
                    # Apply STAMP-style batch correction on topic-gene logits
                    if self.use_covariates and batch_effects:
                        batch_tau_m, batch_delta_m = batch_effects[m]
                        if cat_covs is None:
                            raise ValueError(
                                "Covariates were enabled but `cat_covs` is missing. "
                                "Ensure CAT_COVS_KEY is provided to the model/guide call."
                            )
                        if cat_covs.dim() == 1:
                            cat_covs = cat_covs.unsqueeze(1)
                        batch_idx = cat_covs[:, 0].long()  # (B,)
                        rate_m = torch.zeros_like(x_m)
                        for b in range(self.n_batches):
                            mask_b = (batch_idx == b)
                            if mask_b.any():
                                offset = batch_tau_m * batch_delta_m[b]  # (K, 1) * (F_m,) -> (K, F_m)
                                phi_b = F.softmax(log_phi_m + offset, dim=-1)  # (K, F_m)
                                rate_m[mask_b] = theta[mask_b] @ phi_b  # (n_b, F_m)
                    else:
                        rate_m = theta @ F.softmax(log_phi_m, dim=-1)  # (B, F_m)
                    likelihood_scale = self.likelihood_weights[m]
                    with poutine.scale(scale=likelihood_scale):
                        if L_m == "multinomial":
                            x_m_obs = torch.clamp(x_m, min=0.0).round()
                            N_max = int(x_m_obs.sum(dim=1).max().item()) if x_m_obs.numel() > 0 else 0
                            pyro.sample(
                                f"feature_counts_{m}",
                                CategoricalBoW(N_max, rate_m),
                                obs=x_m_obs,
                            )
                        elif L_m in {"gamma_poisson", "nb"}:
                            # Observations must be non-negative integers; guard against preprocessed inputs
                            x_m_obs = torch.clamp(x_m, min=0.0).round()
                            # mean scaled by lib
                            mu = torch.clamp(rate_m * lib_m.unsqueeze(-1), min=1e-8)

                            if self.learnable_dispersion and m in dispersion_samples:
                                # Use sampled dispersion (STAMP-like parameterization)
                                disp = dispersion_samples[m]
                                inv_disp = 1.0 / (disp ** 2 + 1e-8)
                                # GammaPoisson(concentration, rate) where mean = concentration/rate
                                pyro.sample(
                                    f"feature_counts_{m}",
                                    dist.GammaPoisson(
                                        concentration=inv_disp,
                                        rate=inv_disp / mu
                                    ).to_event(1),
                                    obs=x_m_obs,
                                )
                            else:
                                # Fixed dispersion (original behavior)
                                r = torch.tensor(self.dispersion_rna, device=mu.device)
                                pyro.sample(
                                    f"feature_counts_{m}",
                                    dist.NegativeBinomial(total_count=r, probs=mu / (mu + r)).to_event(1),
                                    obs=x_m_obs,
                                )
                        elif L_m == "bernoulli":
                            x_m_obs = torch.clamp(x_m, min=0.0, max=1.0).round()
                            # Library size scaling (depth normalization)
                            lib_ratio = lib_m / lib_m.mean()  # (B,) - relative depth
                            p_m = rate_m * lib_ratio.unsqueeze(-1)  # (B, F_m) - scale by depth
                            p_m = torch.clamp(p_m, max=1.0)  # Ensure valid probability
                            # Sample Bernoulli observations
                            pyro.sample(
                                f"feature_counts_{m}",
                                dist.Bernoulli(probs=p_m).to_event(1),
                                obs=x_m_obs,
                            )
                        else:
                            raise ValueError(f"Unknown likelihood {L_m}")
                cursor += F_m


# --------------------------------------------------------------------------------------------------
# Guide with Mixture of Experts
# --------------------------------------------------------------------------------------------------


class MultimodalLDAPyroGuide(PyroModule):
    def __init__(
        self,
        n_inputs_modalities: list[int],
        n_topics: int,
        n_hidden: int,
        gcn_n_layers: int = 1,
        gcn_hidden_dims: list[int] | None = None,
        gcn_n_pre_layers: int = 0,
        gcn_conv_type: str='GATv2Conv',
        gcn_alpha_init: float = 0.7,
        gcn_use_learned_alpha: bool = True,
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
        topic_feature_prior_type: str = "logistic_normal",
        use_feature_background: bool = True,
        learnable_bg: bool = True,
        likelihoods: list[str] | None = None,
        learnable_dispersion: bool = False,  # whether to learn dispersion (STAMP-like)
        global_dispersion: bool = True,  # if learnable: one global vs per-gene dispersion
        normalize_encoder_inputs: bool = True,
        encoder_scale_factor: float = 1e4,
        entropy_weight: float = 0.01,
        topic_variance_weight: float = 1.0,
        # Covariate parameters (encoder side: n_cat_list for scvi Encoder)
        n_cats_per_cov: list[int] | None = None,
        n_continuous_cov: int = 0,
        # Batch effect parameters (decoder side: STAMP-style)
        n_batches: int = 0,
        use_batch_covariates: bool = False,
        n_extra_encoder_features: int = 0,
        # Aggregation mode
        aggregation_type: str = "moe",  # "moe" or "attention"
        att_dim: int = 32,              # attention projection dim (used when aggregation_type="attention")
        # Spatial mode: "gcn" (default, trainable GCN/GAT) or "sgc" (STAMP-style precomputed)
        spatial_mode: str = "gcn",
        sgc_n_layers: int = 1,  # number of SGC hops (only used when spatial_mode="sgc")
        # Neighbor sampling (only used when spatial_mode="gcn")
        gcn_sampling: str = "approx",  # "approx" or "exact"
        gcn_fan_out: list[int] | None = None,
        gcn_conv_first: bool = False,
    ) -> None:
        super().__init__("multimodal_lda_guide")
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.n_topics = n_topics
        self.spatial_mode = spatial_mode
        self.use_gcn = spatial
        self.adjacency = adjacency  # keep reference for downstream checks/tests
        self.sgc_n_layers = sgc_n_layers
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.learnable_bg = learnable_bg
        self.n_extra_encoder_features = n_extra_encoder_features
        self.likelihoods = likelihoods if likelihoods is not None else ["gamma_poisson"] * self.n_modalities
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion
        self.normalize_encoder_inputs = normalize_encoder_inputs
        self.encoder_scale_factor = encoder_scale_factor
        self.entropy_weight = entropy_weight
        self._last_entropy = None  # For logging
        self.topic_variance_weight = topic_variance_weight
        self._last_topic_variance = None  # For logging

        # Covariate parameters (for encoder-side batch correction via n_cat_list)
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.use_covariates = (n_cats_per_cov is not None and len(n_cats_per_cov) > 0) or n_continuous_cov > 0

        # STAMP-style batch effect variational parameters (decoder side)
        self.n_batches = n_batches
        self.use_batch_covariates = use_batch_covariates
        if use_batch_covariates and n_batches > 0:
            self.batch_tau_loc = torch.nn.ParameterList()
            self.batch_tau_scale = torch.nn.ParameterList()
            self.batch_delta_loc = torch.nn.ParameterList()
            self.batch_delta_scale = torch.nn.ParameterList()
            for F_m in n_inputs_modalities:
                # batch_tau: Logit-Normal approximation to Beta posterior
                # loc=0 -> sigmoid(0)=0.5 initial mode
                self.batch_tau_loc.append(torch.nn.Parameter(torch.zeros(n_topics)))
                _sp_inv_01 = float(torch.log(torch.exp(torch.tensor(0.1)) - 1))
                self.batch_tau_scale.append(torch.nn.Parameter(torch.full((n_topics,), _sp_inv_01)))
                # batch_delta: Normal approximation to StudentT posterior
                self.batch_delta_loc.append(torch.nn.Parameter(torch.zeros(n_batches, F_m)))
                self.batch_delta_scale.append(torch.nn.Parameter(torch.full((n_batches, F_m), _sp_inv_01)))

        # Regular encoders (always present)
        # scvi Encoder handles categorical covariates via n_cat_list (internal embeddings)
        # Continuous covariates are added to input dimension
        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    n_in + n_continuous_cov + n_extra_encoder_features,
                    n_topics,
                    distribution="ln",
                    return_dist=True,
                    n_hidden=n_hidden,
                    n_cat_list=n_cats_per_cov,  # scvi handles categorical embeddings
                )
                for n_in in n_inputs_modalities
            ]
        )
        
        # Spatial encoders (if spatial)
        if self.use_gcn:
            if adjacency is None:
                raise ValueError("Spatial encoder requested (spatial=True) but no adjacency was provided.")

            if spatial_mode == "sgc":
                # STAMP-style: SGC encoder with precomputed features
                _n_batches_sgc = n_cats_per_cov[0] if (n_cats_per_cov and len(n_cats_per_cov) > 0) else 0
                self.gcn_encoders = torch.nn.ModuleList(
                    [
                        SGCEncoder(
                            n_genes=n_in,
                            n_sgc_layers=sgc_n_layers,
                            n_hidden=n_hidden,
                            n_topics=n_topics,
                            dropout=0.1,
                            n_batches=_n_batches_sgc,
                        )
                        for n_in in n_inputs_modalities
                    ]
                )
                self.multiple_adjacencies = False
                # No edge_index buffers needed — spatial info is in SGC features

            else:
                # Default: trainable GCN/GAT encoders
                self.gcn_encoders = torch.nn.ModuleList(
                    [
                        GCNEncoder(
                            n_in + n_extra_encoder_features,
                            n_topics,
                            n_hidden,
                            gcn_n_layers=gcn_n_layers,
                            gcn_n_pre_layers=gcn_n_pre_layers,
                            conv_type=gcn_conv_type,
                            gcn_hidden_dims=gcn_hidden_dims,
                            alpha_init=gcn_alpha_init,
                            use_learned_alpha=gcn_use_learned_alpha,
                            n_cats_per_cov=None,
                            n_continuous_cov=n_continuous_cov,
                            sampling=gcn_sampling,
                            fan_out=gcn_fan_out,
                            conv_first=gcn_conv_first,
                        )
                        for n_in in n_inputs_modalities
                    ]
                )

                # Convert adjacency to edge_index ONCE and store on CPU.
                # Plain attributes (not buffers) so they are NOT moved to GPU
                # by .to(device) — subgraph extraction in GCNEncoder happens on CPU.
                if isinstance(adjacency, (list, tuple)):
                    for idx, adj in enumerate(adjacency):
                        ei = adjacency_to_edge_index(adj).cpu()
                        setattr(self, f"edge_index_{idx}", ei)
                    self.multiple_adjacencies = True
                else:
                    self.edge_index = adjacency_to_edge_index(adjacency).cpu()
                    self.multiple_adjacencies = False
        else:
            self.gcn_encoders = None
            self.multiple_adjacencies = False

        # per-modality topic-feature posterior params
        if topic_feature_prior_type == "logistic_normal":
            # Existing: per-modality topic-feature posterior params
            self.topic_feature_posterior_mu = torch.nn.ParameterList()
            self.unconstrained_topic_feature_posterior_sigma = torch.nn.ParameterList()
            for F_m in n_inputs_modalities:
                mu_m, sig_m = logistic_normal_approximation(torch.ones(F_m))
                self.topic_feature_posterior_mu.append(torch.nn.Parameter(mu_m.repeat(n_topics, 1)))
                self.unconstrained_topic_feature_posterior_sigma.append(torch.nn.Parameter(sig_m.repeat(n_topics, 1)))

        elif topic_feature_prior_type == "horseshoe":
            # NEW: Horseshoe variational parameters (LogNormal distributions)
            # Following scTM convention: xxx_loc and xxx_scale

            # Per modality: caux (scalar)
            self.caux_loc = torch.nn.ParameterList()
            self.caux_scale = torch.nn.ParameterList()

            # Per modality: tau (K,)
            self.tau_loc = torch.nn.ParameterList()
            self.tau_scale = torch.nn.ParameterList()

            # Per modality: delta (F_m,)
            self.delta_loc = torch.nn.ParameterList()
            self.delta_scale = torch.nn.ParameterList()

            # Per modality: lambda (K, F_m)
            self.lambda_loc = torch.nn.ParameterList()
            self.lambda_scale = torch.nn.ParameterList()

            # Per modality: beta (K, F_m)
            self.beta_loc = torch.nn.ParameterList()
            self.beta_scale = torch.nn.ParameterList()

            # softplus_inv(0.1) — raw param value so that softplus(param) = 0.1
            # This matches STAMP's PyroParam(init=0.1, constraint=positive)
            _sp_inv_01 = float(torch.log(torch.exp(torch.tensor(0.1)) - 1))  # ≈ -2.252

            for F_m in n_inputs_modalities:
                # Initialize caux (STAMP: loc=0.1, scale→0.1 after softplus)
                self.caux_loc.append(torch.nn.Parameter(torch.ones(1) * 0.1))
                self.caux_scale.append(torch.nn.Parameter(torch.full((1,), _sp_inv_01)))

                # Initialize tau (per-topic, STAMP: loc=0, scale→0.1)
                self.tau_loc.append(torch.nn.Parameter(torch.zeros(n_topics)))
                self.tau_scale.append(torch.nn.Parameter(torch.full((n_topics,), _sp_inv_01)))

                # Initialize delta (per-feature, STAMP: loc=0, scale→0.1)
                self.delta_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                self.delta_scale.append(torch.nn.Parameter(torch.full((F_m,), _sp_inv_01)))

                # Initialize lambda (per-topic-feature, STAMP: loc=0, scale→0.1)
                self.lambda_loc.append(torch.nn.Parameter(torch.zeros(n_topics, F_m)))
                self.lambda_scale.append(torch.nn.Parameter(torch.full((n_topics, F_m), _sp_inv_01)))

                # Initialize beta (per-topic-feature, STAMP: loc=0, scale→0.1)
                self.beta_loc.append(torch.nn.Parameter(torch.zeros(n_topics, F_m)))
                self.beta_scale.append(torch.nn.Parameter(torch.full((n_topics, F_m), _sp_inv_01)))

        # Feature background variational parameters (scTM-style)
        # Only for gamma_poisson modalities when use_feature_background=True
        if use_feature_background:
            self.bg_loc = torch.nn.ParameterList()
            self.bg_scale = torch.nn.ParameterList()
            for m, (F_m, likelihood) in enumerate(zip(n_inputs_modalities, self.likelihoods)):
                if likelihood == "gamma_poisson":
                    # Create background parameters for gamma_poisson modalities
                    # STAMP: bg_scale init = 0.1, so raw param = softplus_inv(0.1)
                    _sp_inv_01 = float(torch.log(torch.exp(torch.tensor(0.1)) - 1))
                    self.bg_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                    self.bg_scale.append(torch.nn.Parameter(torch.full((F_m,), _sp_inv_01)))
                else:
                    # Placeholder for multinomial modalities (won't be used)
                    self.bg_loc.append(None)
                    self.bg_scale.append(None)
        else:
            self.bg_loc = None
            self.bg_scale = None

        # Gaussian sigma variational parameters (for Gaussian likelihood modalities)
        # LogNormal posterior: sigma ~ LogNormal(sigma_loc, softplus(sigma_scale))
        has_gaussian = any(l == "gaussian" for l in self.likelihoods)
        if has_gaussian:
            self.gaussian_sigma_loc = torch.nn.ParameterList()
            self.gaussian_sigma_scale = torch.nn.ParameterList()
            for m, (F_m, likelihood) in enumerate(zip(n_inputs_modalities, self.likelihoods)):
                if likelihood == "gaussian":
                    _sp_inv_01 = float(torch.log(torch.exp(torch.tensor(0.1)) - 1))
                    self.gaussian_sigma_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                    self.gaussian_sigma_scale.append(torch.nn.Parameter(torch.full((F_m,), _sp_inv_01)))
                else:
                    self.gaussian_sigma_loc.append(None)
                    self.gaussian_sigma_scale.append(None)
        else:
            self.gaussian_sigma_loc = None
            self.gaussian_sigma_scale = None

        # Dispersion variational parameters (for learnable dispersion, STAMP-like)
        # LogNormal posterior: disp ~ LogNormal(disp_loc, softplus(disp_scale))
        if learnable_dispersion:
            self.disp_loc = torch.nn.ParameterList()
            self.disp_scale = torch.nn.ParameterList()
            for m, (F_m, likelihood) in enumerate(zip(n_inputs_modalities, self.likelihoods)):
                if likelihood in {"gamma_poisson", "nb"}:
                    _sp_inv_01 = float(torch.log(torch.exp(torch.tensor(0.1)) - 1))
                    if global_dispersion:
                        # Single dispersion per modality
                        self.disp_loc.append(torch.nn.Parameter(torch.zeros(1)))
                        self.disp_scale.append(torch.nn.Parameter(torch.full((1,), _sp_inv_01)))
                    else:
                        # Per-gene dispersion (STAMP-like)
                        self.disp_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                        self.disp_scale.append(torch.nn.Parameter(torch.full((F_m,), _sp_inv_01)))
                else:
                    # Placeholder for non-NB modalities
                    self.disp_loc.append(None)
                    self.disp_scale.append(None)
        else:
            self.disp_loc = None
            self.disp_scale = None

        if weight_mode == "equal":
            self.mod_w = None
        elif weight_mode == "universal":
            self.mod_w = torch.nn.Parameter(torch.ones(self.n_modalities))
        elif weight_mode == "cell":
            if max_n_obs is None:
                raise ValueError("Specify `max_n_obs` when weight_mode=='cell'.")
            self.mod_w = torch.nn.Parameter(torch.ones(max_n_obs, self.n_modalities))
        else:
            raise ValueError("weight_mode must be 'equal'|'universal'|'cell'.")

        self.weight_mode = weight_mode
        self.n_obs = None

        # Aggregation type: "moe" (default) or "attention" (SpatialGlue-style)
        if aggregation_type not in ("moe", "attention"):
            raise ValueError("aggregation_type must be 'moe' or 'attention'.")
        self.aggregation_type = aggregation_type
        if aggregation_type == "attention":
            self.attention_aggregator = AttentionAggregator(n_topics, att_dim=att_dim)
        else:
            self.attention_aggregator = None

    @staticmethod
    def _softplus(t: torch.Tensor) -> torch.Tensor:
        return F.softplus(t)

    def _mix_gaussians(
        self,
        mus: torch.Tensor,
        vars_: torch.Tensor,
        masks: torch.Tensor,
        cell_idx: torch.Tensor,
        precomputed_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mix (*M,B,K*) Gaussians using weights w_{m,c}.

        Parameters
        ----------
        mus/vars_ : (M , B , K)
        masks     : (M , B)  (0 = modality absent)
        precomputed_w : (M , B , 1), optional
            Pre-normalised weights (e.g. from AttentionAggregator).  When
            supplied, mod_w / weight_mode are ignored.
        """
        if precomputed_w is not None:
            w = precomputed_w  # (M, B, 1) — already masked and normalised
        elif self.mod_w is None:  # equal
            w = torch.ones_like(masks)
            w = masked_softmax(w, masks, dim=0).unsqueeze(-1)  # (M,B,1)
        elif self.mod_w.dim() == 1:  # universal
            w = self.mod_w.view(-1, 1).expand_as(masks)
            w = masked_softmax(w, masks, dim=0).unsqueeze(-1)
        else:  # cell
            w = self.mod_w[cell_idx, :]  # (B , M) -> transpose
            w = w.T  # (M , B)
            w = masked_softmax(w, masks, dim=0).unsqueeze(-1)
        mu = (w * mus).sum(0)
        mu = clamp_symmetric(mu)
        # Law of total variance: Var[X] = E[Var[X|M]] + Var[E[X|M]]
        # E[Var[X|M]] - expected variance within each encoder
        expected_var = (w * vars_).sum(0)
        # Var[E[X|M]] - variance of means across encoders
        var_of_means = (w * (mus - mu.unsqueeze(0))**2).sum(0)
        var = expected_var + var_of_means
        var = clamp_positive(var)
        return mu, var

    def _get_edge_index(self, modality_idx: int) -> torch.Tensor:
        """Get edge_index for a specific modality."""
        if not self.use_gcn:
            raise RuntimeError("Cannot get edge_index when spatial=False")
        
        if self.multiple_adjacencies:
            return getattr(self, f"edge_index_{modality_idx}")
        else:
            return self.edge_index

    # property for sigmas
    @property
    def topic_feature_posterior_sigma(self) -> list[torch.Tensor]:
        if self.topic_feature_prior_type == "logistic_normal":
            return [self._softplus(u) for u in self.unconstrained_topic_feature_posterior_sigma]
        elif self.topic_feature_prior_type == "horseshoe":
            # For horseshoe, return beta posterior scales
            return [self._softplus(s) for s in self.beta_scale]
        else:
            raise ValueError(f"Unknown prior type: {self.topic_feature_prior_type}")

    # ═══════════════════════════════════════════════════════════════════════
    # NEW METHOD: Initialize full graph data for semi-supervised learning
    # ═══════════════════════════════════════════════════════════════════════
    def set_full_graph_data(
        self,
        x_full_modalities: list[torch.Tensor] | torch.Tensor,
        adjacency_scipy: sp.spmatrix | None = None,
    ):
        """
        Initialize spatial encoders with full graph data.

        For GCN mode: stores normalized features and edge indices.
        For SGC mode: precomputes SGC features and stores them in the encoder.

        Parameters
        ----------
        x_full_modalities : list[torch.Tensor] or torch.Tensor
            Either:
            - List of full feature matrices, one per modality [n_obs, n_features_m]
            - Single concatenated matrix [n_obs, sum(n_features)] (will be split)
        adjacency_scipy : scipy sparse matrix, optional
            Original scipy adjacency matrix. Required for SGC mode to precompute
            spatial smoothing.
        """
        if not self.use_gcn:
            logger.warning("set_full_graph_data() called but spatial=False. No effect.")
            return

        # Split concatenated features if needed
        if isinstance(x_full_modalities, torch.Tensor):
            x_full_list = torch.split(x_full_modalities, self.n_inputs_modalities, dim=1)
        else:
            x_full_list = x_full_modalities

        if self.spatial_mode == "sgc":
            # STAMP-style: precompute SGC features for each modality
            if adjacency_scipy is None:
                raise ValueError("adjacency_scipy is required for SGC mode.")

            for idx, (sgc_enc, x_full_m) in enumerate(zip(self.gcn_encoders, x_full_list)):
                sgc_x = precompute_sgc(x_full_m, adjacency_scipy, n_layers=self.sgc_n_layers)
                sgc_enc.set_sgc_data(sgc_x)

            logger.info(
                f"SGC encoders initialized with {self.sgc_n_layers}-hop precomputed features "
                f"(input dim: {x_full_list[0].shape[1]} -> {x_full_list[0].shape[1] * (self.sgc_n_layers + 1)})"
            )
        else:
            # GCN mode: apply normalization and store in GCN encoders
            if self.normalize_encoder_inputs:
                libs = [x_full_m.sum(dim=1, keepdim=True) for x_full_m in x_full_list]
                x_full_normalized = []
                for m, (x_full_m, lib_m) in enumerate(zip(x_full_list, libs)):
                    if self.likelihoods[m] == "gaussian":
                        x_full_normalized.append(x_full_m)
                    else:
                        lib_m = torch.clamp(lib_m, min=1.0)
                        median_depth = torch.median(lib_m)
                        x_full_m_norm = torch.log1p(x_full_m / lib_m * median_depth)
                        x_full_normalized.append(x_full_m_norm)
                x_full_list = x_full_normalized
                logger.info("Applied library-size and log-normalization to full graph BEFORE spatial convolution")

            for idx, (gcn_enc, x_full_m) in enumerate(zip(self.gcn_encoders, x_full_list)):
                edge_index = self._get_edge_index(idx)
                gcn_enc.set_full_graph_data(x_full_m, edge_index)

            logger.info(
                f"GCN encoders initialized with full graph "
                f"(normalize_encoder_inputs={self.normalize_encoder_inputs})"
            )

    # ---------- forward ----------
    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,  # (B , ΣF)
        libraries: torch.Tensor,  # unused – kept for signature parity
        n_obs: int | None = None,
        kl_weight: float = 1.0,
        batch_indices: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,  # (B, n_cat_covs)
        cont_covs: torch.Tensor | None = None,  # (B, n_continuous_cov)
        encoder_extra: torch.Tensor | None = None,  # (B, n_extra_encoder_features)
    ):
        B = x.shape[0]

        if batch_indices is None:
            batch_indices = torch.arange(B, device=x.device)

        cell_idx = batch_indices  # for weight mixing (if weight_mode=="cell")

        # ϕₖ,ₘ variational distributions
        if self.topic_feature_prior_type == "logistic_normal":
            # EXISTING: Logistic-Normal posterior
            for m in range(self.n_modalities):
                with pyro.plate(f"topics_{m}", self.n_topics):
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"log_topic_feature_dist_{m}",
                            dist.Normal(
                                self.topic_feature_posterior_mu[m],
                                self.topic_feature_posterior_sigma[m]
                            ).to_event(1),
                        )

                # Feature background variational posterior (scTM-style)
                if self.learnable_bg and self.use_feature_background and self.bg_loc is not None and self.bg_loc[m] is not None:
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"bg_{m}",
                            dist.Normal(self.bg_loc[m], self._softplus(self.bg_scale[m])).to_event(1)
                        )

        elif self.topic_feature_prior_type == "horseshoe":
            # NEW: Horseshoe variational posteriors (LogNormal)
            for m in range(self.n_modalities):
                F_m = self.n_inputs_modalities[m]

                # 1. caux variational posterior
                with poutine.scale(scale=kl_weight):
                    pyro.sample(
                        f"caux_{m}",
                        dist.LogNormal(self.caux_loc[m], self._softplus(self.caux_scale[m]))
                    )

                # 2. tau variational posterior
                with pyro.plate(f"topics_tau_{m}", self.n_topics):
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"tau_{m}",
                            dist.LogNormal(self.tau_loc[m], self._softplus(self.tau_scale[m]))
                        )

                # 3. delta variational posterior
                with pyro.plate(f"features_delta_{m}", F_m):
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"delta_{m}",
                            dist.LogNormal(self.delta_loc[m], self._softplus(self.delta_scale[m]))
                        )

                # 4. lambda variational posterior
                with pyro.plate(f"features_lambda_{m}", F_m):
                    with pyro.plate(f"topics_lambda_{m}", self.n_topics):
                        with poutine.scale(scale=kl_weight):
                            pyro.sample(
                                f"lambda_{m}",
                                dist.LogNormal(
                                    self.lambda_loc[m],
                                    self._softplus(self.lambda_scale[m])
                                )
                            )

                # 5. beta variational posterior
                with pyro.plate(f"features_beta_{m}", F_m):
                    with pyro.plate(f"topics_beta_{m}", self.n_topics):
                        with poutine.scale(scale=kl_weight):
                            pyro.sample(
                                f"beta_{m}",
                                dist.Normal(
                                    self.beta_loc[m],
                                    self._softplus(self.beta_scale[m])
                                )
                            )

                # Feature background variational posterior (scTM-style)
                if self.learnable_bg and self.use_feature_background and self.bg_loc is not None and self.bg_loc[m] is not None:
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"bg_{m}",
                            dist.Normal(self.bg_loc[m], self._softplus(self.bg_scale[m])).to_event(1)
                        )

        # STAMP-style batch effect variational posteriors
        if self.use_batch_covariates and self.n_batches > 0:
            for m in range(self.n_modalities):
                F_m = self.n_inputs_modalities[m]

                # batch_tau guide: Logit-Normal (Normal + SigmoidTransform -> (0,1))
                with pyro.plate(f"topics_batch_tau_{m}", self.n_topics):
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"batch_tau_{m}",
                            dist.TransformedDistribution(
                                dist.Normal(
                                    self.batch_tau_loc[m],
                                    self._softplus(self.batch_tau_scale[m]),
                                ),
                                dist.transforms.SigmoidTransform(),
                            ),
                        )

                # batch_delta guide: Normal
                # Plate nesting must match model: genes outer (dim=-1), batches inner (dim=-2)
                with pyro.plate(f"genes_batch_delta_{m}", F_m):
                    with pyro.plate(f"batches_batch_delta_{m}", self.n_batches):
                        with poutine.scale(scale=kl_weight):
                            pyro.sample(
                                f"batch_delta_{m}",
                                dist.Normal(
                                    self.batch_delta_loc[m],
                                    self._softplus(self.batch_delta_scale[m]),
                                ),
                            )

        # Dispersion variational posterior (for learnable dispersion)
        if self.learnable_dispersion and self.disp_loc is not None:
            for m in range(self.n_modalities):
                if self.disp_loc[m] is not None:
                    F_m = self.n_inputs_modalities[m]
                    if self.global_dispersion:
                        # Single dispersion per modality
                        with poutine.scale(scale=kl_weight):
                            pyro.sample(
                                f"disp_{m}",
                                dist.LogNormal(
                                    self.disp_loc[m],
                                    self._softplus(self.disp_scale[m])
                                )
                            )
                    else:
                        # Per-gene dispersion
                        with pyro.plate(f"genes_disp_{m}", F_m):
                            with poutine.scale(scale=kl_weight):
                                pyro.sample(
                                    f"disp_{m}",
                                    dist.LogNormal(
                                        self.disp_loc[m],
                                        self._softplus(self.disp_scale[m])
                                    )
                                )

        # Gaussian sigma variational posterior (for Gaussian likelihood modalities)
        if self.gaussian_sigma_loc is not None:
            for m in range(self.n_modalities):
                if self.gaussian_sigma_loc[m] is not None:
                    F_m = self.n_inputs_modalities[m]
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"gaussian_sigma_{m}",
                            dist.LogNormal(
                                self.gaussian_sigma_loc[m],
                                self._softplus(self.gaussian_sigma_scale[m])
                            ).to_event(1)
                        )

        # θₙ variational
        xs = torch.split(x, self.n_inputs_modalities, dim=1)  # (B,Fₘ)

        # Apply normalization + log transform if requested
        if self.normalize_encoder_inputs:
            # Compute library sizes per modality
            libs = [x_m.sum(dim=1, keepdim=True) for x_m in xs]  # List of (B, 1)
            # Normalize to median depth per modality
            xs_normalized = []
            for m, (x_m, lib_m) in enumerate(zip(xs, libs)):
                if self.likelihoods[m] == "gaussian":
                    # Gaussian modalities: pass through (data is already continuous)
                    xs_normalized.append(x_m)
                else:
                    lib_m = torch.clamp(lib_m, min=1.0)  # Avoid division by zero
                    # Use median depth of this modality as scale factor
                    median_depth = torch.median(lib_m)
                    x_m_norm = torch.log1p(x_m / lib_m * median_depth)
                    xs_normalized.append(x_m_norm)
            xs = xs_normalized

        # Prepare covariate inputs for encoders
        # scvi Encoder expects categorical covariates as *cat_list (one tensor per covariate)
        # Each tensor should be 2D with shape (batch_size, 1)
        # and continuous covariates concatenated to input x
        cat_list = []
        if self.n_cats_per_cov is not None and len(self.n_cats_per_cov) > 0:
            if cat_covs is not None:
                # Split cat_covs into individual covariate tensors, keeping 2D shape
                # Handle both 1D and 2D cat_covs tensors
                if cat_covs.dim() == 1:
                    cat_list = [cat_covs.unsqueeze(1)]
                else:
                    cat_list = [cat_covs[:, i].unsqueeze(1) for i in range(cat_covs.shape[1])]
            else:
                # Encoder expects categorical covariates but none were provided
                # Create dummy zeros as placeholders
                device = x.device
                cat_list = [torch.zeros((B, 1), dtype=torch.long, device=device) for _ in range(len(self.n_cats_per_cov))]

        # Handle continuous covariates - if encoder expects them but they're not provided, create dummies
        if cont_covs is None and self.n_continuous_cov > 0:
            device = x.device
            cont_covs = torch.zeros((B, self.n_continuous_cov), dtype=torch.float32, device=device)

        mus, vars_, masks = [], [], []
        for idx, (enc, x_m) in enumerate(zip(self.encoders, xs, strict=False)):
            # Concatenate extra encoder features + continuous covariates to input
            parts = [x_m]
            if encoder_extra is not None and self.n_extra_encoder_features > 0:
                parts.append(encoder_extra)
            if cont_covs is not None and self.n_continuous_cov > 0:
                parts.append(cont_covs)
            x_m_with_cov = torch.cat(parts, dim=-1) if len(parts) > 1 else x_m

            if self.use_gcn:
                if self.spatial_mode == "sgc":
                    # SGC encoder uses precomputed features; pass cat_list for per-batch BN
                    q_m, _ = self.gcn_encoders[idx](
                        x_m_with_cov, None, batch_indices=batch_indices,
                        cat_list=cat_list,
                    )
                else:
                    edge_index = self._get_edge_index(idx)
                    q_m, _ = self.gcn_encoders[idx](
                        x_m_with_cov, edge_index, batch_indices=batch_indices,
                        cat_list=None  # Not supported in full-graph mode
                    )
            else:
                # scvi Encoder: forward(x, *cat_list)
                q_m, _ = enc(x_m_with_cov, *cat_list)
            mus.append(clamp_symmetric(q_m.loc))
            vars_.append(clamp_positive(q_m.scale**2))
            # Compute mask from ORIGINAL x (before normalization)
            original_xs = torch.split(x, self.n_inputs_modalities, dim=1)
            masks.append((original_xs[idx].sum(1) > 0).float())
        mus = torch.stack(mus)  # (M,B,K)
        vars_ = torch.stack(vars_)  # (M,B,K)
        masks = torch.stack(masks)  # (M,B)

        if self.aggregation_type == "attention":
            attn_w = self.attention_aggregator(mus, masks)  # (M, B, 1)
            muθ, varθ = self._mix_gaussians(mus, vars_, masks, cell_idx, precomputed_w=attn_w)
        else:
            muθ, varθ = self._mix_gaussians(mus, vars_, masks, cell_idx)

        with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=B):
            # Sample cell-topic distribution (with KL weight scaling)
            with poutine.scale(scale=kl_weight):
                log_theta = pyro.sample("log_cell_topic_dist", dist.Normal(muθ, torch.sqrt(varθ)).to_event(1))

            # Compute per-cell entropy and add bonus (extensive formulation, no KL scaling)
            if self.entropy_weight > 0:
                theta = F.softmax(log_theta, dim=-1)  # (B, K)
                # Per-cell entropy: -Σ_k θ_k * log(θ_k)
                entropy_per_cell = -(theta * torch.log(theta + CLAMP_EPS)).sum(dim=-1)  # (B,)
                # Store mean for logging
                self._last_entropy = entropy_per_cell.mean().detach()
                # Add entropy bonus to ELBO (Pyro will sum over batch and scale by n_obs/B)
                # has_rsample=True because entropy is computed from reparametrized sample (log_theta)
                pyro.factor("entropy_bonus", self.entropy_weight * entropy_per_cell, has_rsample=True)

        # Compute topic variance regularization OUTSIDE pyro.plate (batch-level statistic)
        if self.topic_variance_weight > 0:
            # Compute theta if not already computed
            if self.entropy_weight == 0:
                theta = F.softmax(log_theta, dim=-1)  # (B, K)

            # Compute variance of each topic across cells
            topic_variance = theta.var(dim=0)  # (K,) - variance for each topic
            total_variance = topic_variance.sum()  # scalar

            # Store mean for logging (convert to mean across topics for interpretability)
            self._last_topic_variance = topic_variance.mean().detach()

            # Add variance bonus to ELBO (NOT inside kl_weight scale, NOT per-cell)
            # This is a batch-level statistic, so Pyro will NOT automatically scale it
            # has_rsample=True because variance is computed from reparametrized sample (log_theta)
            pyro.factor("topic_variance_bonus", self.topic_variance_weight * total_variance, has_rsample=True)


# --------------------------------------------------------------------------------------------------
# Wrapper module pairing model & guide
# --------------------------------------------------------------------------------------------------


class MultimodalAmortizedLDAPyroModule(PyroBaseModuleClass):
    def __init__(
        self,
        n_inputs_modalities: list[int],
        likelihoods: list[str],
        n_topics: int,
        n_hidden: int,
        cell_topic_prior: float | Sequence[float] | None = None,
        topic_feature_prior: float | Sequence[float] | None = None,
        topic_feature_prior_type: str = "logistic_normal",
        use_feature_background: bool = True,
        learnable_bg: bool = True,
        init_bg_mean: list[torch.Tensor | None] | None = None,
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
        gcn_n_layers: int = 1,
        gcn_n_pre_layers: int = 0,
        gcn_conv_type: str = 'GATv2Conv',
        gcn_hidden_dims: list[int] | None = None,
        gcn_alpha_init: float = 0.7,
        gcn_use_learned_alpha: bool = True,
        dispersion_rna: float = 1.0,
        learnable_dispersion: bool = False,
        global_dispersion: bool = True,
        likelihood_weight_mode: str = "none",
        likelihood_weight_ref: str = "mean",
        normalize_encoder_inputs: bool = True,
        encoder_scale_factor: float = 1e4,
        entropy_weight: float = 0.01,
        topic_variance_weight: float = 1.0,
        kl_weight: float = 1.0,
        # Covariate parameters
        n_cats_per_cov: list[int] | None = None,
        n_continuous_cov: int = 0,
        encode_covariates: bool = True,
        n_extra_encoder_features: int = 0,
        aggregation_type: str = "moe",
        att_dim: int = 32,
        spatial_mode: str = "gcn",
        sgc_n_layers: int = 1,
        gcn_sampling: str = "approx",
        gcn_fan_out: list[int] | None = None,
        gcn_conv_first: bool = False,
    ):
        super().__init__()
        assert len(n_inputs_modalities) == len(likelihoods)
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.n_topics = n_topics
        self.n_hidden = n_hidden
        self.n_modalities = len(n_inputs_modalities)
        self.weight_mode = weight_mode
        self.spatial = spatial
        self.spatial_mode = spatial_mode
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.dispersion_rna = dispersion_rna
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion
        self.normalize_encoder_inputs = normalize_encoder_inputs
        self.n_extra_encoder_features = n_extra_encoder_features
        self.encoder_scale_factor = encoder_scale_factor
        self.entropy_weight = entropy_weight
        self.topic_variance_weight = topic_variance_weight
        self.kl_weight = kl_weight

        # Covariate parameters
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.encode_covariates = encode_covariates

        if cell_topic_prior is None:
            cell_topic_prior_tensor = torch.full((n_topics,), 1 / n_topics)
        elif isinstance(cell_topic_prior, float):
            cell_topic_prior_tensor = torch.full((n_topics,), cell_topic_prior)
        else:
            cell_topic_prior_tensor = torch.tensor(cell_topic_prior)

        topic_feature_priors = []
        for F_m in n_inputs_modalities:
            if topic_feature_prior is None:
                topic_feature_priors.append(torch.full((F_m,), 1 / n_topics))
            elif isinstance(topic_feature_prior, float):
                topic_feature_priors.append(torch.full((F_m,), topic_feature_prior))
            else:
                raise ValueError("Pass list/None/float for topic_feature_prior, not sequence")

        self._model = MultimodalLDAPyroModel(
            n_inputs_modalities,
            likelihoods,
            n_topics,
            cell_topic_prior_tensor,
            topic_feature_priors,
            dispersion_rna=dispersion_rna,
            learnable_dispersion=learnable_dispersion,
            global_dispersion=global_dispersion,
            topic_feature_prior_type=topic_feature_prior_type,
            use_feature_background=use_feature_background,
            learnable_bg=learnable_bg,
            likelihood_weight_mode=likelihood_weight_mode,
            likelihood_weight_ref=likelihood_weight_ref,
            init_bg_mean=init_bg_mean,
            # Covariate parameters (decoder always uses them)
            n_cats_per_cov=n_cats_per_cov,
            n_continuous_cov=n_continuous_cov,
        )
        # Compute batch effect parameters for guide (decoder-side, independent of encode_covariates)
        use_batch_covariates = (n_cats_per_cov is not None and len(n_cats_per_cov) > 0)
        _n_batches = n_cats_per_cov[0] if use_batch_covariates else 0

        self._guide = MultimodalLDAPyroGuide(
            n_inputs_modalities,
            n_topics,
            n_hidden,
            gcn_n_layers=gcn_n_layers,
            gcn_n_pre_layers=gcn_n_pre_layers,
            gcn_conv_type=gcn_conv_type,
            gcn_hidden_dims=gcn_hidden_dims,
            gcn_alpha_init=gcn_alpha_init,
            gcn_use_learned_alpha=gcn_use_learned_alpha,
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
            spatial=spatial,
            adjacency=adjacency,
            topic_feature_prior_type=topic_feature_prior_type,
            use_feature_background=use_feature_background,
            learnable_bg=learnable_bg,
            likelihoods=likelihoods,
            learnable_dispersion=learnable_dispersion,
            global_dispersion=global_dispersion,
            normalize_encoder_inputs=normalize_encoder_inputs,
            encoder_scale_factor=encoder_scale_factor,
            entropy_weight=entropy_weight,
            topic_variance_weight=topic_variance_weight,
            # Covariate parameters (encoder uses them if encode_covariates=True)
            n_cats_per_cov=n_cats_per_cov if encode_covariates else None,
            n_continuous_cov=n_continuous_cov if encode_covariates else 0,
            # Batch effect parameters (decoder side, always active when covariates present)
            n_batches=_n_batches,
            use_batch_covariates=use_batch_covariates,
            n_extra_encoder_features=n_extra_encoder_features,
            aggregation_type=aggregation_type,
            att_dim=att_dim,
            spatial_mode=spatial_mode,
            sgc_n_layers=sgc_n_layers,
            gcn_sampling=gcn_sampling,
            gcn_fan_out=gcn_fan_out,
            gcn_conv_first=gcn_conv_first,
        )

        # We need this method so scvi training plan can create data-loader args
        def _args_from_batch(tdict):
            tdict["n_inputs_modalities"] = n_inputs_modalities  # inject for library calc
            args, kwargs = self._model._get_fn_args_from_batch(tdict)
            kwargs["kl_weight"] = self.kl_weight
            if REGISTRY_KEYS.INDICES_KEY in tdict:
                batch_indices = tdict[REGISTRY_KEYS.INDICES_KEY]
                if batch_indices.dim() > 1:
                    batch_indices = batch_indices.view(-1)
                kwargs["batch_indices"] = batch_indices
            elif "indices" in tdict:
                kwargs["batch_indices"] = tdict["indices"]
            elif "batch_indices" in tdict:
                kwargs["batch_indices"] = tdict["batch_indices"]

            # Extract covariates from batch dict (scVI convention)
            if REGISTRY_KEYS.CAT_COVS_KEY in tdict:
                kwargs["cat_covs"] = tdict[REGISTRY_KEYS.CAT_COVS_KEY]
            if REGISTRY_KEYS.CONT_COVS_KEY in tdict:
                kwargs["cont_covs"] = tdict[REGISTRY_KEYS.CONT_COVS_KEY]
            if "encoder_extra" in tdict:
                kwargs["encoder_extra"] = tdict["encoder_extra"]

            return args, kwargs

        self._get_fn_args_from_batch = _args_from_batch

    # proxies
    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [
            key for key in incompatible.missing_keys
            if key != "_model.likelihood_weights"
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: "
                f"Missing keys: {missing}, Unexpected keys: {unexpected}"
            )
        return incompatible


    def set_full_graph_data(self, x_full: torch.Tensor, adjacency_scipy=None):
        """
        Initialize spatial encoders with full graph data.

        Parameters
        ----------
        x_full : torch.Tensor
            Full concatenated feature matrix for ALL cells [n_obs, sum(n_features)]
        adjacency_scipy : scipy sparse matrix, optional
            Original scipy adjacency. Required for SGC mode.
        """
        if not self.spatial:
            logger.warning("set_full_graph_data() called but spatial=False. No effect.")
            return

        self._guide.set_full_graph_data(x_full, adjacency_scipy=adjacency_scipy)
        logger.info("Module initialized with full graph data")

    # utilities
    def topic_by_feature(self, n_samples: int = 5_000):
        """
        Compute E[ϕₖ,ₘ] via Monte Carlo sampling from variational posterior.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples (default: 5000)

        Returns
        -------
        dict
            Dictionary mapping modality index to topic-feature distribution tensor (K, F_m)
        """
        out = {}

        if self.guide.topic_feature_prior_type == "logistic_normal":
            for m, (mu, sig) in enumerate(
                zip(self.guide.topic_feature_posterior_mu,
                    self.guide.topic_feature_posterior_sigma,
                    strict=False)
            ):
                samples = dist.Normal(mu.detach().cpu(), sig.detach().cpu()).sample((n_samples,))
                if self.guide.likelihoods[m] == "gaussian":
                    # No softmax — topic means are unconstrained for Gaussian
                    tbf = torch.mean(samples, dim=0)
                else:
                    tbf = torch.mean(F.softmax(samples, dim=2), dim=0)
                out[m] = tbf  # (K, F_m)

        elif self.guide.topic_feature_prior_type == "horseshoe":
            # Reconstruct full horseshoe: beta * lambda_tilde + background
            for m in range(self.guide.n_modalities):
                # Compute posterior means of shrinkage parameters (LogNormal mean = exp(loc + scale²/2))
                caux_loc = self.guide.caux_loc[m].detach().cpu()
                caux_scale = self.guide._softplus(self.guide.caux_scale[m]).detach().cpu()
                caux = dist.LogNormal(caux_loc, caux_scale).mean

                tau_loc = self.guide.tau_loc[m].detach().cpu()
                tau_scale = self.guide._softplus(self.guide.tau_scale[m]).detach().cpu()
                tau = dist.LogNormal(tau_loc, tau_scale).mean.unsqueeze(-1)  # (K, 1)

                delta_loc = self.guide.delta_loc[m].detach().cpu()
                delta_scale = self.guide._softplus(self.guide.delta_scale[m]).detach().cpu()
                delta = dist.LogNormal(delta_loc, delta_scale).mean  # (F_m,)

                lambda_loc = self.guide.lambda_loc[m].detach().cpu()
                lambda_scale = self.guide._softplus(self.guide.lambda_scale[m]).detach().cpu()
                lambda_ = dist.LogNormal(lambda_loc, lambda_scale).mean  # (K, F_m)

                lambda_tilde = horseshoe_shrinkage(caux, tau, delta, lambda_)  # (K, F_m)

                # Sample beta and apply shrinkage
                beta_loc = self.guide.beta_loc[m].detach().cpu()
                beta_scale = self.guide._softplus(self.guide.beta_scale[m]).detach().cpu()
                beta_samples = dist.Normal(beta_loc, beta_scale).sample((n_samples,))  # (n_samples, K, F_m)
                shrunk_samples = beta_samples * lambda_tilde.unsqueeze(0)  # (n_samples, K, F_m)

                # Add background if available
                init_bg = getattr(self.model, f"init_bg_mean_{m}", None)
                if init_bg is not None and init_bg.numel() > 1:
                    if self.guide.learnable_bg and self.guide.bg_loc is not None and self.guide.bg_loc[m] is not None:
                        bg_loc = self.guide.bg_loc[m].detach().cpu()
                        bg = bg_loc + init_bg.cpu()  # (F_m,)
                    else:
                        # Fixed background (STAMP-style): just use init_bg
                        bg = init_bg.cpu()  # (F_m,)
                    shrunk_samples = shrunk_samples + bg.unsqueeze(0).unsqueeze(0)  # (n_samples, K, F_m)

                if self.guide.likelihoods[m] == "gaussian":
                    tbf = torch.mean(shrunk_samples, dim=0)  # (K, F_m)
                else:
                    tbf = torch.mean(F.softmax(shrunk_samples, dim=2), dim=0)  # (K, F_m)
                out[m] = tbf

        return out

    def get_topic_distribution(
        self, x: torch.Tensor, n_samples: int = 5_000,
        encoder_extra: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)
        if encoder_extra is not None:
            encoder_extra = encoder_extra.to(device, non_blocking=True)

        B = x.shape[0]
        xs = torch.split(x, self.model.n_inputs_modalities, dim=1)

        # Apply normalization + log transform if requested (must match training)
        if self.guide.normalize_encoder_inputs:
            libs = [x_m.sum(dim=1, keepdim=True) for x_m in xs]
            xs_normalized = []
            for m, (x_m, lib_m) in enumerate(zip(xs, libs)):
                if self.guide.likelihoods[m] == "gaussian":
                    xs_normalized.append(x_m)
                else:
                    lib_m = torch.clamp(lib_m, min=1.0)
                    median_depth = torch.median(lib_m)
                    x_m_norm = torch.log1p(x_m / lib_m * median_depth)
                    xs_normalized.append(x_m_norm)
            xs = xs_normalized

        # Global cell indices — required for spatial (GCN) models so that
        # k_hop_subgraph can extract the correct neighbourhood.
        if batch_indices is not None:
            batch_indices = batch_indices.to(device)
        else:
            batch_indices = torch.arange(B, device=device)

        # Prepare dummy categorical covariates if encoder expects them
        cat_list = []
        if self.guide.n_cats_per_cov is not None and len(self.guide.n_cats_per_cov) > 0:
            # Create dummy zeros since we don't have actual covariate values during inference
            cat_list = [torch.zeros((B, 1), dtype=torch.long, device=device) for _ in range(len(self.guide.n_cats_per_cov))]

        # Prepare dummy continuous covariates if encoder expects them
        if self.guide.n_continuous_cov > 0:
            cont_covs = torch.zeros((B, self.guide.n_continuous_cov), dtype=torch.float32, device=device)
        else:
            cont_covs = None

        # run encoders
        mus, vars_, masks = [], [], []
        for idx, (enc, x_m) in enumerate(zip(self.guide.encoders, xs, strict=False)):
            # Add extra encoder features + continuous covariates to input
            parts = [x_m]
            if encoder_extra is not None and self.guide.n_extra_encoder_features > 0:
                parts.append(encoder_extra)
            if cont_covs is not None:
                parts.append(cont_covs)
            x_m_with_cov = torch.cat(parts, dim=-1) if len(parts) > 1 else x_m

            if self.guide.use_gcn:
                if self.guide.spatial_mode == "sgc":
                    q_m, _ = self.guide.gcn_encoders[idx](
                        x_m_with_cov, None, batch_indices=batch_indices,
                        cat_list=cat_list,
                    )
                else:
                    edge_index = self.guide._get_edge_index(idx)
                    q_m, _ = self.guide.gcn_encoders[idx](x_m_with_cov, edge_index, batch_indices=batch_indices)
            else:
                q_m, _ = enc(x_m_with_cov, *cat_list)
            mus.append(q_m.loc)
            vars_.append(q_m.scale**2)
            # Compute mask from original x
            original_xs = torch.split(x, self.model.n_inputs_modalities, dim=1)
            masks.append((original_xs[idx].sum(1) > 0).float())
        mus = torch.stack(mus)
        vars_ = torch.stack(vars_)
        masks = torch.stack(masks)
        muθ, varθ = self.guide._mix_gaussians(mus, vars_, masks, torch.arange(B, device=x.device))
        samps = dist.Normal(muθ, torch.sqrt(varθ)).sample((n_samples,))
        return torch.softmax(samps, -1).mean(0)

    @auto_move_data
    @torch.inference_mode()
    def get_elbo(self, x: torch.Tensor, libs: torch.Tensor, n_obs: int, kl_weight: float | None = None):
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)
        if kl_weight is None:
            kl_weight = self.kl_weight
        return Trace_ELBO().loss(self.model, self.guide, x, libs, n_obs=n_obs, kl_weight=kl_weight)

    @auto_move_data
    @torch.inference_mode()
    def get_cell_entropy(self, x: torch.Tensor, libs: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Compute per-cell entropy of the cell-topic distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, total_features)
        libs : torch.Tensor
            Per-modality library sizes (batch_size, n_modalities)
        n_samples : int
            Number of samples to draw for Monte Carlo estimation (default: 100)

        Returns
        -------
        torch.Tensor
            Per-cell entropy values, shape (batch_size,)
            H(θ_n) = -Σ_k θ_n,k * log(θ_n,k) for each cell n
        """
        device = next(self._guide.parameters()).device
        x = x.to(device, non_blocking=True)
        libs = libs.to(device, non_blocking=True)

        # Get cell-topic distribution (averaged over samples)
        theta = self.get_topic_distribution(x, libs, n_samples=n_samples)  # (B, K)

        # Compute entropy per cell
        entropy = -(theta * torch.log(theta + 1e-10)).sum(dim=-1)  # (B,)
        return entropy

    def get_last_entropy(self) -> float | None:
        """
        Get the last computed entropy value from the guide.

        Returns
        -------
        float | None
            Mean entropy from the last forward pass, or None if not available
        """
        if hasattr(self.guide, '_last_entropy') and self.guide._last_entropy is not None:
            return float(self.guide._last_entropy)
        return None

    def get_last_topic_variance(self) -> float | None:
        """
        Get the last computed topic variance value from the guide.

        Returns
        -------
        float | None
            Mean topic variance from the last forward pass, or None if not available
        """
        if hasattr(self.guide, '_last_topic_variance') and self.guide._last_topic_variance is not None:
            return float(self.guide._last_topic_variance)
        return None

    def get_topic_variance(self, x: torch.Tensor, libs: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Compute per-topic variance of topic usage across cells.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, total_features)
        libs : torch.Tensor
            Per-modality library sizes (batch_size, n_modalities)
        n_samples : int
            Number of samples to draw for Monte Carlo estimation (default: 100)

        Returns
        -------
        torch.Tensor
            Per-topic variance values, shape (n_topics,)
            Var(θ_:,k) = variance of topic k usage across all cells
        """
        device = next(self._guide.parameters()).device
        x = x.to(device, non_blocking=True)
        libs = libs.to(device, non_blocking=True)

        # Get cell-topic distribution (averaged over samples)
        theta = self.get_topic_distribution(x, libs, n_samples=n_samples)  # (B, K)

        # Compute variance of each topic across cells
        topic_variance = theta.var(dim=0)  # (K,)
        return topic_variance

    @auto_move_data
    @torch.inference_mode()
    def get_per_modality_log_prob(self, x: torch.Tensor, libs: torch.Tensor, n_obs: int):
        """
        Compute log-probability for each modality separately using poutine.trace.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, total_features)
        libs : torch.Tensor
            Per-modality library sizes (batch_size, n_modalities)
        n_obs : int
            Total number of observations in the dataset

        Returns
        -------
        dict[int, float]
            Dictionary mapping modality index to log-probability
        """
        device = next(self._guide.parameters()).device
        x = x.to(device, non_blocking=True)

        # Trace model and guide
        guide_trace = poutine.trace(self.guide).get_trace(x, libs, n_obs=n_obs)
        model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace(x, libs, n_obs=n_obs)
        model_trace.compute_log_prob()

        # Extract per-modality log-probs
        per_mod_logprob = {}
        for m in range(len(self.n_inputs_modalities)):
            site_name = f"feature_counts_{m}"
            if site_name in model_trace.nodes:
                per_mod_logprob[m] = model_trace.nodes[site_name]["log_prob"].sum().item()

        return per_mod_logprob

    @torch.inference_mode()
    def get_learned_dispersion(self, modality: int = 0, n_samples: int = 1000) -> torch.Tensor:
        """
        Get the learned dispersion parameters via Monte Carlo sampling.

        Parameters
        ----------
        modality : int
            Modality index (default: 0)
        n_samples : int
            Number of Monte Carlo samples

        Returns
        -------
        torch.Tensor
            Mean dispersion values. Shape: (1,) if global_dispersion=True,
            (n_features,) if global_dispersion=False.
            If learnable_dispersion=False, returns fixed dispersion value.
        """
        if not self.learnable_dispersion:
            return torch.tensor([self.model.dispersion_rna])

        if self.guide.disp_loc is None or self.guide.disp_loc[modality] is None:
            return torch.tensor([self.model.dispersion_rna])

        loc = self.guide.disp_loc[modality].detach()
        scale = self.guide._softplus(self.guide.disp_scale[modality]).detach()

        # Sample from LogNormal posterior
        samples = dist.LogNormal(loc, scale).sample((n_samples,))
        return samples.mean(dim=0)
