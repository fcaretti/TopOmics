# multimodal_lda_module.py
"""
The file declares three public objects
-------------------------------------
* `MultimodalLDAPyroModel`   – generative process with modality plate & mixed likelihoods.
* `MultimodalLDAPyroGuide`   – per-modality encoders + combined θₙ posterior + per-modality ϕₖ,ₘ posterior.
* `MultimodalAmortizedLDAPyroModule` – thin wrapper pairing the two above and exposing
  helper utilities (`topic_by_feature`, `get_topic_distribution`, `get_elbo`).

A higher-level `ModelClass` wrapper (*MultimodalAmortizedLDA*) is provided at the bottom so that
users get the usual `train()`/`get_latent_representation()`/etc. API.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import pyro
import pyro.distributions as dist
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

logger = logging.getLogger(__name__)
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

# --------------------------------------------------------------------------------------------------
# Helper utils
# --------------------------------------------------------------------------------------------------


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



'''class GCNEncoder(nn.Module):
    """
    GCN encoder using PyTorch Geometric used with spatial data.
    """
    
    def __init__(
        self, 
        n_in: int, 
        n_topics: int, 
        n_hidden: int,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        conv_type: str = 'GCNConv', # can also be GCNConv
        heads = 4, # used only by GAT
        normalize: bool = True, # used only by GCN
        concat: bool = True, #multi-head strategy for GAT
    ) -> None:
        super().__init__()
        
        # Single graph convolution (spatial aggregation)
        if conv_type == 'GATv2Conv':
            self.conv = GATv2Conv(
                in_channels=n_in,
                out_channels=n_hidden if not concat else n_hidden // heads,
                heads=heads,
                add_self_loops=add_self_loops,
                dropout=dropout,
                concat=concat,
            )
        else:
            self.conv = GCNConv(
                n_in, 
                n_hidden,
                add_self_loops=add_self_loops,
                normalize=normalize, 
                #dropout=dropout,
            )
        
        # Two-layer MLP (feature transformation)
        self.mlp_hidden = nn.Linear(n_hidden, n_hidden)
        self.mlp_out = nn.Linear(n_hidden, 2 * n_topics)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("x_full", torch.empty(0, n_in))
        self.register_buffer("edge_index_full", torch.empty(2, 0, dtype=torch.long))
        self._graph_initialized = False
        
    def set_full_graph_data(self, x_full: torch.Tensor, edge_index_full: torch.Tensor):
        """
        Initialize full graph data for full-graph training.
        
        Must be called after model initialization and before training.
        
        Parameters
        ----------
        x_full : torch.Tensor
            Full gene expression matrix for ALL cells [n_obs, n_features]
        edge_index_full : torch.Tensor
            Full graph edge indices in COO format [2, n_edges]
        """
        self.x_full = clamp_symmetric(x_full)
        self.edge_index_full = edge_index_full
        self._graph_initialized = True
        logger.info(f"GCN graph initialized: {x_full.shape[0]} cells, {edge_index_full.shape[1]} edges")
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_indices: torch.Tensor | None = None):
        """
        Forward pass using the full-graph data when initialized.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (batch_size, n_in) - IGNORED when full-graph data is initialized
        edge_index : torch.Tensor
            Graph connectivity - IGNORED when full-graph data is initialized
        batch_indices : torch.Tensor, optional
            Indices of cells in current batch [batch_size]
            Required when full-graph data is initialized
        
        Returns
        -------
        distribution : pyro.distributions.Normal
            Variational posterior q(z|x, A)
        None
            Placeholder for compatibility
        """

        if self._graph_initialized:
            if batch_indices is None:
                raise ValueError(
                    "batch_indices required for semi-supervised learning. "
                    "This should be provided automatically by the guide."
                )
            
            # STEP 1: Compute on FULL graph
            h_full = self.conv(self.x_full, self.edge_index_full)
            h_full = F.relu(h_full)
            #h_full = self.dropout(h_full)
            
            # MLP layers on full graph
            h_full = self.mlp_hidden(h_full)
            h_full = F.relu(h_full)
            #h_full = self.dropout(h_full)
            h_full = self.mlp_out(h_full)
            
            # STEP 2: Subset to batch
            h = h_full[batch_indices]  # [batch_size, 2 * n_topics]
            
        else:
            raise ValueError(
                "Graph should be precomputer"
            )
        
        # Split into mean and scale
        mu, raw_scale = h.chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + 1e-4
        
        return dist.Normal(mu, scale), None'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
from torch_geometric.nn import GCNConv, GATv2Conv

logger = logging.getLogger(__name__)


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

        prev_dim = n_in
        for hidden_dim in gcn_hidden_dims:
            conv, conv_out_dim = _make_conv(prev_dim, hidden_dim)
            self.convs.append(conv)
            self.conv_bns.append(nn.BatchNorm1d(conv_out_dim))
            self.conv_out_dims.append(conv_out_dim)
            prev_dim = conv_out_dim

        # scvi-style encoder for the self signal
        fc_n_layers = kwargs.pop("n_layers", 1)
        fc_dropout_rate = kwargs.pop("dropout_rate", dropout)
        self.encoder = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_cat_list=None,
            n_layers=fc_n_layers,
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
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1) exclusive for sigmoid transformation.")
        # Store alpha in logit space, apply sigmoid during forward pass
        alpha_logit = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
        if use_learned_alpha:
            self._alpha_logit = nn.Parameter(alpha_logit.clone().detach())
        else:
            self.register_buffer("_alpha_logit", alpha_logit.clone().detach())

        # Full-graph buffers
        self.register_buffer("x_full", torch.empty(0, n_in))
        self.register_buffer("edge_index_full", torch.empty(2, 0, dtype=torch.long))
        self._graph_initialized = False

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
            Alpha value in (0, 1) exclusive.
        """
        value = float(value)
        # Clamp to valid range for logit (avoid inf)
        value = min(max(value, 1e-6), 1.0 - 1e-6)
        logit_value = torch.logit(torch.tensor(value, dtype=self._alpha_logit.dtype, device=self._alpha_logit.device))
        self._alpha_logit.data.copy_(logit_value)

    def set_full_graph_data(self, x_full: torch.Tensor, edge_index_full: torch.Tensor):
        """
        Initialize full graph data for full-graph training.

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

        self.x_full = x_full
        self.edge_index_full = edge_index_full
        self._graph_initialized = True
        logger.info(
            f"GCN graph initialized: {x_full.shape[0]} cells, {edge_index_full.shape[1]} edges"
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
    ):
        """
        Forward pass. Uses full-graph data when initialized.

        Parameters
        ----------
        x : torch.Tensor
            Ignored when full-graph is initialized
        edge_index : torch.Tensor
            Ignored when full-graph is initialized
        batch_indices : torch.Tensor, optional
            Indices of nodes in the current minibatch. Required if full-graph is initialized.

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
                "batch_indices required when using full-graph training. "
                "This should be provided automatically by the guide."
            )

        # ---- STEP 1: compute on FULL graph ----
        x_full = clamp_symmetric(self.x_full)
        ei_full = self.edge_index_full

        # Self path (scvi-style)
        h_self_full = clamp_symmetric(self.encoder(x_full))

        # Neighbor aggregation path (stacked graph convs with BatchNorm)
        h_nei_full = x_full
        for i, (conv, bn) in enumerate(zip(self.convs, self.conv_bns, strict=False)):
            h_nei_full = conv(h_nei_full, ei_full)
            h_nei_full = clamp_symmetric(bn(h_nei_full))
            if i < len(self.convs) - 1:
                h_nei_full = clamp_symmetric(F.relu(h_nei_full))
                if self.gcn_dropout > 0:
                    h_nei_full = F.dropout(
                        h_nei_full, p=self.gcn_dropout, training=self.training
                    )

        # Mix self + neighbors (skip connection)
        h_full = clamp_symmetric(self._mix_self_and_neighbors(h_self_full, h_nei_full))

        # ---- STEP 2: subset to minibatch ----
        h = clamp_symmetric(h_full[batch_indices])  # [batch_size, n_hidden]

        # scvi-style mean/variance heads
        q_m = clamp_symmetric(self.mean_encoder(h))
        raw_q_v = clamp_positive(self.var_activation(self.var_encoder(h)))
        q_v = clamp_positive(raw_q_v + self.var_eps)

        # NaN detection warning
        if torch.isnan(q_m).any() or torch.isnan(q_v).any():
            nan_info = []
            if torch.isnan(h_self_full).any():
                nan_info.append("h_self_full")
            if torch.isnan(h_nei_full).any():
                nan_info.append("h_nei_full")
            if torch.isnan(h_full).any():
                nan_info.append("h_full")
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
        likelihood_weight_mode: str = "none",
        likelihood_weight_ref: str = "mean",
        init_bg_mean: list[torch.Tensor | None] | None = None,
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

        # Feature background initialization (scTM-style)
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
    ):
        # ----- topic-feature distributions (per modality) -----
        topic_feature_dists = []  # will store φₖ,ₘ tensors

        if self.topic_feature_prior_type == "logistic_normal":
            # EXISTING: Logistic-Normal prior
            # First, sample backgrounds outside of topics plate (per-feature, not per-topic)
            # Note: Background terms only apply to count-based likelihoods (gamma_poisson, nb).
            # Bernoulli and multinomial likelihoods do not use feature backgrounds.
            bg_samples = []
            for m in range(self.n_modalities):
                if self.use_feature_background and self.likelihoods[m] in {"gamma_poisson", "nb"}:
                    init_bg = getattr(self, f"init_bg_mean_{m}")
                    if init_bg.numel() > 1:  # Not a placeholder
                        with poutine.scale(scale=kl_weight):
                            bg_m = pyro.sample(
                                f"bg_{m}",
                                dist.Normal(torch.zeros_like(init_bg), torch.ones_like(init_bg)).to_event(1)
                            )
                        bg_m = bg_m + init_bg  # Add empirical baseline
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

                    topic_feature_dists.append(F.softmax(log_phi, dim=-1))  # (K, Fₘ)

        elif self.topic_feature_prior_type == "horseshoe":
            # NEW: Horseshoe prior (following scTM)
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
                        with poutine.scale(scale=kl_weight):
                            bg_m = pyro.sample(
                                f"bg_{m}",
                                dist.Normal(torch.zeros_like(init_bg), torch.ones_like(init_bg)).to_event(1)
                            )
                        bg_m = bg_m + init_bg  # Add empirical baseline
                        beta_shrunk_m = beta_shrunk_m + bg_m.unsqueeze(0)  # (K, F_m) + (1, F_m) -> (K, F_m)

                # Register as named sample for guide to match
                pyro.deterministic(f"log_topic_feature_dist_{m}", beta_shrunk_m)

                # 8. Convert to probability simplex
                topic_feature_dists.append(F.softmax(beta_shrunk_m, dim=-1))

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
                phi_m = topic_feature_dists[m]  # (K, F_m)
                rate_m = theta @ phi_m  # (B, F_m)
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
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
        topic_feature_prior_type: str = "logistic_normal",
        use_feature_background: bool = True,
        likelihoods: list[str] | None = None,
        learnable_dispersion: bool = False,  # whether to learn dispersion (STAMP-like)
        global_dispersion: bool = True,  # if learnable: one global vs per-gene dispersion
        normalize_encoder_inputs: bool = True,
        encoder_scale_factor: float = 1e4,
        entropy_weight: float = 0.01,
        topic_variance_weight: float = 1.0,
    ) -> None:
        super().__init__("multimodal_lda_guide")
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.n_topics = n_topics
        self.use_gcn = spatial
        self.adjacency = adjacency  # keep reference for downstream checks/tests
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.likelihoods = likelihoods if likelihoods is not None else ["gamma_poisson"] * self.n_modalities
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion
        self.normalize_encoder_inputs = normalize_encoder_inputs
        self.encoder_scale_factor = encoder_scale_factor
        self.entropy_weight = entropy_weight
        self._last_entropy = None  # For logging
        self.topic_variance_weight = topic_variance_weight
        self._last_topic_variance = None  # For logging

        # Regular encoders (always present)
        self.encoders = torch.nn.ModuleList(
            [
                Encoder(n_in, n_topics, distribution="ln", return_dist=True, n_hidden=n_hidden)
                for n_in in n_inputs_modalities
            ]
        )
        
        # GCN encoders and edge indices (if spatial)
        if self.use_gcn:
            if adjacency is None:
                raise ValueError("GCN encoder requested (spatial=True) but no adjacency was provided.")
            
            # Create GCN encoders
            self.gcn_encoders = torch.nn.ModuleList(
                [
                    GCNEncoder(
                        n_in,
                        n_topics,
                        n_hidden,
                        gcn_n_layers=gcn_n_layers,
                        gcn_hidden_dims=gcn_hidden_dims,
                    )
                    for n_in in n_inputs_modalities
                ]
            )
            
            # Convert adjacency to edge_index ONCE and store as buffers
            if isinstance(adjacency, (list, tuple)):
                # Multiple adjacencies (one per modality)
                for idx, adj in enumerate(adjacency):
                    edge_index = adjacency_to_edge_index(adj)
                    self.register_buffer(f"edge_index_{idx}", edge_index)
                self.multiple_adjacencies = True
            else:
                # Single shared adjacency for all modalities
                edge_index = adjacency_to_edge_index(adjacency)
                self.register_buffer("edge_index", edge_index)
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

            for F_m in n_inputs_modalities:
                # Initialize caux
                self.caux_loc.append(torch.nn.Parameter(torch.ones(1)))
                self.caux_scale.append(torch.nn.Parameter(torch.ones(1)))

                # Initialize tau (per-topic)
                self.tau_loc.append(torch.nn.Parameter(torch.zeros(n_topics)))
                self.tau_scale.append(torch.nn.Parameter(torch.ones(n_topics)))

                # Initialize delta (per-feature)
                self.delta_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                self.delta_scale.append(torch.nn.Parameter(torch.ones(F_m)))

                # Initialize lambda (per-topic-feature)
                self.lambda_loc.append(torch.nn.Parameter(torch.zeros(n_topics, F_m)))
                self.lambda_scale.append(torch.nn.Parameter(torch.ones(n_topics, F_m)))

                # Initialize beta (per-topic-feature)
                self.beta_loc.append(torch.nn.Parameter(torch.zeros(n_topics, F_m)))
                self.beta_scale.append(torch.nn.Parameter(torch.ones(n_topics, F_m)))

        # Feature background variational parameters (scTM-style)
        # Only for gamma_poisson modalities when use_feature_background=True
        if use_feature_background:
            self.bg_loc = torch.nn.ParameterList()
            self.bg_scale = torch.nn.ParameterList()
            for m, (F_m, likelihood) in enumerate(zip(n_inputs_modalities, self.likelihoods)):
                if likelihood == "gamma_poisson":
                    # Create background parameters for gamma_poisson modalities
                    self.bg_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                    self.bg_scale.append(torch.nn.Parameter(torch.ones(F_m)))
                else:
                    # Placeholder for multinomial modalities (won't be used)
                    self.bg_loc.append(None)
                    self.bg_scale.append(None)
        else:
            self.bg_loc = None
            self.bg_scale = None

        # Dispersion variational parameters (for learnable dispersion, STAMP-like)
        # LogNormal posterior: disp ~ LogNormal(disp_loc, softplus(disp_scale))
        if learnable_dispersion:
            self.disp_loc = torch.nn.ParameterList()
            self.disp_scale = torch.nn.ParameterList()
            for m, (F_m, likelihood) in enumerate(zip(n_inputs_modalities, self.likelihoods)):
                if likelihood in {"gamma_poisson", "nb"}:
                    if global_dispersion:
                        # Single dispersion per modality
                        self.disp_loc.append(torch.nn.Parameter(torch.zeros(1)))
                        self.disp_scale.append(torch.nn.Parameter(torch.ones(1)))
                    else:
                        # Per-gene dispersion (STAMP-like)
                        self.disp_loc.append(torch.nn.Parameter(torch.zeros(F_m)))
                        self.disp_scale.append(torch.nn.Parameter(torch.ones(F_m)))
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

    @staticmethod
    def _softplus(t: torch.Tensor) -> torch.Tensor:
        return F.softplus(t)

    def _mix_gaussians(
        self, mus: torch.Tensor, vars_: torch.Tensor, masks: torch.Tensor, cell_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mix (*M,B,K*) Gaussians using weights w_{m,c}.

        Parameters
        ----------
        mus/vars_ : (M , B , K)
        masks     : (M , B)  (0 = modality absent)
        """
        if self.mod_w is None:  # equal
            w = torch.ones_like(masks)
        elif self.mod_w.dim() == 1:  # universal
            w = self.mod_w.view(-1, 1).expand_as(masks)
        else:  # cell
            w = self.mod_w[cell_idx, :]  # (B , M) -> transpose
            w = w.T  # (M , B)

        w = masked_softmax(w, masks, dim=0).unsqueeze(-1)  # (M,B,1)
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
    ):
        """
        Initialize GCN encoders with full graph data for full-graph training.

        CRITICAL: When normalize_encoder_inputs=True, this applies log-normalization
        to the FULL graph BEFORE storing it in the GCN encoder. This ensures that
        spatial graph convolution operates on normalized features.

        Must be called after model initialization and before training when spatial=True.

        Parameters
        ----------
        x_full_modalities : list[torch.Tensor] or torch.Tensor
            Either:
            - List of full feature matrices, one per modality [n_obs, n_features_m]
            - Single concatenated matrix [n_obs, sum(n_features)] (will be split)

        Notes
        -----
        Edge indices are already stored as buffers in __init__.
        This method only needs to initialize the feature data.
        """
        if not self.use_gcn:
            logger.warning("set_full_graph_data() called but spatial=False. No effect.")
            return

        # Split concatenated features if needed
        if isinstance(x_full_modalities, torch.Tensor):
            x_full_list = torch.split(x_full_modalities, self.n_inputs_modalities, dim=1)
        else:
            x_full_list = x_full_modalities

        # IMPORTANT: Apply normalization BEFORE storing in GCN encoder
        # This ensures graph convolution operates on normalized features
        if self.normalize_encoder_inputs:
            libs = [x_full_m.sum(dim=1, keepdim=True) for x_full_m in x_full_list]
            x_full_normalized = []
            for x_full_m, lib_m in zip(x_full_list, libs):
                lib_m = torch.clamp(lib_m, min=1.0)
                # Use median depth of this modality as scale factor
                median_depth = torch.median(lib_m)
                x_full_m_norm = torch.log1p(x_full_m / lib_m * median_depth)
                x_full_normalized.append(x_full_m_norm)
            x_full_list = x_full_normalized
            logger.info("Applied library-size and log-normalization to full graph BEFORE spatial convolution")

        # Initialize each GCN encoder with (potentially normalized) full graph
        for idx, (gcn_enc, x_full_m) in enumerate(zip(self.gcn_encoders, x_full_list)):
            edge_index = self._get_edge_index(idx)
            # This stores x_full_m in gcn_enc.x_full
            # During forward pass, GCN will do: conv(x_full, edge_index)
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
                if self.use_feature_background and self.bg_loc is not None and self.bg_loc[m] is not None:
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
                if self.use_feature_background and self.bg_loc is not None and self.bg_loc[m] is not None:
                    with poutine.scale(scale=kl_weight):
                        pyro.sample(
                            f"bg_{m}",
                            dist.Normal(self.bg_loc[m], self._softplus(self.bg_scale[m])).to_event(1)
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

        # θₙ variational
        xs = torch.split(x, self.n_inputs_modalities, dim=1)  # (B,Fₘ)

        # Apply normalization + log transform if requested
        if self.normalize_encoder_inputs:
            # Compute library sizes per modality
            libs = [x_m.sum(dim=1, keepdim=True) for x_m in xs]  # List of (B, 1)
            # Normalize to median depth per modality
            xs_normalized = []
            for x_m, lib_m in zip(xs, libs):
                lib_m = torch.clamp(lib_m, min=1.0)  # Avoid division by zero
                # Use median depth of this modality as scale factor
                median_depth = torch.median(lib_m)
                x_m_norm = torch.log1p(x_m / lib_m * median_depth)
                xs_normalized.append(x_m_norm)
            xs = xs_normalized

        mus, vars_, masks = [], [], []
        for idx, (enc, x_m) in enumerate(zip(self.encoders, xs, strict=False)):
            if self.use_gcn:
                edge_index = self._get_edge_index(idx)
                q_m, _ = self.gcn_encoders[idx](x_m, edge_index, batch_indices=batch_indices)
            else:
                q_m, _ = enc(x_m)  # q(z|x)
            mus.append(clamp_symmetric(q_m.loc))
            vars_.append(clamp_positive(q_m.scale**2))
            # Compute mask from ORIGINAL x (before normalization)
            original_xs = torch.split(x, self.n_inputs_modalities, dim=1)
            masks.append((original_xs[idx].sum(1) > 0).float())
        mus = torch.stack(mus)  # (M,B,K)
        vars_ = torch.stack(vars_)  # (M,B,K)
        masks = torch.stack(masks)  # (M,B)

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
# Wrapper module pairing model & guide with TRANSDUCTIVE LEARNING support
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
        init_bg_mean: list[torch.Tensor | None] | None = None,
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
        gcn_n_layers: int = 1,
        gcn_hidden_dims: list[int] | None = None,
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
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.dispersion_rna = dispersion_rna
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion
        self.normalize_encoder_inputs = normalize_encoder_inputs
        self.encoder_scale_factor = encoder_scale_factor
        self.entropy_weight = entropy_weight
        self.topic_variance_weight = topic_variance_weight
        self.kl_weight = kl_weight

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
            likelihood_weight_mode=likelihood_weight_mode,
            likelihood_weight_ref=likelihood_weight_ref,
            init_bg_mean=init_bg_mean,
        )
        self._guide = MultimodalLDAPyroGuide(
            n_inputs_modalities,
            n_topics,
            n_hidden,
            gcn_n_layers=gcn_n_layers,
            gcn_hidden_dims=gcn_hidden_dims,
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
            spatial=spatial,
            adjacency=adjacency,
            topic_feature_prior_type=topic_feature_prior_type,
            use_feature_background=use_feature_background,
            likelihoods=likelihoods,
            learnable_dispersion=learnable_dispersion,
            global_dispersion=global_dispersion,
            normalize_encoder_inputs=normalize_encoder_inputs,
            encoder_scale_factor=encoder_scale_factor,
            entropy_weight=entropy_weight,
            topic_variance_weight=topic_variance_weight,
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


    def set_full_graph_data(self, x_full: torch.Tensor):
        """
        Initialize GCN encoders with full graph data for semi-supervised learning.
        
        Must be called after model initialization and before training when spatial=True.
        
        Parameters
        ----------
        x_full : torch.Tensor
            Full concatenated feature matrix for ALL cells [n_obs, sum(n_features)]
            Will be automatically split by modality
        """
        if not self.spatial:
            logger.warning("set_full_graph_data() called but spatial=False. No effect.")
            return
        
        self._guide.set_full_graph_data(x_full)
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
                tbf = torch.mean(
                    F.softmax(
                        dist.Normal(mu.detach().cpu(), sig.detach().cpu()).sample((n_samples,)),
                        dim=2
                    ),
                    dim=0,
                )
                out[m] = tbf  # (K, F_m)

        elif self.guide.topic_feature_prior_type == "horseshoe":
            # Sample from beta posterior and apply softmax
            for m, (mu, sig) in enumerate(
                zip(self.guide.beta_loc, self.guide.beta_scale, strict=False)
            ):
                beta_samples = dist.Normal(
                    mu.detach().cpu(),
                    self.guide._softplus(sig).detach().cpu()
                ).sample((n_samples,))  # (n_samples, K, F_m)

                tbf = torch.mean(F.softmax(beta_samples, dim=2), dim=0)  # (K, F_m)
                out[m] = tbf

        return out

    def get_topic_distribution(self, x: torch.Tensor, n_samples: int = 5_000) -> torch.Tensor:
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)

        B = x.shape[0]
        xs = torch.split(x, self.model.n_inputs_modalities, dim=1)

        # Apply normalization + log transform if requested (must match training)
        if self.guide.normalize_encoder_inputs:
            libs = [x_m.sum(dim=1, keepdim=True) for x_m in xs]
            xs_normalized = []
            for x_m, lib_m in zip(xs, libs):
                lib_m = torch.clamp(lib_m, min=1.0)
                # Use median depth of this modality as scale factor
                median_depth = torch.median(lib_m)
                x_m_norm = torch.log1p(x_m / lib_m * median_depth)
                xs_normalized.append(x_m_norm)
            xs = xs_normalized

        batch_indices = torch.arange(B, device=x.device)

        # run encoders
        mus, vars_, masks = [], [], []
        for idx, (enc, x_m) in enumerate(zip(self.guide.encoders, xs, strict=False)):
            if self.guide.use_gcn:
                edge_index = self.guide._get_edge_index(idx)
                q_m, _ = self.guide.gcn_encoders[idx](x_m, edge_index, batch_indices=batch_indices)
            else:
                q_m, _ = enc(x_m)
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
