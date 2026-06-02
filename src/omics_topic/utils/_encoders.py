"""Spatial encoder ``nn.Module`` building blocks for the amortized LDA guide.

Contains the graph-based encoders:

* :class:`GCNEncoder` — GCN/GAT encoder with k-hop subgraph sampling.
* :class:`SGCEncoder` — STAMP-style MLP on precomputed SGC features.
"""

from __future__ import annotations

import logging

import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.nn import FCLayers
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import k_hop_subgraph

from omics_topic.utils._amortized_utils import clamp_positive, clamp_symmetric

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
