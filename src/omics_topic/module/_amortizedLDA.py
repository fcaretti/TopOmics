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
from scvi.nn import Encoder
from torch_geometric.nn import GCNConv, GATv2Conv

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Helper utils
# --------------------------------------------------------------------------------------------------


def logistic_normal_approximation(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Laplace approximation parameters (μ, σ) of Logistic-Normal ≈ Dirichlet(α)."""
    K = alpha.shape[-1]
    mu = torch.log(alpha) - torch.log(alpha).sum() / K
    sigma = torch.sqrt((1 - 2 / K) / alpha + torch.sum(1 / alpha) / K**2)
    return mu, sigma


def masked_softmax(weights: torch.Tensor, mask: torch.Tensor, dim: int = 0):
    """Softmax **ignoring** masked entries (mask == 0)."""
    weights = weights.masked_fill(~mask.bool(), -1e9)
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



class GCNEncoder(nn.Module):
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
        conv_type: str = 'GATv2Conv', # can also be GCNConv
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
                dropout=dropout,
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
        Initialize full graph data for transductive learning.
        
        Must be called after model initialization and before training.
        
        Parameters
        ----------
        x_full : torch.Tensor
            Full gene expression matrix for ALL cells [n_obs, n_features]
        edge_index_full : torch.Tensor
            Full graph edge indices in COO format [2, n_edges]
        """
        self.x_full = x_full
        self.edge_index_full = edge_index_full
        self._graph_initialized = True
        logger.info(f"GCN graph initialized: {x_full.shape[0]} cells, {edge_index_full.shape[1]} edges")
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_indices: torch.Tensor | None = None):
        """
        Forward pass with transductive learning.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (batch_size, n_in) - IGNORED for transductive learning
        edge_index : torch.Tensor
            Graph connectivity - IGNORED for transductive learning
        batch_indices : torch.Tensor, optional
            Indices of cells in current batch [batch_size]
            Required for transductive learning (when graph is initialized)
        
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
        
        return dist.Normal(mu, scale), None


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
    ) -> None:
        super().__init__("multimodal_lda")

        assert len(n_inputs_modalities) == len(likelihoods) == len(topic_feature_priors)
        self.n_modalities = len(n_inputs_modalities)
        self.n_topics = n_topics
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.dispersion_rna = dispersion_rna  # used for Gamma-Poisson / NB

        # Pre-compute Logistic-Normal approximations for priors
        cell_mu, cell_sigma = logistic_normal_approximation(cell_topic_prior)
        self.register_buffer("cell_mu", cell_mu)
        self.register_buffer("cell_sigma", cell_sigma)

        self.topic_prior_mus = torch.nn.ParameterList()
        self.topic_prior_sigmas = torch.nn.ParameterList()
        for t_prior in topic_feature_priors:
            mu_m, sig_m = logistic_normal_approximation(t_prior)
            self.topic_prior_mus.append(torch.nn.Parameter(mu_m, requires_grad=False))
            self.topic_prior_sigmas.append(torch.nn.Parameter(sig_m, requires_grad=False))

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
    ):
        # ----- topic-feature distributions (per modality) -----
        topic_feature_dists = []  # will store φₖ,ₘ tensors
        with pyro.plate("topics", self.n_topics):
            for m in range(self.n_modalities):
                mu_m = self.topic_prior_mus[m].unsqueeze(0).expand(self.n_topics, -1)
                sig_m = self.topic_prior_sigmas[m].unsqueeze(0).expand(self.n_topics, -1)

                with poutine.scale(scale=kl_weight):
                    log_phi = pyro.sample(f"log_topic_feature_dist_{m}", dist.Normal(mu_m, sig_m).to_event(1))
                topic_feature_dists.append(F.softmax(log_phi, dim=-1))  # (K, Fₘ)

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
                lib_m = libraries[:, m]
                phi_m = topic_feature_dists[m]  # (K, F_m)
                rate_m = theta @ phi_m  # (B, F_m)

                if L_m == "multinomial":
                    N_max = int(lib_m.max().item())
                    pyro.sample(
                        f"feature_counts_{m}",
                        CategoricalBoW(N_max, rate_m),
                        obs=x_m,
                    )
                elif L_m in {"gamma_poisson", "nb"}:
                    # mean scaled by lib; dispersion shared globally per feature set
                    mu = rate_m * lib_m.unsqueeze(-1)
                    r = torch.tensor(self.dispersion_rna, device=mu.device)
                    pyro.sample(
                        f"feature_counts_{m}",
                        dist.NegativeBinomial(total_count=r, probs=mu / (mu + r)).to_event(1),
                        obs=x_m,
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
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> None:
        super().__init__("multimodal_lda_guide")
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.n_topics = n_topics
        self.use_gcn = spatial
        self.adjacency = adjacency  # keep reference for downstream checks/tests

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
                [GCNEncoder(n_in, n_topics, n_hidden) for n_in in n_inputs_modalities]
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
        self.topic_feature_posterior_mu = torch.nn.ParameterList()
        self.unconstrained_topic_feature_posterior_sigma = torch.nn.ParameterList()
        for F_m in n_inputs_modalities:
            mu_m, sig_m = logistic_normal_approximation(torch.ones(F_m))
            self.topic_feature_posterior_mu.append(torch.nn.Parameter(mu_m.repeat(n_topics, 1)))
            self.unconstrained_topic_feature_posterior_sigma.append(torch.nn.Parameter(sig_m.repeat(n_topics, 1)))

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
        var = (w * vars_).sum(0)
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
        return [self._softplus(u) for u in self.unconstrained_topic_feature_posterior_sigma]

    # ═══════════════════════════════════════════════════════════════════════
    # NEW METHOD: Initialize full graph data for semi-supervised learning
    # ═══════════════════════════════════════════════════════════════════════
    def set_full_graph_data(
        self, 
        x_full_modalities: list[torch.Tensor] | torch.Tensor,
    ):
        """
        Initialize GCN encoders with full graph data for transductive learning.
        
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
        
        # Initialize each GCN encoder with its modality's full data
        for idx, (gcn_enc, x_full_m) in enumerate(zip(self.gcn_encoders, x_full_list)):
            edge_index = self._get_edge_index(idx)
            gcn_enc.set_full_graph_data(x_full_m, edge_index)
        
        logger.info("All GCN encoders initialized with full graph data for transductive learning")

    # ---------- forward ----------
    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,  # (B , ΣF)
        libraries: torch.Tensor,  # unused – kept for signature parity
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        B = x.shape[0]
        
        batch_indices = torch.arange(B, device=x.device)
        
        cell_idx = batch_indices  # for weight mixing (if weight_mode=="cell")

        # ϕₖ,ₘ variational dists
        for m in range(self.n_modalities):
            with pyro.plate(f"topics_{m}", self.n_topics):
                with poutine.scale(scale=kl_weight):
                    pyro.sample(
                        f"log_topic_feature_dist_{m}",
                        dist.Normal(self.topic_feature_posterior_mu[m], self.topic_feature_posterior_sigma[m]).to_event(
                            1
                        ),
                    )

        # θₙ variational
        xs = torch.split(x, self.n_inputs_modalities, dim=1)  # (B,Fₘ)
        mus, vars_, masks = [], [], []
        for idx, (enc, x_m) in enumerate(zip(self.encoders, xs, strict=False)):
            if self.use_gcn:
                edge_index = self._get_edge_index(idx)
                q_m, _ = self.gcn_encoders[idx](x_m, edge_index, batch_indices=batch_indices)
            else:
                q_m, _ = enc(x_m)  # q(z|x)
            mus.append(q_m.loc)
            vars_.append(q_m.scale**2)
            masks.append((x_m.sum(1) > 0).float())
        mus = torch.stack(mus)  # (M,B,K)
        vars_ = torch.stack(vars_)  # (M,B,K)
        masks = torch.stack(masks)  # (M,B)

        muθ, varθ = self._mix_gaussians(mus, vars_, masks, cell_idx)

        with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=B), poutine.scale(None, kl_weight):
            pyro.sample("log_cell_topic_dist", dist.Normal(muθ, torch.sqrt(varθ)).to_event(1))


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
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
        spatial: bool = False,
        adjacency: torch.Tensor | Sequence[torch.Tensor] | None = None,
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
        )
        self._guide = MultimodalLDAPyroGuide(
            n_inputs_modalities,
            n_topics,
            n_hidden,
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
            spatial=spatial,
            adjacency=adjacency,
        )

        # We need this method so scvi training plan can create data-loader args
        def _args_from_batch(tdict):
            tdict["n_inputs_modalities"] = n_inputs_modalities  # inject for library calc
            return self._model._get_fn_args_from_batch(tdict)

        self._get_fn_args_from_batch = _args_from_batch

    # proxies
    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide


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
        logger.info("Module initialized for transductive learning with full graph data")

    # utilities
    def topic_by_feature(self, n_samples: int = 5_000):
        out = {}
        for m, (mu, sig) in enumerate(
            zip(self.guide.topic_feature_posterior_mu, self.guide.topic_feature_posterior_sigma, strict=False)
        ):
            tbf = torch.mean(
                F.softmax(dist.Normal(mu.detach().cpu(), sig.detach().cpu()).sample((n_samples,)), dim=2),
                dim=0,
            )
            out[m] = tbf  # user can map index→ modality name externally
        return out

    def get_topic_distribution(self, x: torch.Tensor, n_samples: int = 5_000) -> torch.Tensor:
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)

        B = x.shape[0]
        xs = torch.split(x, self.model.n_inputs_modalities, dim=1)
        
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
            masks.append((x_m.sum(1) > 0).float())
        mus = torch.stack(mus)
        vars_ = torch.stack(vars_)
        masks = torch.stack(masks)
        muθ, varθ = self.guide._mix_gaussians(mus, vars_, masks, torch.arange(B, device=x.device))
        samps = dist.Normal(muθ, torch.sqrt(varθ)).sample((n_samples,))
        return torch.softmax(samps, -1).mean(0)

    @auto_move_data
    @torch.inference_mode()
    def get_elbo(self, x: torch.Tensor, libs: torch.Tensor, n_obs: int):
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)
        return Trace_ELBO().loss(self.model, self.guide, x, libs, n_obs=n_obs)
