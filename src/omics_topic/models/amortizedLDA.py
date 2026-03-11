# multimodal_amortized_lda.py
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyro
from pyro.infer import Trace_ELBO
import scipy.sparse as sp
import torch
from anndata import AnnData
from mudata import MuData
from scvi._constants import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model.base import BaseModelClass, PyroSviTrainMixin
from scvi.utils import setup_anndata_dsp

from omics_topic.module._amortizedLDA import MultimodalAmortizedLDAPyroModule
from omics_topic.utils.training_plan import MultimodalLDAPyroTrainingPlan

from .base_model import BaseTopicModel

if TYPE_CHECKING:
    from collections.abc import Sequence as _Seq

logger = logging.getLogger(__name__)


def _resolve_spatial_graph_from_adata(adata: AnnData, spatial_key: str | None):
    """
    Fetch a precomputed spatial graph from ``adata.obsp`` when provided.

    Assumes Scanpy already built the graph; we only check basic shape alignment.
    """
    if spatial_key is None:
        return None
    if spatial_key not in adata.obsp:
        raise KeyError(f"spatial_key '{spatial_key}' not found in adata.obsp")

    graph = adata.obsp[spatial_key]
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(
            f"Spatial graph at obsp['{spatial_key}'] has shape {graph.shape}, expected ({adata.n_obs}, {adata.n_obs})"
        )
    return {"adjacency": graph, "key": spatial_key}


def _normalize_to_torch_sparse(adj) -> torch.Tensor:
    """Convert (and normalise) a CSR/COO adjacency to torch sparse with self-loops."""
    if sp.issparse(adj):
        coo = adj.tocoo()
    else:
        coo = sp.coo_matrix(adj)

    # add self-loops
    coo = (coo + sp.eye(coo.shape[0], format="coo")).tocoo()
    row, col, data = coo.row, coo.col, coo.data
    deg = np.asarray(coo.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    norm = 1.0 / np.sqrt(deg)
    norm_data = data * norm[row] * norm[col]

    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    values = torch.tensor(norm_data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


def _prepare_adjacency_tensors(spatial_uns, modality_names: list[str] | None = None):
    """Build torch sparse adjacency tensor(s) from stored spatial graph metadata."""
    if spatial_uns is None:
        return None

    def convert(entry):
        adj = entry.get("adjacency")
        if adj is None:
            raise ValueError("Spatial graph entry missing 'adjacency'.")
        return _normalize_to_torch_sparse(adj)

    if isinstance(spatial_uns, dict) and "adjacency" in spatial_uns:
        return convert(spatial_uns)

    if not isinstance(spatial_uns, dict):
        raise ValueError("Expected spatial graph metadata as dict or mapping.")

    order = modality_names or list(spatial_uns.keys())
    tensors = []
    for mod in order:
        if mod not in spatial_uns:
            continue
        tensors.append(convert(spatial_uns[mod]))
    return tensors if tensors else None


class MultimodalAmortizedLDA(PyroSviTrainMixin, BaseModelClass, BaseTopicModel):
    """
    **Multimodal Amortized LDA with Mixture-of-Experts (MoE)**

    Extends :class:`scvi.model.AmortizedLDA` to *M* modalities with
    modality-specific encoders and likelihoods. Each modality is encoded
    separately, and representations are mixed via weighted Gaussian combination
    before inferring the shared cell-topic distribution θₙ.

    Parameters
    ----------
    adata
        :class:`~anndata.AnnData` with *concatenated* features (RNA + protein + …).
    n_inputs_modalities
        List with feature counts per modality, in the order they appear in ``adata.X``.
    likelihoods
        Length-matched list of likelihood strings for each modality.
    n_topics
        Number of topics (K).
    n_hidden
        Hidden units of each encoder network.
    cell_topic_prior
        Dirichlet concentration for θₙ.  ``None`` ⇒ symmetric 1/K.
    topic_feature_prior
        Dirichlet concentration for each ϕₖ,ₘ.  ``None`` ⇒ symmetric 1/K.
    weight_mode
        How to weight modality-specific representations:
        - ``"equal"``: All modalities weighted equally (default)
        - ``"universal"``: Learn a single weight per modality
        - ``"cell"``: Learn per-cell, per-modality weights

    Notes
    -----
    The Mixture-of-Experts architecture processes each modality through a
    separate encoder network, then combines their latent representations
    using learned or fixed weights. This allows the model to handle
    heterogeneous data types and missing modalities.
    ```
    """

    _module_cls = MultimodalAmortizedLDAPyroModule  # type: ignore
    _training_plan_cls = MultimodalLDAPyroTrainingPlan

    # --------------------------------------------------------------------- #
    #                                init                                   #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        adata: AnnData,
        n_inputs_modalities: list[int],
        likelihoods: list[str],
        n_topics: int = 20,
        n_hidden: int = 128,
        cell_topic_prior: float | Sequence[float] | None = None,
        topic_feature_prior: float | Sequence[float] | None = None,
        topic_feature_prior_type: str = "logistic_normal",
        use_feature_background: bool = True,
        dispersion_rna: float = 1.0,
        learnable_dispersion: bool = False,
        global_dispersion: bool = True,
        modality_names: list[str] | None = None,
        weight_mode: str = "cell",
        likelihood_weight_mode: str = "none",
        likelihood_weight_ref: str = "mean",
        gcn_n_layers: int = 1,
        gcn_conv_type: str = 'GATv2Conv',
        gcn_hidden_dims: list[int] | None = None,
        gcn_alpha_init: float = 0.7,
        gcn_use_learned_alpha: bool = True,
        normalize_encoder_inputs: bool = True,
        encoder_scale_factor: float = 1e6,
        entropy_weight: float = 0.01,
        topic_variance_weight: float = 1.0,
        kl_weight: float = 1.0,
        encode_covariates: bool = True,
        bg_offset: float = 1.0,
        learnable_bg: bool = True,
    ):
        """
        Initialize MultimodalAmortizedLDA with Mixture-of-Experts (MoE) architecture.

        Parameters
        ----------
        adata
            AnnData with concatenated features (for scvi compatibility).
            Can also be MuData if setup_mudata() or setup_data() was called.
        n_inputs_modalities
            List of feature counts per modality.
        likelihoods
            List of likelihood models, one per modality.
            Options:
                - "multinomial": For discrete count data (RNA-seq)
                - "gamma_poisson" or "nb": For overdispersed count data (scRNA-seq, protein)
                - "bernoulli": For binary presence/absence data (ATAC-seq peaks, methylation)
        n_topics
            Number of topics.
        n_hidden
            Hidden units in encoder networks.
        cell_topic_prior
            Dirichlet concentration for θₙ.
        topic_feature_prior
            Dirichlet concentration for ϕₖ,ₘ.
        topic_feature_prior_type
            Type of prior for topic-feature distributions ϕₖ,ₘ (default: "logistic_normal"):

            - ``"logistic_normal"``: Logistic-Normal approximation of Dirichlet (current behavior)
            - ``"horseshoe"``: Regularized horseshoe prior for adaptive sparsity (scTM-style)

            The horseshoe prior induces sparsity in topic-feature associations, making topics
            more interpretable by pushing many features toward zero while preserving important
            features. Useful for high-dimensional feature spaces (genes, proteins, peaks).
        use_feature_background
            Whether to use feature-specific background terms (default: True).
            Only applies to modalities with ``likelihood="gamma_poisson"``.
            Following scTM, this separates baseline feature expression from topic-specific
            variation by adding a learned background term bg_m to log-probabilities.
            Ignored for ``likelihood="multinomial"`` modalities.
        dispersion_rna
            Fixed dispersion value for Negative Binomial likelihood when
            ``learnable_dispersion=False``. Default: 1.0.
        learnable_dispersion
            Whether to learn dispersion parameters (True) or use fixed dispersion (False).
            When True, dispersion is sampled with a HalfCauchy prior in the model and
            LogNormal variational posterior in the guide (STAMP-like behavior).
            Default: False (backward compatible).
        global_dispersion
            When ``learnable_dispersion=True``:

            - If True: learn one global dispersion per modality (single scalar)
            - If False: learn per-gene dispersion (one parameter per feature, STAMP-like)

            Ignored when ``learnable_dispersion=False``. Default: True.
        modality_names
            Optional list of modality names (e.g., ["rna", "protein"]). If None, uses indices.
        weight_mode
            How to weight modality-specific representations when mixing:

            - "equal": All modalities weighted equally (default, simplest)
            - "universal": Learn a single weight per modality across all cells
            - "cell": Learn per-cell, per-modality weights (most flexible)
        likelihood_weight_mode
            How to rescale per-modality likelihood terms (default: "none"):

            - "none": no rescaling (current behavior)
            - "inverse_features": scale by (F_ref / F_m)
            - "sqrt_inverse_features": scale by sqrt(F_ref / F_m)
        likelihood_weight_ref
            Reference feature count for rescaling (default: "mean"):
            one of {"mean", "median", "max"}.
        gcn_n_layers
            Number of graph convolution layers for spatial encoders (default: 1).
        gcn_hidden_dims
            Optional list of hidden sizes for each GCN layer (length = gcn_n_layers).
        normalize_encoder_inputs
            If ``True``, normalize counts by library size and apply log1p before encoding.
            Each modality is normalized to its own median sequencing depth:
            ``log(counts / library_size * median_depth + 1)``
            This is standard preprocessing in scRNA-seq analysis. Default: ``True``.
        encoder_scale_factor
            Deprecated. Scale factor is now computed dynamically as the median depth
            per modality. This parameter is kept for backward compatibility but is ignored.
        entropy_weight
            Weight for entropy regularization term (default: 0.0).
            When > 0, adds an entropy bonus to the ELBO objective to encourage diverse
            topic usage and prevent topic collapse:
            ``Objective = ELBO + entropy_weight * Σ_n H(θ_n)``
            where H(θ_n) = -Σ_k θ_n,k * log(θ_n,k) is the entropy of cell-topic distribution.
            Typical values: 0.001-0.1. Higher values → more uniform topic distributions.
            Trade-off: too high can hurt reconstruction quality.
        topic_variance_weight
            Weight for topic variance regularization (default: 0.0).
            When > 0, encourages different cells to use different topics, preventing
            cell collapse where all cells have identical topic distributions.
            ``Objective = ELBO + topic_variance_weight * Σ_k Var(θ_:,k)``
            where Var(θ_:,k) is the variance of topic k usage across cells in the batch.
            Typical values: 1.0-10.0 (higher than entropy_weight because variance is smaller).
            This regularization is complementary to entropy_weight: entropy encourages
            each cell to use many topics uniformly, while topic variance encourages
            different cells to specialize in different topics.
        kl_weight
            Weight applied to KL terms in the ELBO (default: 1.0).
            This scales KL contributions in both the model and guide.
        encode_covariates
            Whether to concatenate covariates to encoder input (default: True).
            If True, covariates are used in both encoder and decoder (batch-corrected latent space).
            If False, covariates are only used in decoder (scVI-style decoder-only correction).
        bg_offset
            Offset added to mean expression before log-transform for feature background:
            ``init_bg_mean = log(mean_expr + bg_offset)``
            Default: 1.0. STAMP uses 1e-15 which preserves the full dynamic range of
            log-expression. Higher values compress the range, requiring the model to
            learn larger deviations from the background.

        Notes
        -----
        The model uses Mixture-of-Experts architecture where each modality is encoded
        separately and then mixed via weighted Gaussian combination before sampling
        the shared cell-topic distribution θₙ.
        """
        # If MuData provided, extract flattened AnnData
        if hasattr(adata, 'mod'):  # MuData check
            if "_flattened_ann_data" in adata.uns:
                adata = adata.uns["_flattened_ann_data"]
            else:
                raise ValueError(
                    "MuData must be setup with setup_mudata() or setup_data() first. "
                    "Call MultimodalAmortizedLDA.setup_data(mdata, ...) before instantiation."
                )

        pyro.clear_param_store()
        super().__init__(adata)

        # Resolve spatial metadata early so module/guide can consume it
        spatial_uns = self.adata.uns.get("_spatial_graph") or self.adata.uns.get("_spatial_graphs")
        self.spatial = bool(spatial_uns)
        # Modality names are established once we know modality count (below); set a placeholder here
        modality_names = modality_names if modality_names is not None else []
        adjacency = None

        if len(n_inputs_modalities) != len(likelihoods):
            raise ValueError("`n_inputs_modalities` and `likelihoods` must be same length")

        if sum(n_inputs_modalities) != self.summary_stats.n_vars:
            raise ValueError(
                "Sum(n_inputs_modalities) must equal adata.n_vars "
                f"(got {sum(n_inputs_modalities)} vs {self.summary_stats.n_vars})"
            )

        # Validate weight_mode
        valid_modes = {"equal", "universal", "cell"}
        if weight_mode not in valid_modes:
            raise ValueError(f"weight_mode must be one of {valid_modes}, got '{weight_mode}'")

        valid_likelihood_modes = {"none", "inverse_features", "sqrt_inverse_features"}
        if likelihood_weight_mode not in valid_likelihood_modes:
            raise ValueError(
                f"likelihood_weight_mode must be one of {valid_likelihood_modes}, "
                f"got '{likelihood_weight_mode}'"
            )

        valid_likelihood_refs = {"mean", "median", "max"}
        if likelihood_weight_ref not in valid_likelihood_refs:
            raise ValueError(
                f"likelihood_weight_ref must be one of {valid_likelihood_refs}, "
                f"got '{likelihood_weight_ref}'"
            )

        # Validate topic_feature_prior_type
        valid_prior_types = {"logistic_normal", "horseshoe"}
        if topic_feature_prior_type not in valid_prior_types:
            raise ValueError(
                f"topic_feature_prior_type must be one of {valid_prior_types}, "
                f"got '{topic_feature_prior_type}'"
            )

        # Validate that Bernoulli modalities contain only binary values {0, 1}
        for m, likelihood_m in enumerate(likelihoods):
            if likelihood_m == "bernoulli":
                # Get modality data from adata
                cursor = sum(n_inputs_modalities[:m])
                n_features = n_inputs_modalities[m]

                # Check if data is binary
                x_modality = adata.X[:, cursor:cursor + n_features]
                if hasattr(x_modality, 'toarray'):
                    x_modality = x_modality.toarray()

                unique_vals = np.unique(x_modality)
                if not np.all(np.isin(unique_vals, [0, 1])):
                    raise ValueError(
                        f"Modality {m} (likelihood='bernoulli') contains non-binary values. "
                        f"Found values: {unique_vals}. Bernoulli likelihood requires data "
                        f"to be strictly 0 or 1. For binarized ATAC-seq data, ensure peaks "
                        f"are encoded as presence/absence (not counts)."
                    )

        # Extract covariate information from registry
        # This follows scVI conventions for covariate handling
        adata_manager = self.get_anndata_manager(adata)
        n_cats_per_cov = None
        n_continuous_cov = 0

        if REGISTRY_KEYS.CAT_COVS_KEY in adata_manager.data_registry:
            cat_loc = adata_manager.data_registry[REGISTRY_KEYS.CAT_COVS_KEY]
            # Get number of categories per categorical covariate
            if hasattr(cat_loc, "n_cats_per_key"):
                n_cats_per_cov = list(cat_loc.n_cats_per_key)
            elif hasattr(adata_manager, "get_state_registry"):
                state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
                if "n_cats_per_key" in state_registry:
                    n_cats_per_cov = list(state_registry["n_cats_per_key"])

        if REGISTRY_KEYS.CONT_COVS_KEY in adata_manager.data_registry:
            cont_loc = adata_manager.data_registry[REGISTRY_KEYS.CONT_COVS_KEY]
            if hasattr(cont_loc, "columns") and cont_loc.columns is not None:
                n_continuous_cov = len(cont_loc.columns)
            elif hasattr(adata_manager, "get_state_registry"):
                state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)
                if "columns" in state_registry and state_registry["columns"] is not None:
                    n_continuous_cov = len(state_registry["columns"])

        # Detect extra encoder features from obsm registration
        n_extra_encoder_features = 0
        if "encoder_extra" in adata_manager.data_registry:
            extra_shape = adata.obsm[adata_manager.data_registry["encoder_extra"].attr_key].shape
            n_extra_encoder_features = extra_shape[1]
            logger.info(f"Extra encoder features: {n_extra_encoder_features} from obsm")

        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.n_extra_encoder_features = n_extra_encoder_features
        self.encode_covariates = encode_covariates

        if n_continuous_cov > 0:
            raise ValueError(
                "Continuous covariates are not supported. "
                "Please use categorical covariates only."
            )

        # Log covariate info if present
        if n_cats_per_cov or n_continuous_cov:
            logger.info(
                f"Covariates registered: {len(n_cats_per_cov or [])} categorical "
                f"(categories: {n_cats_per_cov}), {n_continuous_cov} continuous. "
                f"encode_covariates={encode_covariates}"
            )

        # Store modality information
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.modality_names = modality_names if modality_names else [str(i) for i in range(self.n_modalities)]
        self.modalities = self.modality_names  # alias for BaseTopicModel utilities
        self.weight_mode = weight_mode
        self.likelihood_weight_mode = likelihood_weight_mode
        self.likelihood_weight_ref = likelihood_weight_ref
        self.gcn_n_layers = gcn_n_layers
        self.gcn_conv_type = gcn_conv_type,
        self.gcn_hidden_dims = gcn_hidden_dims
        self.gcn_alpha_init = gcn_alpha_init
        self.gcn_use_learned_alpha = gcn_use_learned_alpha
        self.n_topics = n_topics
        self.topic_feature_prior_type = topic_feature_prior_type
        self.use_feature_background = use_feature_background
        self.normalize_encoder_inputs = normalize_encoder_inputs
        self.encoder_scale_factor = encoder_scale_factor
        self.entropy_weight = entropy_weight
        self.dispersion_rna = dispersion_rna
        self.learnable_dispersion = learnable_dispersion
        self.global_dispersion = global_dispersion

        # Log dispersion settings
        if learnable_dispersion:
            disp_type = "global" if global_dispersion else "per-gene"
            logger.info(
                f"Using learnable {disp_type} dispersion for Negative Binomial modalities. "
                f"Prior: HalfCauchy(1), Posterior: LogNormal."
            )

        # Log horseshoe usage
        if topic_feature_prior_type == "horseshoe":
            logger.info(
                "Using regularized horseshoe prior for topic-feature distributions. "
                "This induces sparsity for more interpretable topics."
            )

        if likelihood_weight_mode != "none":
            logger.info(
                "Using likelihood rescaling mode '%s' with reference '%s'.",
                likelihood_weight_mode,
                likelihood_weight_ref,
            )

        # Compute feature background initialization (scTM-style)
        # Only for gamma_poisson modalities when use_feature_background=True
        init_bg_mean_list = []
        if use_feature_background:
            # Get data
            X_full = self.adata.X
            if sp.issparse(X_full):
                X_full = X_full.toarray()
            X_full = np.asarray(X_full, dtype=np.float32)

            # Extract per-modality data and compute background
            start_idx = 0
            for m, (n_features, likelihood) in enumerate(zip(n_inputs_modalities, likelihoods)):
                end_idx = start_idx + n_features

                if likelihood == "gamma_poisson":
                    # Extract modality data
                    X_m = X_full[:, start_idx:end_idx]

                    # Compute mean expression per feature (scTM approach)
                    mean_expr = X_m.mean(axis=0)  # (F_m,)
                    init_bg_mean_m = np.log(mean_expr + bg_offset)  # (F_m,)

                    # Convert to tensor
                    init_bg_mean_list.append(torch.as_tensor(init_bg_mean_m, dtype=torch.float32))

                    logger.info(
                        f"Modality {m} ({likelihood}): Computed feature background "
                        f"(mean={init_bg_mean_m.mean():.3f}, std={init_bg_mean_m.std():.3f})"
                    )
                else:
                    # No background for multinomial
                    init_bg_mean_list.append(None)

                start_idx = end_idx
        else:
            # No background
            init_bg_mean_list = [None] * self.n_modalities

        if self.spatial:
            adjacency = _prepare_adjacency_tensors(spatial_uns, self.modality_names)

        # Inform user about the MoE architecture
        if self.n_modalities > 1:
            logger.info(
                f"Using {self.n_modalities} modalities with Mixture-of-Experts (MoE) architecture. "
                f"Weight mode: '{weight_mode}'. Each modality is encoded separately and mixed via "
                "weighted Gaussian combination."
            )

        # Determine max_n_obs for cell-specific weights
        max_n_obs = self.summary_stats.n_cells if weight_mode == "cell" else None

        self.module = self._module_cls(
            n_inputs_modalities=n_inputs_modalities,
            likelihoods=likelihoods,
            n_topics=n_topics,
            n_hidden=n_hidden,
            cell_topic_prior=cell_topic_prior,
            topic_feature_prior=topic_feature_prior,
            topic_feature_prior_type=topic_feature_prior_type,
            use_feature_background=use_feature_background,
            init_bg_mean=init_bg_mean_list,
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
            spatial=self.spatial,
            adjacency=adjacency,
            gcn_n_layers=gcn_n_layers,
            gcn_conv_type=gcn_conv_type,
            gcn_hidden_dims=gcn_hidden_dims,
            gcn_alpha_init=gcn_alpha_init,
            gcn_use_learned_alpha=gcn_use_learned_alpha,
            dispersion_rna=dispersion_rna,
            learnable_dispersion=learnable_dispersion,
            global_dispersion=global_dispersion,
            likelihood_weight_mode=likelihood_weight_mode,
            likelihood_weight_ref=likelihood_weight_ref,
            normalize_encoder_inputs=normalize_encoder_inputs,
            encoder_scale_factor=encoder_scale_factor,
            entropy_weight=entropy_weight,
            topic_variance_weight=topic_variance_weight,
            kl_weight=kl_weight,
            # Covariate parameters
            n_cats_per_cov=n_cats_per_cov,
            n_continuous_cov=n_continuous_cov,
            encode_covariates=encode_covariates,
            learnable_bg=learnable_bg,
            n_extra_encoder_features=n_extra_encoder_features,
        )

        # For spatial models, initialise GCN encoders with full-graph data
        if self.spatial:
            X_full = self.adata.X
            if sp.issparse(X_full):
                X_full = X_full.toarray()
            x_tensor = torch.as_tensor(np.asarray(X_full), dtype=torch.float32)
            self.module.set_full_graph_data(x_tensor)

        self.init_params_ = self._get_init_params(locals())

        # Initialize metric cache (from BaseTopicModel pattern)
        self._cached_metrics = {}

        if self.spatial:
            if isinstance(spatial_uns, dict) and "adjacency" in spatial_uns:
                keys_info = spatial_uns.get("key")
            elif isinstance(spatial_uns, dict):
                keys_info = list(spatial_uns.keys())
            else:
                keys_info = spatial_uns
            logger.info("Spatial graph provided (keys: %s); GCN encoder path enabled.", keys_info)

    # ------------------------------------------------------------------ #
    #                            anndata setup                           #
    # ------------------------------------------------------------------ #
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        spatial_key: str | None = None,
        modalities: list[str] | None = None,
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        encoder_extra_obsm_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        spatial_key
            Optional key in ``adata.obsp`` pointing to a precomputed spatial graph.
        modalities
            List of modality names (for new API). If None, defaults to ["rna"].
        layers
            Layer specifications (for new API). Can be string or dict.
        spatial_keys
            Spatial graph keys (for new API). Can be string or dict.
        categorical_covariate_keys
            Keys in ``adata.obs`` for categorical covariates (e.g., batch, sample).
            These will be embedded and used for batch effect correction.
        continuous_covariate_keys
            Keys in ``adata.obs`` for continuous covariates (e.g., age, percent_mito).
            These will be directly concatenated to the covariate representation.
        encoder_extra_obsm_key
            Optional key in ``adata.obsm`` for extra encoder input features (e.g.,
            precomputed SGC-smoothed features). These are concatenated to the encoder
            input but NOT used by the decoder/likelihood. The reconstruction target
            remains the raw counts in ``.X``.
        """
        # Handle new API with data extraction
        if modalities is not None or layers is not None or spatial_keys is not None:
            from omics_topic.data import extract_from_anndata

            # Normalize parameters
            modality_name = modalities[0] if modalities else "rna"
            layer_to_extract = layers if isinstance(layers, str) else (layers.get(modality_name) if isinstance(layers, dict) else None)
            spatial_key_to_use = spatial_keys if isinstance(spatial_keys, str) else (spatial_keys.get(modality_name) if isinstance(spatial_keys, dict) else None)

            # Extract data using utilities
            adata_processed, metadata = extract_from_anndata(
                adata, modality_name, layer_to_extract, spatial_key_to_use
            )

            # Store spatial info if present
            if metadata["spatial_info"] is not None:
                adata_processed.uns["_spatial_graph"] = metadata["spatial_info"]

            # Use processed data for registration
            adata = adata_processed
            layer = None  # Already extracted to .X
            spatial_key = None  # Already in metadata
        else:
            # Old API - spatial graph resolution
            spatial_info = _resolve_spatial_graph_from_adata(adata, spatial_key)
            if spatial_info is not None:
                adata.uns["_spatial_graph"] = spatial_info
                logger.info(
                    "Spatial graph provided via obsp['%s']; GCN encoder not implemented yet, graph will be ignored.",
                    spatial_key,
                )

        setup_args = cls._get_setup_method_args(**locals())

        # Register global cell indices for spatial/transductive models
        # This enables minibatch training with GCN encoders
        adata.obs["_indices"] = np.arange(adata.n_obs)

        fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        if encoder_extra_obsm_key is not None:
            fields.append(ObsmField("encoder_extra", encoder_extra_obsm_key))
        adata_manager = AnnDataManager(
            fields=fields,
            setup_method_args=setup_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        return adata

    @classmethod
    def setup_data(
        cls,
        data,
        modalities: list[str] | None = None,
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        table_key: str = "table",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """Universal setup method with automatic type detection.

        Detects the type of input data (AnnData, MuData, SpatialData, or dict[str, AnnData])
        and routes to the appropriate type-specific setup method.

        This method follows scvi-tools conventions, performing data registration as a side effect.
        After calling this method, instantiate the model with the same data object.

        Parameters
        ----------
        data
            Input data of any supported type:
            - AnnData: single modality data
            - MuData: multi-modal data
            - SpatialData: spatial omics data
            - dict[str, AnnData]: dictionary mapping modality names to AnnData objects
        modalities
            List of modality names to use. If None, uses all available modalities.
        layers
            Layer specifications for data extraction. Can be:
            - None: use .X for all modalities
            - str: use same layer for all modalities (e.g., "counts")
            - dict: per-modality layer specification (e.g., {"rna": "counts", "protein": "raw"})
        spatial_keys
            Spatial graph keys in .obsp. Can be:
            - None: no spatial graphs
            - str: use same key for all modalities
            - dict: per-modality spatial keys
        table_key
            For SpatialData only: key in sdata.tables to extract. Default: "table".
        **kwargs
            Additional arguments passed to scvi registration.

        Examples
        --------
        >>> # With MuData
        >>> MultimodalAmortizedLDA.setup_data(
        ...     mdata,
        ...     modalities=["rna", "protein"],
        ...     layers="counts",
        ...     spatial_keys="connectivities"
        ... )
        >>> model = MultimodalAmortizedLDA(mdata, n_topics=20)

        >>> # With AnnData
        >>> MultimodalAmortizedLDA.setup_data(
        ...     adata,
        ...     modalities=["rna"],
        ...     layers="counts"
        ... )
        >>> model = MultimodalAmortizedLDA(adata, n_topics=20)

        >>> # With dict[str, AnnData]
        >>> adata_dict = {"rna": adata_rna, "protein": adata_protein}
        >>> MultimodalAmortizedLDA.setup_data(
        ...     adata_dict,
        ...     layers={"rna": "counts", "protein": "raw"}
        ... )
        >>> # For dict, get the processed AnnData from the return value
        >>> adata_concat = MultimodalAmortizedLDA.setup_data(adata_dict, layers="counts")
        >>> model = MultimodalAmortizedLDA(adata_concat, n_topics=20)
        """
        from omics_topic.data import detect_data_type, validate_data_type

        # Validate and detect type
        validate_data_type(data)
        data_type = detect_data_type(data)

        # Route to type-specific setup method
        if data_type == "anndata":
            return cls.setup_anndata(
                data, modalities=modalities, layers=layers, spatial_keys=spatial_keys,
                categorical_covariate_keys=categorical_covariate_keys,
                continuous_covariate_keys=continuous_covariate_keys, **kwargs
            )
        elif data_type == "mudata":
            return cls.setup_mudata(
                data, modalities=modalities, layers=layers, spatial_keys=spatial_keys,
                categorical_covariate_keys=categorical_covariate_keys,
                continuous_covariate_keys=continuous_covariate_keys, **kwargs
            )
        elif data_type == "spatialdata":
            return cls.setup_spatialdata(
                data, table_key=table_key, modalities=modalities, layers=layers, spatial_keys=spatial_keys,
                categorical_covariate_keys=categorical_covariate_keys,
                continuous_covariate_keys=continuous_covariate_keys, **kwargs
            )
        elif data_type == "dict":
            return cls.setup_adata_dict(
                data, layers=layers, spatial_keys=spatial_keys,
                categorical_covariate_keys=categorical_covariate_keys,
                continuous_covariate_keys=continuous_covariate_keys, **kwargs
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        modality_order: list[str] | None = None,
        layer_dict: dict[str, str] | None = None,
        spatial_key: str | None = None,
        spatial_modality_keys: dict[str, str] | None = None,
        modalities: list[str] | None = None,
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ) -> tuple[MuData, list[str], list[int]]:
        """
        Setup MuData for multimodal AmortizedLDA.

        This method stores modality metadata in ``mdata.uns`` and prepares
        the data for the model without concatenating features.

        Parameters
        ----------
        mdata
            MuData object containing multiple modalities.
        modality_order
            Order of modalities to use. If None, uses all modalities in mdata.mod.keys().
            (Old parameter name, prefer `modalities`)
        layer_dict
            Dictionary mapping modality names to layer names to use for each modality.
            (Old parameter name, prefer `layers`)
        spatial_key
            Single obsp key applied to all modalities (if spatial_modality_keys is not provided).
            (Old parameter name, prefer `spatial_keys`)
        spatial_modality_keys
            Mapping of modality -> obsp key for modality-specific spatial graphs.
            (Old parameter name, prefer `spatial_keys`)
        modalities
            List of modality names to use (new parameter name, alias for modality_order).
        layers
            Layer specifications (new parameter name). Can be:
            - None: use .X for all modalities
            - str: use same layer for all modalities
            - dict: per-modality layer specification
        spatial_keys
            Spatial graph keys (new parameter name). Can be:
            - None: no spatial graphs
            - str: use same key for all modalities
            - dict: per-modality spatial keys
        **kwargs
            Additional arguments passed to setup_anndata.

        Returns
        -------
        mdata
            The input MuData object with metadata stored in .uns.
        modality_names
            List of modality names in the order they will be processed.
        feat_counts
            List of feature counts per modality.

        Notes
        -----
        This method uses extraction utilities for flexible layer and spatial graph handling.
        """
        from omics_topic.data import extract_from_mudata

        # Normalize parameters (merge old and new API)
        if modalities is None:
            modalities = modality_order
        if layers is None:
            layers = layer_dict
        if spatial_keys is None:
            if spatial_modality_keys is not None:
                spatial_keys = spatial_modality_keys
            elif spatial_key is not None:
                spatial_keys = spatial_key

        # Use extraction utilities
        adata_concat, metadata = extract_from_mudata(
            mdata, modalities=modalities, layers=layers, spatial_keys=spatial_keys
        )

        # Store metadata in mdata.uns (existing behavior)
        mdata.uns["_multimodal_setup"] = {
            "modality_order": metadata["modality_names"],
            "feat_counts": metadata["feature_counts"],
            "layer_dict": metadata["layer_dict"],
            "setup_method": "separate_modalities",
        }

        # Store flattened data (existing behavior)
        mdata.uns["_flattened_ann_data"] = adata_concat

        # Store spatial info
        if metadata["spatial_info"] is not None:
            if isinstance(metadata["spatial_info"], dict) and len(metadata["spatial_info"]) > 1:
                adata_concat.uns["_spatial_graphs"] = metadata["spatial_info"]
                mdata.uns["_spatial_graphs"] = metadata["spatial_info"]
                logger.info(
                    "Spatial graph(s) provided (keys: %s); GCN encoder not implemented yet, graph will be ignored.",
                    list(metadata["spatial_info"].keys()),
                )
            else:
                # Single spatial graph
                spatial_graph = metadata["spatial_info"] if not isinstance(metadata["spatial_info"], dict) else list(metadata["spatial_info"].values())[0]
                adata_concat.uns["_spatial_graph"] = spatial_graph

        # Register with scvi using this class (data already in .X after extraction)
        cls.setup_anndata(
            adata_concat, layer=None, spatial_key=None,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys, **kwargs
        )

        return mdata, metadata["modality_names"], metadata["feature_counts"]

    @classmethod
    def setup_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        modalities: list[str] | None = None,
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """Setup method for SpatialData input.

        Extracts the specified table from SpatialData and processes it for model training.

        Parameters
        ----------
        sdata
            SpatialData object containing spatial omics data.
        table_key
            Key in sdata.tables to extract. Default: "table".
        modalities
            List of modality names to use. If None, uses all available.
        layers
            Layer specifications (per-modality dict, or string for all).
        spatial_keys
            Spatial graph keys (per-modality dict, or string for all).
        **kwargs
            Additional arguments passed to scvi registration.

        Returns
        -------
        adata_concat
            The processed and registered AnnData object.
        """
        from omics_topic.data import extract_from_spatialdata

        # Extract data from SpatialData
        adata_concat, metadata = extract_from_spatialdata(
            sdata, table_key, modalities, layers, spatial_keys
        )

        # Store spatial info
        if metadata["spatial_info"] is not None:
            if isinstance(metadata["spatial_info"], dict) and len(metadata["spatial_info"]) > 1:
                adata_concat.uns["_spatial_graphs"] = metadata["spatial_info"]
                logger.info(
                    "Spatial graph(s) provided (keys: %s); GCN encoder not implemented yet, graph will be ignored.",
                    list(metadata["spatial_info"].keys()),
                )
            else:
                adata_concat.uns["_spatial_graph"] = metadata["spatial_info"]

        # Store metadata
        adata_concat.uns["_spatialdata_setup"] = {
            "table_key": table_key,
            "modality_names": metadata["modality_names"],
            "feature_counts": metadata["feature_counts"],
            "layer_dict": metadata["layer_dict"],
        }

        # Register with scvi using this class (data already in .X)
        cls.setup_anndata(
            adata_concat, layer=None, spatial_key=None,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys, **kwargs
        )

        return adata_concat

    @classmethod
    def setup_adata_dict(
        cls,
        adata_dict: dict[str, AnnData],
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """Setup method for dict[str, AnnData] input.

        Converts dictionary to concatenated AnnData and processes it for model training.

        Parameters
        ----------
        adata_dict
            Dictionary mapping modality names to AnnData objects.
        layers
            Layer specifications (per-modality dict, or string for all).
        spatial_keys
            Spatial graph keys (per-modality dict, or string for all).
        **kwargs
            Additional arguments passed to scvi registration.

        Returns
        -------
        adata_concat
            The processed and registered AnnData object.
        """
        from omics_topic.data import extract_from_adata_dict

        # Extract data
        adata_concat, metadata = extract_from_adata_dict(
            adata_dict, layers, spatial_keys
        )

        # Store spatial info
        if metadata["spatial_info"] is not None:
            if isinstance(metadata["spatial_info"], dict) and len(metadata["spatial_info"]) > 1:
                adata_concat.uns["_spatial_graphs"] = metadata["spatial_info"]
                logger.info(
                    "Spatial graph(s) provided (keys: %s); GCN encoder not implemented yet, graph will be ignored.",
                    list(metadata["spatial_info"].keys()),
                )
            else:
                adata_concat.uns["_spatial_graph"] = metadata["spatial_info"]

        # Store metadata
        adata_concat.uns["_adata_dict_setup"] = {
            "modality_names": metadata["modality_names"],
            "feature_counts": metadata["feature_counts"],
            "layer_dict": metadata["layer_dict"],
        }

        # Register with scvi using this class (data already in .X)
        cls.setup_anndata(
            adata_concat, layer=None, spatial_key=None,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys, **kwargs
        )

        return adata_concat

    # -- one-shot convenience (exactly like MultiVI) -------------
    @classmethod
    def from_mudata(
        cls,
        mdata: MuData,
        modality_order: list[str] | None = None,
        layer_dict: dict[str, str] | None = None,
        spatial_key: str | None = None,
        spatial_modality_keys: dict[str, str] | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **model_kwargs,
    ):
        """
        High-level constructor for multimodal AmortizedLDA from MuData.

        Parameters
        ----------
        mdata
            MuData object containing multiple modalities.
        modality_order
            Order of modalities to use. If None, uses all modalities in mdata.mod.keys().
        layer_dict
            Dictionary mapping modality names to layer names to use for each modality.
        spatial_key
            Single obsp key applied to all modalities (if spatial_modality_keys is not provided).
        spatial_modality_keys
            Mapping of modality -> obsp key for modality-specific spatial graphs.
        **model_kwargs
            Additional arguments passed to the model constructor.
            Common arguments include:
            - n_topics: Number of topics (default: 20)
            - n_hidden: Hidden units in encoders (default: 128)
            - weight_mode: "equal", "universal", or "cell" (default: "equal")
            - likelihood_weight_mode: "none", "inverse_features", "sqrt_inverse_features" (default: "none")
            - likelihood_weight_ref: "mean", "median", or "max" (default: "mean")
            - gcn_n_layers: Number of graph conv layers for spatial encoders (default: 1)
            - gcn_hidden_dims: List of hidden sizes per graph conv layer
            - likelihoods: List of likelihoods per modality ("multinomial", "gamma_poisson"/"nb", "bernoulli"; auto-inferred if not provided)
        """
        if modality_order is None:
            modality_order = list(mdata.mod.keys())

        mdata, modality_names, feat_counts = cls.setup_mudata(
            mdata,
            modality_order=modality_order,
            layer_dict=layer_dict,
            spatial_key=spatial_key,
            spatial_modality_keys=spatial_modality_keys,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )

        # infer default likelihoods if the caller didn't pass them
        if "likelihoods" not in model_kwargs:
            default_like = ["gamma_poisson" if mod == "rna" else "multinomial" for mod in modality_names]
            model_kwargs["likelihoods"] = default_like

        # Get the flattened AnnData for scvi compatibility
        adata_flat = mdata.uns["_flattened_ann_data"]

        return cls(
            adata_flat,
            n_inputs_modalities=feat_counts,
            modality_names=modality_names,
            **model_kwargs
        )

    # -- universal constructor with type detection -------------
    @classmethod
    def from_data(
        cls,
        data,  # AnnData | MuData | SpatialData | dict[str, AnnData]
        modalities: list[str] | None = None,
        layers: dict[str, str | None] | str | None = None,
        spatial_keys: dict[str, str] | str | None = None,
        table_key: str = "table",  # for SpatialData only
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **model_kwargs,
    ):
        """
        Convenience constructor: setup + instantiation in one call.

        This is equivalent to calling setup_data() followed by the model constructor.
        Automatically detects input type and handles all preprocessing.

        Parameters
        ----------
        data
            Input data - can be:
            - AnnData: Single or concatenated modalities
            - MuData: Multiple modalities
            - SpatialData: Spatial omics data
            - dict[str, AnnData]: Dictionary mapping modality names to AnnData
        modalities : list[str] | None
            Modalities to use. For MuData/SpatialData, subset selection.
            For AnnData, provide single modality name (default: "rna").
            For dict, uses dict keys if None.
        layers : dict[str, str | None] | str | None
            Layer specification:
            - dict: Per-modality layers {"rna": "counts", "protein": None}
            - str: Same layer for all modalities
            - None: Use .X for all
        spatial_keys : dict[str, str] | str | None
            Spatial graph keys in .obsp:
            - dict: Per-modality spatial keys
            - str: Same key for all modalities
            - None: No spatial data
        table_key : str
            For SpatialData only: which table to use (default: "table")
        **model_kwargs
            Additional arguments passed to model constructor:
            - n_topics: Number of topics (required)
            - n_hidden: Hidden units (default: 128)
            - weight_mode: "equal", "universal", or "cell" (default: "equal")
            - likelihood_weight_mode: "none", "inverse_features", "sqrt_inverse_features" (default: "none")
            - likelihood_weight_ref: "mean", "median", or "max" (default: "mean")
            - gcn_n_layers: Number of graph conv layers for spatial encoders (default: 1)
            - gcn_hidden_dims: List of hidden sizes per graph conv layer
            - likelihoods: List of likelihoods per modality ("multinomial", "gamma_poisson"/"nb", "bernoulli"; auto-inferred if not provided)

        Returns
        -------
        MultimodalAmortizedLDA
            Initialized model ready for training

        Examples
        --------
        >>> # From MuData with layer selection
        >>> model = MultimodalAmortizedLDA.from_data(
        ...     mdata,
        ...     modalities=["rna", "protein"],
        ...     layers={"rna": "counts"},
        ...     n_topics=20
        ... )

        >>> # From SpatialData
        >>> model = MultimodalAmortizedLDA.from_data(
        ...     sdata,
        ...     table_key="table",
        ...     layers="counts",
        ...     spatial_keys="spatial_connectivities",
        ...     n_topics=20
        ... )

        >>> # From dict of AnnData
        >>> model = MultimodalAmortizedLDA.from_data(
        ...     {"rna": adata_rna, "protein": adata_protein},
        ...     layers={"rna": "counts"},
        ...     n_topics=20
        ... )

        >>> # From single AnnData
        >>> model = MultimodalAmortizedLDA.from_data(
        ...     adata,
        ...     modalities=["rna"],
        ...     layers="counts",
        ...     n_topics=20
        ... )
        """
        from omics_topic.data import detect_data_type

        # Extract setup-only kwargs before passing to model constructor
        encoder_extra_obsm_key = model_kwargs.pop("encoder_extra_obsm_key", None)

        # Call setup_data to handle all preprocessing and registration
        result = cls.setup_data(
            data,
            modalities=modalities,
            layers=layers,
            spatial_keys=spatial_keys,
            table_key=table_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            encoder_extra_obsm_key=encoder_extra_obsm_key,
        )

        # Detect data type to know how to extract metadata
        data_type = detect_data_type(data)

        # Extract metadata for instantiation based on data type
        if data_type == "mudata":
            # MuData stores metadata in .uns
            modality_names = data.uns["_multimodal_setup"]["modality_order"]
            feature_counts = data.uns["_multimodal_setup"]["feat_counts"]
            adata_for_model = data.uns["_flattened_ann_data"]
        elif data_type in ["spatialdata", "dict"]:
            # These return processed AnnData with metadata in .uns
            adata_for_model = result
            setup_dict = adata_for_model.uns.get("_spatialdata_setup") or adata_for_model.uns.get("_adata_dict_setup")
            modality_names = setup_dict["modality_names"]
            feature_counts = setup_dict["feature_counts"]
        else:  # anndata
            # Single modality case
            modality_names = [modalities[0]] if modalities else ["rna"]
            adata_for_model = result if isinstance(result, AnnData) else data
            feature_counts = [adata_for_model.n_vars]

        # Auto-infer likelihoods if not provided
        if "likelihoods" not in model_kwargs:
            likelihoods = [
                "gamma_poisson" if mod == "rna" else "multinomial" for mod in modality_names
            ]
            model_kwargs["likelihoods"] = likelihoods

        # Instantiate model
        return cls(
            adata_for_model,
            n_inputs_modalities=feature_counts,
            modality_names=modality_names,
            **model_kwargs,
        )

    # ------------------------------------------------------------------ #
    #                         public helper methods                      #
    # ------------------------------------------------------------------ #
    def get_feature_topic_dist(
        self,
        modality: str | int | None = None,
        n_samples: int = 5_000,
        as_dict: bool = False,
    ) -> dict[int, pd.DataFrame] | pd.DataFrame:
        """
        Monte-Carlo estimate of E[ϕₖ,ₘ].

        Parameters
        ----------
        modality
            Modality name or index. If provided, return only that modality's
            topic-feature distribution; otherwise return all.
        n_samples
            MC samples from variational posterior.
        as_dict
            If True, return ``{m: DataFrame}`` per modality; otherwise concatenate
            along features (like original single-modality API).

        Returns
        -------
        • dict of DataFrames (default) – index = feature names, columns = topics
        • or a single concatenated DataFrame if ``as_dict=False``.
        """
        self._check_if_trained(warn=False)
        tbf_dict = self.module.topic_by_feature(n_samples)

        dfs = {}
        cursor = 0
        for m, tbf in tbf_dict.items():
            features = self.adata.var_names[cursor : cursor + tbf.shape[1]]
            cursor += tbf.shape[1]
            dfs[m] = pd.DataFrame(data=tbf.T, index=features, columns=[f"topic_{k}" for k in range(tbf.shape[0])])

        if as_dict:
            return dfs
        if modality is not None:
            # Allow modality name or index
            if isinstance(modality, str):
                if modality not in self.modality_names:
                    raise ValueError(f"Unknown modality '{modality}'. Valid modalities: {self.modality_names}")
                mod_idx = self.modality_names.index(modality)
            else:
                mod_idx = int(modality)
            if mod_idx not in dfs:
                raise KeyError(f"Topic-feature distribution missing for modality index {mod_idx}.")
            return dfs[mod_idx]
        # concat to mimic original signature
        return pd.concat(dfs.values(), axis=0)

    # ------------------------------------------------------------------ #
    def get_topic_diversity(self, modality: str | None = None) -> float:
        """
        Compute topic diversity (average pairwise cosine distance) per modality or overall.
        """
        if modality is None:
            cache_key = "topic_diversity_all_modalities"
            if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
                return self._cached_metrics[cache_key]

            diversities = [self.get_topic_diversity(mod) for mod in self.modality_names]
            result = float(np.mean(diversities))
            if hasattr(self, "_cached_metrics"):
                self._cached_metrics[cache_key] = result
            return result

        cache_key = f"topic_diversity_modality={modality}"
        if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        if modality not in self.modality_names:
            raise ValueError(f"Unknown modality '{modality}'. Valid modalities: {self.modality_names}")
        mod_idx = self.modality_names.index(modality)

        phi_dict = self.get_feature_topic_dist(as_dict=True)
        if mod_idx not in phi_dict:
            raise KeyError(f"Topic-feature distribution missing for modality index {mod_idx}.")
        phi_df = phi_dict[mod_idx]

        phi = phi_df.values.T if isinstance(phi_df, pd.DataFrame) else np.asarray(phi_df).T
        phi_norm = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-12)
        cosine_sim = phi_norm @ phi_norm.T

        K = phi.shape[0]
        upper_tri_indices = np.triu_indices(K, k=1)
        similarities = cosine_sim[upper_tri_indices]
        result = float(1 - similarities.mean())

        if hasattr(self, "_cached_metrics"):
            self._cached_metrics[cache_key] = result
        return result

    # ------------------------------------------------------------------ #
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        n_samples: int = 5_000,
    ) -> pd.DataFrame:
        """
        Infer θₙ for all cells (or subset).

        Returns
        -------
        DataFrame (cells × topics) with softmax-normalized expectations.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self.module.eval()
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        thetas = []
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            enc_extra = tensors.get("encoder_extra", None)
            thetas.append(self.module.get_topic_distribution(x, n_samples, encoder_extra=enc_extra))
        theta = torch.cat(thetas).cpu().numpy()

        return pd.DataFrame(theta, index=adata.obs_names, columns=[f"topic_{k}" for k in range(theta.shape[1])])

    def get_cell_topic_dist(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        n_samples: int = 5_000,
    ) -> np.ndarray:
        """
        Get the cell-topic matrix Θ (C × K).

        Parameters
        ----------
        adata
            AnnData object to use (default: self.adata).
        indices
            Subset of cells to use.
        batch_size
            Batch size for inference.
        n_samples
            Number of samples for Monte Carlo estimation.

        Returns
        -------
        Θ : np.ndarray
            Cell-topic matrix, where C is the number of cells and K is the number of topics.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self.module.eval()
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        thetas = []
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            enc_extra = tensors.get("encoder_extra", None)
            thetas.append(self.module.get_topic_distribution(x, n_samples, encoder_extra=enc_extra))
        return torch.cat(thetas).cpu().numpy()

    # ------------------------------------------------------------------ #
    def _batch_library_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-modality library sizes for a *mini-batch* ``x``.

        Assumes modalities are concatenated in the same order as during initialisation.
        """
        libs = []
        cursor = 0
        for F_m in self.module.n_inputs_modalities:
            lib_m = x[:, cursor : cursor + F_m].sum(dim=1)
            libs.append(torch.clamp(lib_m, min=0.0))  # guard against negative/centered inputs
            cursor += F_m
        return torch.stack(libs, dim=1)  # (B, M)

    # ------------------------------------------------------------------ #
    def get_elbo(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Average ELBO across cells (higher is better)."""
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self.module.eval()
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        n_obs = len(dl.indices)

        elbos = []
        for tensors in dl:
            args, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs["n_obs"] = n_obs
            elbos.append(Trace_ELBO().loss(self.module.model, self.module.guide, *args, **kwargs))
        return float(np.mean(elbos))

    # ------------------------------------------------------------------ #
    def get_perplexity(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """exp( – ELBO / total counts ) – lower is better."""
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        total_counts = sum(tensors[REGISTRY_KEYS.X_KEY].sum().item() for tensors in dl)

        return float(np.exp(-self.get_elbo(adata, indices, batch_size) / total_counts))

    # ------------------------------------------------------------------ #
    def get_entropy(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        normalised: bool = True,
    ) -> float:
        """
        Compute mean entropy of cell-topic distributions.

        Higher entropy means topics are more evenly distributed across cells.

        Parameters
        ----------
        adata
            AnnData object to use (default: self.adata).
        indices
            Subset of cells to use.
        batch_size
            Batch size for inference.
        normalised
            Whether to normalize cell-topic distributions before computing entropy.

        Returns
        -------
        float
            Mean entropy across cells
        """
        cache_key = f"entropy_normalised={normalised}"
        if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        # Get cell-topic distributions
        theta = self.get_cell_topic_dist(adata=adata, indices=indices, batch_size=batch_size)

        if normalised:
            # Normalize to ensure rows sum to 1
            theta = theta / (theta.sum(axis=1, keepdims=True) + 1e-12)

        # Compute entropy per cell: -Σ_k θ_ck * log(θ_ck)
        entropy_per_cell = -(theta * np.log(np.clip(theta, 1e-8, None))).sum(axis=1)

        # Return mean entropy across cells
        result = float(entropy_per_cell.mean())

        # Cache result
        if hasattr(self, "_cached_metrics"):
            self._cached_metrics[cache_key] = result

        return result

    # ------------------------------------------------------------------ #
    def get_likelihood_per_modality(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """
        Compute log-likelihood for each modality separately.

        Higher is better.

        Parameters
        ----------
        adata
            AnnData object to use (default: self.adata).
        indices
            Subset of cells to use.
        batch_size
            Batch size for inference.

        Returns
        -------
        dict[str, float]
            Dictionary mapping modality names to log-likelihood values
        """
        cache_key = "likelihood_per_modality"
        if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        # Compute per-modality log-likelihood
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self.module.eval()
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # Accumulate per-modality log-probs across batches
        per_mod_logprob_total = {m: 0.0 for m in range(self.n_modalities)}

        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            libs = self._batch_library_tensor(x)
            batch_logprobs = self.module.get_per_modality_log_prob(x, libs, len(dl.indices))

            for m, logprob in batch_logprobs.items():
                per_mod_logprob_total[m] += logprob

        # Convert modality indices to names
        result = {
            self.modality_names[m]: logprob
            for m, logprob in per_mod_logprob_total.items()
        }

        # Cache result
        if hasattr(self, "_cached_metrics"):
            self._cached_metrics[cache_key] = result

        return result

    # ------------------------------------------------------------------ #
    def get_perplexity_per_modality(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """
        Compute perplexity for each modality separately.

        Lower is better. Perplexity = exp(-log_likelihood / N_tokens)

        Parameters
        ----------
        adata
            AnnData object to use (default: self.adata).
        indices
            Subset of cells to use.
        batch_size
            Batch size for inference.

        Returns
        -------
        dict[str, float]
            Dictionary mapping modality names to perplexity values
        """
        cache_key = "perplexity_per_modality"
        if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        # Get log-likelihoods
        log_liks = self.get_likelihood_per_modality(adata, indices, batch_size)

        # Compute total counts per modality
        adata = self._validate_anndata(adata)
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        per_mod_counts = {m: 0 for m in range(self.n_modalities)}
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            cursor = 0
            for m, F_m in enumerate(self.n_inputs_modalities):
                per_mod_counts[m] += x[:, cursor:cursor + F_m].sum().item()
                cursor += F_m

        # Compute perplexities
        result = {
            mod_name: float(np.exp(-log_lik / per_mod_counts[m]))
            for m, (mod_name, log_lik) in enumerate(log_liks.items())
        }

        # Cache result
        if hasattr(self, "_cached_metrics"):
            self._cached_metrics[cache_key] = result

        return result

    # ------------------------------------------------------------------ #
    def get_modality_weights(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        return_format: str = "dataframe",
    ) -> pd.DataFrame | dict[str, np.ndarray]:
        """
        Get normalized mixing weights showing how much each modality contributes to topic assignments.

        The mixing weights control how much each modality's encoder output contributes to the
        final cell-topic distribution (θ) in the Mixture-of-Experts architecture.

        Returns weights in range [0, 1] that sum to 1 per cell (or globally for "universal" mode).
        Higher weight = model relies more on that modality for inferring topics.

        Parameters
        ----------
        adata : AnnData, optional
            AnnData object to compute weights for. If None, uses the registered dataset.
        indices : Sequence[int], optional
            Indices of cells to include. If None, uses all cells.
        batch_size : int, optional
            Batch size for inference.
        return_format : str
            Output format: "dataframe" returns pd.DataFrame (cells × modalities),
            "dict" returns dict mapping modality names to 1D arrays.

        Returns
        -------
        pd.DataFrame or dict[str, np.ndarray]
            Normalized mixing weights for each cell and modality.
            For "universal" mode: returns single row with global weights.
            For "equal" mode: returns uniform weights (1/n_modalities).
            For "cell" mode: returns per-cell learned weights.
        """
        cache_key = "modality_weights"
        if hasattr(self, "_cached_metrics") and cache_key in self._cached_metrics:
            cached = self._cached_metrics[cache_key]
            if return_format == "dataframe":
                return cached if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)
            else:
                return cached if isinstance(cached, dict) else cached.to_dict("list")

        # Validate inputs
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self.module.eval()

        # Handle different weight modes
        if self.weight_mode == "equal":
            # Equal weights for all modalities: 1/M
            n_cells = adata.n_obs if indices is None else len(indices)
            weights_array = np.ones((n_cells, self.n_modalities)) / self.n_modalities

        elif self.weight_mode == "universal":
            # Single learned weight per modality (same across all cells)
            # Extract raw weights and compute softmax
            raw_weights = self.module.guide.mod_w.detach().cpu().numpy()

            # Create dummy mask (all modalities present)
            mask = np.ones(self.n_modalities)

            # Apply softmax
            weights_normalized = self._masked_softmax_np(raw_weights, mask)

            # Broadcast to all cells
            n_cells = adata.n_obs if indices is None else len(indices)
            weights_array = np.tile(weights_normalized, (n_cells, 1))

        else:  # weight_mode == "cell"
            # Per-cell learned weights - need to extract from data loader
            dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

            all_weights = []
            n_cells_processed = 0

            for tensors in dl:
                x = tensors[REGISTRY_KEYS.X_KEY]
                B = x.shape[0]

                # Split by modality and compute masks
                xs = torch.split(x, self.n_inputs_modalities, dim=1)
                masks = torch.stack([(x_m.sum(1) > 0).float() for x_m in xs])  # (M, B)

                # Get batch indices for cell-specific weights
                batch_indices = torch.arange(n_cells_processed, n_cells_processed + B, device=x.device)

                # Extract raw weights for this batch
                raw_w = self.module.guide.mod_w[batch_indices, :].T  # (M, B)

                # Apply masked softmax
                from omics_topic.module._amortizedLDA import masked_softmax
                masks = masks.to(raw_w.device, non_blocking=True)
                weights_batch = masked_softmax(raw_w, masks, dim=0)  # (M, B)

                # Transpose to (B, M) and store
                all_weights.append(weights_batch.T.detach().cpu().numpy())

                n_cells_processed += B

            weights_array = np.vstack(all_weights)

        # Convert to desired output format
        if return_format == "dataframe":
            cell_indices = indices if indices is not None else np.arange(adata.n_obs)
            result = pd.DataFrame(
                weights_array,
                index=cell_indices,
                columns=self.modality_names,
            )
        else:  # dict format
            result = {
                mod_name: weights_array[:, i]
                for i, mod_name in enumerate(self.modality_names)
            }

        # Cache result (store DataFrame for consistency)
        if hasattr(self, "_cached_metrics"):
            if isinstance(result, pd.DataFrame):
                self._cached_metrics[cache_key] = result
            else:
                self._cached_metrics[cache_key] = pd.DataFrame(result)

        return result

    def get_likelihood_weights(self, return_format: str = "dataframe") -> pd.DataFrame | dict[str, float]:
        """
        Return per-modality likelihood scaling weights used in the generative model.

        Parameters
        ----------
        return_format : str
            "dataframe" returns a single-row DataFrame (modalities as columns),
            "dict" returns a mapping of modality name -> weight.

        Returns
        -------
        pd.DataFrame or dict[str, float]
            Likelihood weights for each modality.
        """
        weights = self.module.model.likelihood_weights.detach().cpu().numpy()
        if return_format == "dict":
            return {name: float(weights[i]) for i, name in enumerate(self.modality_names)}
        if return_format != "dataframe":
            raise ValueError("return_format must be 'dataframe' or 'dict'.")
        return pd.DataFrame([weights], columns=self.modality_names)

    def get_entropy_weight(self) -> float:
        """
        Get the entropy regularization weight.

        Returns
        -------
        float
            Current entropy_weight value used for regularization
        """
        return self.entropy_weight

    def get_last_entropy(self) -> float | None:
        """
        Get the mean entropy from the last forward pass through the model.

        Returns
        -------
        float | None
            Mean cell-topic entropy from last forward pass, or None if not available
        """
        return self.module.get_last_entropy()

    def get_cell_entropy(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute per-cell entropy of cell-topic distributions.

        Parameters
        ----------
        adata
            AnnData object with data. If None, uses the training data.
        indices
            Indices of cells to compute entropy for. If None, uses all cells.
        batch_size
            Batch size for computation. If None, processes all cells at once.
        n_samples
            Number of posterior samples for Monte Carlo estimation (default: 100)

        Returns
        -------
        np.ndarray
            Per-cell entropy values, shape (n_cells,)
            H(θ_n) = -Σ_k θ_n,k * log(θ_n,k) for each cell n
        """
        adata = self._validate_anndata(adata)
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        all_entropies = []
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            libs = self._batch_library_tensor(x)
            entropy_batch = self.module.get_cell_entropy(x, libs, n_samples=n_samples)
            all_entropies.append(entropy_batch.detach().cpu().numpy())

        return np.concatenate(all_entropies)

    def get_topic_variance_weight(self) -> float:
        """
        Get the topic variance regularization weight.

        Returns
        -------
        float
            Topic variance weight used during training
        """
        return self.module.topic_variance_weight

    def get_last_topic_variance(self) -> float | None:
        """
        Get the last computed mean topic variance from training.

        Returns
        -------
        float | None
            Mean topic variance from the last forward pass, or None if not available
        """
        return self.module.get_last_topic_variance()

    def get_topic_variance(
        self,
        adata: AnnData | None = None,
        indices: _Seq[int] | None = None,
        batch_size: int | None = None,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute per-topic variance of topic usage across cells.

        Parameters
        ----------
        adata
            AnnData object with data. If None, uses the training data.
        indices
            Indices of cells to compute variance for. If None, uses all cells.
        batch_size
            Batch size for computation. If None, processes all cells at once.
        n_samples
            Number of posterior samples for Monte Carlo estimation (default: 100)

        Returns
        -------
        np.ndarray
            Per-topic variance values, shape (n_topics,)
            Var(θ_:,k) = variance of topic k usage across all cells
        """
        adata = self._validate_anndata(adata)

        # Collect all cell-topic distributions first
        all_theta = []
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            libs = self._batch_library_tensor(x)
            theta_batch = self.module.get_topic_distribution(x, libs, n_samples=n_samples)
            all_theta.append(theta_batch.detach().cpu())

        # Concatenate all batches and compute variance across cells
        all_theta = torch.cat(all_theta, dim=0)  # (n_cells, n_topics)
        topic_variance = all_theta.var(dim=0)  # (n_topics,)

        return topic_variance.numpy()

    def get_learned_dispersion(
        self,
        modality: str | int | None = None,
        n_samples: int = 1000,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Get the learned or fixed dispersion parameters.

        Parameters
        ----------
        modality : str | int | None, optional
            Modality name or index. If None, returns dispersion for all
            NB modalities as a dict.
        n_samples : int, optional
            Number of Monte Carlo samples for learned dispersion. Default: 1000.

        Returns
        -------
        dict[str, np.ndarray] | np.ndarray
            If modality is None: dict mapping modality names to dispersion arrays
            If modality specified: dispersion array for that modality

        Notes
        -----
        - If learnable_dispersion=False, returns the fixed dispersion value
        - If learnable_dispersion=True and global_dispersion=True: returns (1,) array
        - If learnable_dispersion=True and global_dispersion=False: returns (n_features,) array
        """
        self._check_if_trained(warn=False)

        if modality is not None:
            # Single modality
            if isinstance(modality, str):
                if modality not in self.modality_names:
                    raise ValueError(f"Unknown modality '{modality}'")
                mod_idx = self.modality_names.index(modality)
            else:
                mod_idx = int(modality)

            disp = self.module.get_learned_dispersion(mod_idx, n_samples)
            return disp.cpu().numpy()

        # All modalities
        result = {}
        for m, (mod_name, likelihood) in enumerate(zip(self.modality_names, self.likelihoods)):
            if likelihood in {"gamma_poisson", "nb"}:
                disp = self.module.get_learned_dispersion(m, n_samples)
                result[mod_name] = disp.cpu().numpy()

        return result

    @staticmethod
    def _masked_softmax_np(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """NumPy implementation of masked softmax."""
        weights = weights.copy()
        weights[mask == 0] = -1e9
        exp_w = np.exp(weights - weights.max())
        return exp_w / exp_w.sum()

    # ------------------------------------------------------------------ #
    # Training plan hook
    # ------------------------------------------------------------------ #
    def _create_training_plan(self, **kwargs):
        """
        Use custom training plan that logs validation ELBO when a val split exists.

        Accepts ``kl_warmup_fraction`` (default 0.25) to control KL annealing.
        Set to 0 to disable KL warmup (recommended for horseshoe prior).
        """
        return MultimodalLDAPyroTrainingPlan(self.module, **kwargs)

    # ------------------------------------------------------------------ #
    # Ensure validation runs when requested
    # ------------------------------------------------------------------ #
    def train(self, *args, validation_size=None, **kwargs):  # type: ignore[override]
        """
        Override to default to running validation when a split is requested.

        scvi's Trainer defaults to `check_val_every_n_epoch = sys.maxsize` unless
        early stopping or checkpointing is enabled, which effectively disables
        the validation loop. Here we set it to 1 when a validation set is present
        so that `elbo_val` is logged every epoch.
        """
        if "check_val_every_n_epoch" not in kwargs:
            if validation_size is None or validation_size > 0:
                kwargs["check_val_every_n_epoch"] = 1
        return super().train(*args, validation_size=validation_size, **kwargs)


def mudata_to_concat_adata(
    mdata: MuData,
    modality_order: list[str] | None = None,
) -> tuple[AnnData, list[int]]:
    """Flatten a `MuData` into a single `AnnData`.

    The resulting `.X` has shape ``(n_cells, Σ features_of_each_modality)``.

    Returns
    -------
    adata_flat
        The concatenated `AnnData`.
    feat_counts
        One integer per modality (same order) giving its feature count.
    """
    if modality_order is None:
        modality_order = list(mdata.mod.keys())

    matrices = []
    feat_counts = []
    var_names = []

    n_cells_ref = mdata.n_obs

    for mod in modality_order:
        X = mdata.mod[mod].X

        # convert sparse → csr, dense stays dense
        if sp.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        # ensure 2-D: (n,)  ->  (n,1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Sanity-check number of cells
        if X.shape[0] != n_cells_ref:
            raise ValueError(f"Modality {mod!r} has {X.shape[0]} cells, but MuData has {n_cells_ref}.")

        matrices.append(X)
        feat_counts.append(X.shape[1])
        var_names.extend(mdata.mod[mod].var_names)

    # --------------------------------------------------------------
    # concatenate (sparse if any input was sparse, else dense)
    # --------------------------------------------------------------
    if any(sp.issparse(M) for M in matrices):
        X_concat = sp.hstack(matrices, format="csr")
    else:
        X_concat = np.hstack(matrices)

    adata = AnnData(X_concat, obs=mdata.obs.copy())
    adata.var_names = var_names

    return adata, feat_counts
