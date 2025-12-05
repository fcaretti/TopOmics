# multimodal_amortized_lda.py
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyro
import scipy.sparse as sp
import torch
from anndata import AnnData
from mudata import MuData
from scvi._constants import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
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
        modality_names: list[str] | None = None,
        weight_mode: str = "equal",
    ):
        """
        Initialize MultimodalAmortizedLDA with Mixture-of-Experts (MoE) architecture.

        Parameters
        ----------
        adata
            AnnData with concatenated features (for scvi compatibility).
        n_inputs_modalities
            List of feature counts per modality.
        likelihoods
            List of likelihood strings per modality ("multinomial" or "gamma_poisson").
        n_topics
            Number of topics.
        n_hidden
            Hidden units in encoder networks.
        cell_topic_prior
            Dirichlet concentration for θₙ.
        topic_feature_prior
            Dirichlet concentration for ϕₖ,ₘ.
        modality_names
            Optional list of modality names (e.g., ["rna", "protein"]). If None, uses indices.
        weight_mode
            How to weight modality-specific representations when mixing:
            - "equal": All modalities weighted equally (default, simplest)
            - "universal": Learn a single weight per modality across all cells
            - "cell": Learn per-cell, per-modality weights (most flexible)

        Notes
        -----
        The model uses Mixture-of-Experts architecture where each modality is encoded
        separately and then mixed via weighted Gaussian combination before sampling
        the shared cell-topic distribution θₙ.
        """
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

        # Store modality information
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.modality_names = modality_names if modality_names else [str(i) for i in range(self.n_modalities)]
        self.weight_mode = weight_mode

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
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
            spatial=self.spatial,
            adjacency=adjacency,
        )

        # For spatial models, initialise GCN encoders with full-graph data
        if self.spatial:
            X_full = self.adata.X
            if sp.issparse(X_full):
                X_full = X_full.toarray()
            x_tensor = torch.as_tensor(np.asarray(X_full), dtype=torch.float32)
            self.module.set_full_graph_data(x_tensor)

        self.init_params_ = self._get_init_params(locals())

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
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        spatial_key
            Optional key in ``adata.obsp`` pointing to a precomputed spatial graph.
        """
        spatial_info = _resolve_spatial_graph_from_adata(adata, spatial_key)
        if spatial_info is not None:
            adata.uns["_spatial_graph"] = spatial_info
            logger.info(
                "Spatial graph provided via obsp['%s']; GCN encoder not implemented yet, graph will be ignored.",
                spatial_key,
            )

        setup_args = cls._get_setup_method_args(**locals())
        adata_manager = AnnDataManager(
            fields=[LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True)],
            setup_method_args=setup_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        modality_order: list[str] | None = None,
        layer_dict: dict[str, str] | None = None,
        spatial_key: str | None = None,
        spatial_modality_keys: dict[str, str] | None = None,
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
        layer_dict
            Dictionary mapping modality names to layer names to use for each modality.
        spatial_key
            Single obsp key applied to all modalities (if spatial_modality_keys is not provided).
        spatial_modality_keys
            Mapping of modality -> obsp key for modality-specific spatial graphs.
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
        This is a new implementation that does NOT concatenate features.
        Modality-specific data is kept separate for MoE/PoE architecture.
        """
        if modality_order is None:
            modality_order = list(mdata.mod.keys())

        feat_counts = []
        modality_names = []
        spatial_graphs: dict[str, dict[str, object]] = {}

        # Validate modalities and collect feature counts
        n_cells_ref = mdata.n_obs
        for mod in modality_order:
            if mod not in mdata.mod:
                raise ValueError(f"Modality '{mod}' not found in MuData. Available: {list(mdata.mod.keys())}")

            adata_mod = mdata.mod[mod]
            if adata_mod.n_obs != n_cells_ref:
                raise ValueError(
                    f"Modality '{mod}' has {adata_mod.n_obs} cells, "
                    f"but MuData has {n_cells_ref} cells. All modalities must be aligned."
                )

            feat_counts.append(adata_mod.n_vars)
            modality_names.append(mod)

            key_for_mod = (spatial_modality_keys or {}).get(mod, spatial_key)
            spatial_info = _resolve_spatial_graph_from_adata(adata_mod, key_for_mod)
            if spatial_info is not None:
                spatial_graphs[mod] = spatial_info

        # Store metadata in mdata.uns for later retrieval
        mdata.uns["_multimodal_setup"] = {
            "modality_order": modality_names,
            "feat_counts": feat_counts,
            "layer_dict": layer_dict or {},
            "setup_method": "separate_modalities",  # Flag for new implementation
        }
        if spatial_graphs:
            mdata.uns["_spatial_graphs"] = spatial_graphs

        # For now, we still need to create a concatenated AnnData for scvi registration
        # but we'll store the modality information for the module to use
        adata_flat, _ = mudata_to_concat_adata(mdata, modality_order)
        mdata.uns["_flattened_ann_data"] = adata_flat
        if spatial_graphs:
            adata_flat.uns["_spatial_graphs"] = spatial_graphs
            logger.info(
                "Spatial graph(s) provided (keys: %s); GCN encoder not implemented yet, graph will be ignored.",
                list(spatial_graphs.keys()),
            )

        # Register with scvi
        cls.setup_anndata(
            adata_flat,
            layer=layer_dict.get("rna") if layer_dict else None,
            spatial_key=None,
            **kwargs,
        )

        return mdata, modality_names, feat_counts

    # -- one-shot convenience (exactly like MultiVI) -------------
    @classmethod
    def from_mudata(
        cls,
        mdata: MuData,
        modality_order: list[str] | None = None,
        layer_dict: dict[str, str] | None = None,
        spatial_key: str | None = None,
        spatial_modality_keys: dict[str, str] | None = None,
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
            - likelihoods: List of likelihoods per modality (auto-inferred if not provided)

        Returns
        -------
        model
            Instance of MultimodalAmortizedLDA.

        Examples
        --------
        >>> # Equal weighting (default MoE)
        >>> model = MultimodalAmortizedLDA.from_mudata(
        ...     mdata,
        ...     modality_order=["rna", "protein"],
        ...     n_topics=10,
        ...     n_hidden=128
        ... )
        >>>
        >>> # With learned universal weights
        >>> model = MultimodalAmortizedLDA.from_mudata(
        ...     mdata,
        ...     modality_order=["rna", "protein"],
        ...     n_topics=10,
        ...     weight_mode="universal"
        ... )
        """
        if modality_order is None:
            modality_order = list(mdata.mod.keys())

        mdata, modality_names, feat_counts = cls.setup_mudata(
            mdata,
            modality_order=modality_order,
            layer_dict=layer_dict,
            spatial_key=spatial_key,
            spatial_modality_keys=spatial_modality_keys,
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

    # ------------------------------------------------------------------ #
    #                         public helper methods                      #
    # ------------------------------------------------------------------ #
    def get_feature_topic_dist(
        self, n_samples: int = 5_000, as_dict: bool = False
    ) -> dict[int, pd.DataFrame] | pd.DataFrame:
        """
        Monte-Carlo estimate of E[ϕₖ,ₘ].

        Parameters
        ----------
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
        # concat to mimic original signature
        return pd.concat(dfs.values(), axis=0)

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
            thetas.append(self.module.get_topic_distribution(x, n_samples))
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
            thetas.append(self.module.get_topic_distribution(x, n_samples))
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
            libs.append(x[:, cursor : cursor + F_m].sum(dim=1))
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

        elbos = []
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            libs = self._batch_library_tensor(x)
            elbos.append(self.module.get_elbo(x, libs, len(dl.indices)))
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
    # Training plan hook
    # ------------------------------------------------------------------ #
    def _create_training_plan(self, **kwargs):
        """
        Use custom training plan that logs validation ELBO when a val split exists.
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
