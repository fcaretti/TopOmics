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

from .base_model import BaseTopicModel

if TYPE_CHECKING:
    from collections.abc import Sequence as _Seq

logger = logging.getLogger(__name__)


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
        self.modality_names = modality_names if modality_names is not None else [str(i) for i in range(self.n_modalities)]
        self.weight_mode = weight_mode

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
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------ #
    #                            anndata setup                           #
    # ------------------------------------------------------------------ #
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        """
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

        # Store metadata in mdata.uns for later retrieval
        mdata.uns["_multimodal_setup"] = {
            "modality_order": modality_names,
            "feat_counts": feat_counts,
            "layer_dict": layer_dict or {},
            "setup_method": "separate_modalities",  # Flag for new implementation
        }

        # For now, we still need to create a concatenated AnnData for scvi registration
        # but we'll store the modality information for the module to use
        adata_flat, _ = mudata_to_concat_adata(mdata, modality_order)
        mdata.uns["_flattened_ann_data"] = adata_flat

        # Register with scvi
        cls.setup_anndata(adata_flat, layer=layer_dict.get("rna") if layer_dict else None, **kwargs)

        return mdata, modality_names, feat_counts

    # -- one-shot convenience (exactly like MultiVI) -------------
    @classmethod
    def from_mudata(
        cls,
        mdata: MuData,
        modality_order: list[str] | None = None,
        layer_dict: dict[str, str] | None = None,
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
