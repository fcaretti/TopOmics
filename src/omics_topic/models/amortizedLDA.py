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
    **Multimodal Amortized LDA**

    Extends :class:`scvi.model.AmortizedLDA` to *M* modalities with
    modality-specific likelihoods (``"multinomial"`` or ``"gamma_poisson"``).

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
    ):
        pyro.clear_param_store()
        super().__init__(adata)

        if len(n_inputs_modalities) != len(likelihoods):
            raise ValueError("`n_inputs_modalities` and `likelihoods` must be same length")

        if sum(n_inputs_modalities) != self.summary_stats.n_vars:
            raise ValueError(
                "Sum(n_inputs_modalities) must equal adata.n_vars "
                f"(got {sum(n_inputs_modalities)} vs {self.summary_stats.n_vars})"
            )

        self.module = self._module_cls(
            n_inputs_modalities=n_inputs_modalities,
            likelihoods=likelihoods,
            n_topics=n_topics,
            n_hidden=n_hidden,
            cell_topic_prior=cell_topic_prior,
            topic_feature_prior=topic_feature_prior,
        )
        self.n_modalities = len(n_inputs_modalities)
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
    ) -> tuple[AnnData, list[int]]:
        """
        Flatten *mdata* into a single AnnData.

        Register it with
        ``cls.setup_anndata`` **and return it** so the user can pass it
        straight into the constructor.

        Stores helper metadata in
        ``mdata.uns`` (feature counts, flattened AnnData reference).
        """
        adata_flat, feat_counts = mudata_to_concat_adata(mdata, modality_order)
        mdata.uns["_feat_counts"] = feat_counts
        mdata.uns["_flattened_ann_data"] = adata_flat

        # the usual registration — you may pass `layer_dict` if you want
        cls.setup_anndata(adata_flat, layer=layer_dict.get("rna") if layer_dict else None, **kwargs)
        return adata_flat, feat_counts

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
        High-level constructor::

            model = MultimodalAmortizedLDA.from_mudata(
                        mdata,
                        modality_order=["rna", "protein", "atac"],
                        layer_dict={"rna": "counts", "protein": "counts"},
                        n_topics=10,
                        ...
                    )
        """
        if modality_order is None:
            modality_order = list(mdata.mod.keys())

        adata_flat, feat_counts = cls.setup_mudata(
            mdata,
            modality_order=modality_order,
            layer_dict=layer_dict,
        )

        # infer default likelihoods if the caller didn’t pass them
        if "likelihoods" not in model_kwargs:
            default_like = ["gamma_poisson" if mod == "rna" else "multinomial" for mod in modality_order]
            model_kwargs["likelihoods"] = default_like

        return cls(adata_flat, n_inputs_modalities=feat_counts, **model_kwargs)

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
