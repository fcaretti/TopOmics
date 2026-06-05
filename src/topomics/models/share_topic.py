"""ShareTopic_LDA_Multi – SHARE-Topic Gibbs sampler in the topomics framework.

Combines RNA (Gamma-Poisson), ATAC (sparse Bernoulli), and optionally Protein
(NB or Multinomial) modalities with a shared Dirichlet cell-topic prior and
SHARE-Topic-style sparse ATAC batching.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch import Tensor

from topomics.module._share_topic import (
    anndata_to_sparse_coords,
    compute_log_likelihood,
    create_cell_batches,
    create_region_batches,
    initialize_parameters,
    run_gibbs_sampler,
    smart_initialize,
    to_numpy,
)

from .base_model import BaseTopicModel

MuDataType = "muon.MuData"

__all__ = ["ShareTopic_LDA_Multi"]


def _extract_X(X):
    """Unwrap AnnData to its .X matrix if needed."""
    if hasattr(X, "X"):  # AnnData
        return X.X
    return X


def _to_dense_float(X, device: str) -> Tensor:
    """Convert AnnData / .X / sparse / ndarray to dense float tensor on *device*."""
    X = _extract_X(X)
    if isinstance(X, Tensor):
        return X.float().to(device, non_blocking=True)
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return torch.from_numpy(X).to(device, non_blocking=True)


class ShareTopic_LDA_Multi(BaseTopicModel):
    """Multimodal LDA with SHARE-Topic sparse ATAC Gibbs sampler.

    Parameters
    ----------
    mdata
        Input data: MuData, ``dict[str, AnnData]``, ``list[AnnData]``, or
        single ``AnnData``.
    modalities
        Modality names when *mdata* is a list or single AnnData.
    n_topics
        Number of latent topics.
    alpha
        Dirichlet prior concentration for cell-topic proportions.
    gamma, tau
        Gamma prior shape/rate for RNA rate parameters.  ``None`` = empirical
        Bayes (estimated from data moments).
    beta
        Dirichlet prior concentration for chromatin region proportions.
    protein_likelihood
        ``"nb"`` (Gamma-Poisson / Negative Binomial) or ``"multinomial"``
        (Dirichlet-Multinomial) for the protein modality.
    protein_gamma, protein_tau
        Gamma prior shape/rate for protein NB rates.  ``None`` = empirical
        Bayes.
    protein_beta
        Dirichlet prior concentration for protein multinomial.
    device
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    smart_init
        Seed parameters via NMF instead of sampling from the prior.
    smart_init_mod
        Modality used for NMF initialisation (default ``"rna"``).
    """

    def __init__(
        self,
        mdata: MuDataType | dict[str, AnnData] | list[AnnData] | AnnData,
        *,
        modalities: list[str] | None = None,
        n_topics: int = 20,
        alpha: float | list[float] = 0.1,
        # RNA priors
        gamma: float | None = None,
        tau: float | None = None,
        # Chromatin prior
        beta: float = 0.1,
        # Protein likelihood
        protein_likelihood: str = "nb",
        protein_gamma: float | None = None,
        protein_tau: float | None = None,
        protein_beta: float = 0.1,
        # Device / init
        device: str | None = None,
        smart_init: bool = True,
        smart_init_mod: str = "rna",
    ):
        super().__init__(mdata, modalities)

        if protein_likelihood not in ("nb", "multinomial"):
            raise ValueError(f"protein_likelihood must be 'nb' or 'multinomial', got {protein_likelihood!r}")

        self.K = int(n_topics)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.protein_likelihood = protein_likelihood

        # ---- Convert data to tensors -----------------------------------
        # dense_data: dense float tensors on device (for Gibbs + LL)
        # atac_coords: sparse [2, nnz] for chromatin (or None)
        self.dense_data: dict[str, Tensor] = {}
        self.atac_coords: Tensor | None = None

        for m in self.modalities:
            raw = self.data_dict[m]
            raw_X = _extract_X(raw)
            if m == "chromatin":
                # Store both sparse coords and dense version
                self.atac_coords = anndata_to_sparse_coords(raw_X)
                self.dense_data[m] = _to_dense_float(raw_X, self.device)
            else:
                self.dense_data[m] = _to_dense_float(raw_X, self.device)

        self.feature_dims: dict[str, int] = {m: self.dense_data[m].shape[1] for m in self.modalities}

        # ---- Dirichlet prior on theta ----------------------------------
        if np.isscalar(alpha):
            alpha = float(alpha)
            self.alpha = torch.full(
                (1, self.K),
                alpha,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.alpha = torch.tensor(
                alpha,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, -1)

        # ---- Modality-specific priors ----------------------------------
        self.priors: dict[str, dict[str, Tensor]] = {}
        self._init_priors(gamma, tau, beta, protein_gamma, protein_tau, protein_beta)

        # ---- ATAC batching (built lazily in fit) -----------------------
        self._cell_batches = None
        self._region_batches = None

        # ---- Initialise parameters -------------------------------------
        self.smart_init = smart_init
        self.smart_init_mod = smart_init_mod
        self._smart_init_done = False

        # ---- Posterior storage (populated by fit) ----------------------
        self.theta_samples: Tensor | None = None
        self.lambda_samples: dict[str, Tensor] | None = None
        self.ll_history: list[float] = []
        self.burnin_ll_trace: list[float] = []
        self.burnin_converged_at: int | None = None

    # ------------------------------------------------------------------
    #  Priors
    # ------------------------------------------------------------------
    def _init_priors(
        self,
        gamma: float | None,
        tau: float | None,
        beta: float,
        protein_gamma: float | None,
        protein_tau: float | None,
        protein_beta: float,
        kappa: float = 1.0,
        eps: float = 1e-3,
    ):
        """Set per-modality priors.  ``None`` values → empirical Bayes."""
        for m in self.modalities:
            X = self.dense_data[m]
            if m == "rna":
                if gamma is not None and tau is not None:
                    g = torch.tensor(gamma, device=self.device)
                    t = torch.tensor(tau, device=self.device)
                else:
                    mean = X.mean()
                    var = X.var(unbiased=False).clamp_min(1e-6)
                    g = mean**2 / var
                    t = mean / var
                self.priors[m] = {"gamma": g, "tau": t}
            elif m == "chromatin":
                peak_sum = X.sum(0).float()
                mean_prop = peak_sum / (peak_sum.sum() + eps)
                b = kappa * mean_prop + eps
                self.priors[m] = {"beta": b.to(self.device)}
            elif m == "protein":
                if self.protein_likelihood == "nb":
                    if protein_gamma is not None and protein_tau is not None:
                        g = torch.tensor(protein_gamma, device=self.device)
                        t = torch.tensor(protein_tau, device=self.device)
                    else:
                        mean = X.mean()
                        var = X.var(unbiased=False).clamp_min(1e-6)
                        g = mean**2 / var
                        t = mean / var
                    self.priors[m] = {"gamma": g, "tau": t}
                else:
                    peak_sum = X.sum(0).float()
                    mean_prop = peak_sum / (peak_sum.sum() + eps)
                    b = kappa * mean_prop + eps
                    self.priors[m] = {"beta": b.to(self.device)}

    # ------------------------------------------------------------------
    #  fit
    # ------------------------------------------------------------------
    def fit(
        self,
        *,
        batch_size: int = 512,
        n_samples: int = 500,
        thin: int = 10,
        burnin: int = 1_000,
        initial_burnin: int = 500,
        progress: bool = True,
        ll_every: int = 1,
        # auto burn-in
        auto_burnin: bool = True,
        min_initial_burnin: int = 50,
        burnin_window: int = 20,
        burnin_rtol: float = 1e-4,
        burnin_patience: int = 3,
    ) -> None:
        """Run the Gibbs sampler.

        Parameters
        ----------
        batch_size : int
            Mini-batch size (cells per update).
        n_samples : int
            Number of retained posterior samples.
        thin : int
            Keep every *thin*-th iteration during sampling.
        burnin : int
            Discarded iterations after initial burn-in.
        initial_burnin : int
            Maximum initial equilibration steps (may end early with
            *auto_burnin*).
        progress : bool
            Show tqdm progress bars.
        ll_every : int
            Evaluate joint log-likelihood every *ll_every* iterations.
        auto_burnin : bool
            Enable automatic thermalization detection during initial burn-in.
        min_initial_burnin : int
            Minimum iterations before auto burn-in can trigger.
        burnin_window : int
            Sliding window size for running-mean comparison.
        burnin_rtol : float
            Relative tolerance for LL stabilisation.
        burnin_patience : int
            Consecutive passes below *burnin_rtol* required to stop.
        """
        # Build ATAC batching structures if needed
        if "chromatin" in self.modalities and self.atac_coords is not None:
            self._cell_batches = create_cell_batches(
                self.atac_coords,
                batch_size,
                self.n_cells,
            )
            self._region_batches = create_region_batches(
                self.atac_coords,
                self._cell_batches,
            )

        # Initialise parameters
        if self.smart_init and not self._smart_init_done:
            mod = self.smart_init_mod
            if mod not in self.modalities:
                mod = self.modalities[0]
            theta, lambda_dict = smart_initialize(
                self.dense_data,
                self.K,
                self.priors,
                self.modalities,
                self.feature_dims,
                self.protein_likelihood,
                mod,
                self.device,
            )
            self._smart_init_done = True
        else:
            theta, lambda_dict = initialize_parameters(
                self.K,
                self.n_cells,
                self.priors,
                self.modalities,
                self.feature_dims,
                self.protein_likelihood,
                self.alpha,
                self.device,
            )

        results = run_gibbs_sampler(
            dense_data=self.dense_data,
            atac_coords=self.atac_coords,
            modalities=self.modalities,
            n_topics=self.K,
            feature_dims=self.feature_dims,
            alpha_vec=self.alpha,
            priors=self.priors,
            protein_likelihood=self.protein_likelihood,
            cell_batches=self._cell_batches,
            region_batches=self._region_batches,
            theta=theta,
            lambda_dict=lambda_dict,
            device=torch.device(self.device),
            n_samples=n_samples,
            thin=thin,
            burnin=burnin,
            initial_burnin=initial_burnin,
            batch_size=batch_size,
            progress=progress,
            ll_every=ll_every,
            auto_burnin=auto_burnin,
            min_initial_burnin=min_initial_burnin,
            burnin_window=burnin_window,
            burnin_rtol=burnin_rtol,
            burnin_patience=burnin_patience,
        )

        self.theta_samples = results["theta_samples"]
        self.lambda_samples = results["lambda_samples"]
        self.ll_history = results["ll_history"]
        self.burnin_ll_trace = results["burnin_ll_trace"]
        self.burnin_converged_at = results["burnin_converged_at"]

    # ------------------------------------------------------------------
    #  Accessors (BaseTopicModel contract)
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.theta_samples is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    @torch.inference_mode()
    def get_cell_topic_dist(self, normalised: bool = True) -> np.ndarray:
        """Posterior mean cell-topic proportions (C × K)."""
        self._check_fitted()
        theta = self.theta_samples.mean(0)
        if normalised:
            theta = theta / theta.sum(1, keepdim=True)
        return theta.cpu().numpy()

    @torch.inference_mode()
    def get_feature_topic_dist(self, modality: str) -> np.ndarray:
        """Posterior mean feature-topic matrix (K × Gm)."""
        self._check_fitted()
        if modality not in self.lambda_samples:
            raise ValueError(f"Modality {modality!r} not found.")
        return self.lambda_samples[modality].mean(0).cpu().numpy()

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------
    def get_perplexity(self, **kwargs) -> float:
        """exp(-LL / total_counts)."""
        self._check_fitted()
        theta = self.theta_samples.mean(0).to(self.device)
        lam = {m: self.lambda_samples[m].mean(0).to(self.device) for m in self.modalities}
        ll = compute_log_likelihood(
            self.dense_data,
            theta,
            lam,
            self.modalities,
            self.protein_likelihood,
        )["total"]
        total_counts = sum(self.dense_data[m].sum().item() for m in self.modalities)
        return float(np.exp(-ll / max(total_counts, 1)))

    def get_likelihood_per_modality(self, **kwargs) -> dict[str, float]:
        self._check_fitted()
        theta = self.theta_samples.mean(0).to(self.device)
        lam = {m: self.lambda_samples[m].mean(0).to(self.device) for m in self.modalities}
        ll = compute_log_likelihood(
            self.dense_data,
            theta,
            lam,
            self.modalities,
            self.protein_likelihood,
        )
        ll.pop("total", None)
        return ll

    def get_perplexity_per_modality(self, **kwargs) -> dict[str, float]:
        self._check_fitted()
        theta = self.theta_samples.mean(0).to(self.device)
        lam = {m: self.lambda_samples[m].mean(0).to(self.device) for m in self.modalities}
        ll = compute_log_likelihood(
            self.dense_data,
            theta,
            lam,
            self.modalities,
            self.protein_likelihood,
        )
        result = {}
        for m in self.modalities:
            total = self.dense_data[m].sum().item()
            result[m] = float(np.exp(-ll[m] / max(total, 1)))
        return result

    def get_modality_weights(self, **kwargs):
        """Not applicable for Gibbs — returns equal weights."""
        import pandas as pd

        w = 1.0 / len(self.modalities)
        return pd.DataFrame(
            {m: np.full(self.n_cells, w) for m in self.modalities},
        )

    # ------------------------------------------------------------------
    #  Generate
    # ------------------------------------------------------------------
    def generate(
        self,
        n_cells: int,
        theta: Tensor | None = None,
    ) -> dict[str, AnnData]:
        """Sample synthetic data from the fitted model.

        Returns
        -------
        dict[str, AnnData]
            One AnnData per modality with generated counts.
        """
        self._check_fitted()
        import anndata as ad
        from scipy.sparse import csr_matrix

        # Posterior means
        lam = {m: self.lambda_samples[m].mean(0) for m in self.modalities}

        if theta is None:
            alpha_vec = self.alpha.squeeze().cpu()
            theta = torch.distributions.Dirichlet(alpha_vec).sample([n_cells])

        result: dict[str, AnnData] = {}
        for m in self.modalities:
            rate = theta @ lam[m]
            if m == "rna" or (m == "protein" and self.protein_likelihood == "nb"):
                counts = torch.poisson(rate)
            elif m == "chromatin":
                counts = torch.bernoulli(rate.clamp(0, 1))
            else:
                # protein multinomial — sample from categorical per cell
                n_per_cell = torch.poisson(rate.sum(1, keepdim=True))
                probs = rate / (rate.sum(1, keepdim=True) + 1e-12)
                counts = (
                    torch.distributions.Multinomial(
                        total_count=1,
                        probs=probs,
                    )
                    .sample()
                    .squeeze()
                    * n_per_cell
                )

            adata = ad.AnnData(X=csr_matrix(to_numpy(counts)))
            result[m] = adata

        return result

    # ------------------------------------------------------------------
    #  Save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save fitted model to a ``.pt`` file."""
        self._check_fitted()
        state = {
            "n_topics": self.K,
            "alpha": to_numpy(self.alpha),
            "protein_likelihood": self.protein_likelihood,
            "modalities": self.modalities,
            "feature_dims": self.feature_dims,
            "priors": {m: {k: v.cpu() for k, v in p.items()} for m, p in self.priors.items()},
            "theta_samples": self.theta_samples,
            "lambda_samples": {m: v for m, v in self.lambda_samples.items()},
            "ll_history": self.ll_history,
            "burnin_ll_trace": self.burnin_ll_trace,
            "burnin_converged_at": self.burnin_converged_at,
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        mdata: MuDataType | dict[str, AnnData] | list[AnnData] | AnnData,
        *,
        modalities: list[str] | None = None,
        device: str | None = None,
    ) -> ShareTopic_LDA_Multi:
        """Load a saved model.

        The original data (*mdata*) must be passed again because we do not
        serialise the raw count matrices.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            mdata,
            modalities=modalities or state["modalities"],
            n_topics=state["n_topics"],
            alpha=state["alpha"].tolist() if hasattr(state["alpha"], "tolist") else state["alpha"],
            protein_likelihood=state["protein_likelihood"],
            device=device,
            smart_init=False,
        )
        model.theta_samples = state["theta_samples"]
        model.lambda_samples = state["lambda_samples"]
        model.ll_history = state["ll_history"]
        model.burnin_ll_trace = state["burnin_ll_trace"]
        model.burnin_converged_at = state["burnin_converged_at"]
        return model

    # ------------------------------------------------------------------
    def __repr__(self):
        status = "fitted" if self.theta_samples is not None else "not fitted"
        return (
            f"ShareTopic_LDA_Multi(n_topics={self.K}, "
            f"modalities={self.modalities}, "
            f"protein_likelihood={self.protein_likelihood!r}, "
            f"status={status})"
        )
