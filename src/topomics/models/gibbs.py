from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from anndata import AnnData
from torch import Tensor
from torch.distributions import Dirichlet, Gamma
from tqdm.auto import tqdm

from .base_model import BaseTopicModel

"""gibbs_model.py – Collapsed‑conjugate Gibbs sampler for multimodal LDA

The model places independent likelihoods on each modality:

* **RNA / protein** → Gamma–Poisson (aka Negative‑Binomial)
* **Chromatin / ATAC** → Dirichlet–Multinomial

Topic proportions **θₙ** are given a Dirichlet(α) prior that is *shared*
across modalities; each modality has its own feature‑specific priors.
"""

MuDataType = "muon.MuData"  # soft‑dep – no import unless actually used

__all__ = ["Gibbs_LDA_Multi"]


# helper
def _move_to(x: Tensor | np.ndarray, device: str) -> Tensor:
    """Ensure *x* is a Tensor on *device* (non‑blocking when possible)."""
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x)
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x


# -----------------------------------------------------------------------------
# main class ------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Gibbs_LDA_Multi(BaseTopicModel):
    """Multimodal LDA trained with a collapsed‑conjugate Gibbs sampler."""

    def __init__(
        self,
        mdata: MuDataType | dict[str, AnnData] | list[AnnData] | AnnData,
        *,
        modalities: list[str] | None = None,
        n_topics: int = 20,
        alpha: float | list[float] = 0.1,
        device: str | None = None,
        smart_init: bool = True,
        smart_init_mod: str = "rna",
    ):
        super().__init__(mdata, modalities)

        self.K = int(n_topics)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.data_dict: dict[str, Tensor] = {
            m: _move_to(self._ensure_int_tensor(X), self.device) for m, X in self.data_dict.items()
        }
        self.C = next(iter(self.data_dict.values())).shape[0]

        # ---------------- Dirichlet prior on θ -----------------------------
        if np.isscalar(alpha):
            alpha = float(alpha)
            alpha = torch.full((1, self.K), alpha, dtype=torch.float32, device=self.device)
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.alpha: Tensor = alpha  # shape (1, K)

        # ---------------- modality‑specific priors -------------------------
        self._init_priors()

        # ---------------- initial parameter samples ------------------------
        self._initialise_parameters(smart_init=smart_init, smart_init_mod=smart_init_mod)

    # ------------------------------------------------------------------
    #                          public interface
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
    ) -> None:
        """Run Gibbs sampler and cache **θ**, **Λ/Φ** and log‑likelihood.

        Parameters
        ----------
        batch_size : int – mini‑batch size (cells per update)
        n_samples  : int – number of *retained* posterior samples
        thin       : int – keep every *thin‑th* iteration
        burnin     : int – discarded iterations after initial burn‑in
        initial_burnin : int – fast equilibration steps not recorded
        progress   : bool – show tqdm progress bar
        ll_every   : int – evaluate joint log‑likelihood this often
        """
        total_iters = initial_burnin + burnin + n_samples * thin

        keep_mask = torch.zeros(total_iters, dtype=torch.bool)
        keep_mask[initial_burnin + burnin :: thin] = True
        S = int(keep_mask.sum().item())  # number of stored samples

        # ---------------- allocate storage ---------------------------
        theta_samps = torch.empty(S, self.C, self.K, dtype=torch.float32)
        lambda_samps: dict[str, Tensor] = {
            m: torch.empty(S, self.K, self.data_dict[m].shape[1], dtype=torch.float32) for m in self.modalities
        }
        ll_history: list[float] = []
        store_idx = 0

        iterator: Iterator[int] = range(total_iters)
        if progress:
            iterator = tqdm(iterator, total=total_iters, desc="Gibbs")

        # ---------------- main loop ----------------------------------
        for it in iterator:
            self._gibbs_epoch(batch_size)

            # ---- likelihood -------------------------------------
            if (it >= initial_burnin) and (it % ll_every == 0):
                ll_val = self._log_likelihood().item()
                ll_history.append(ll_val)
                if progress:
                    iterator.set_postfix({"ll": f"{ll_val:.2e}"})
                else:
                    print(f"Iter {it:>6d}  LL = {ll_val:.3e}")

            # ---- store samples ----------------------------------
            if keep_mask[it]:
                theta_samps[store_idx] = self.theta.cpu()
                for m in self.modalities:
                    lambda_samps[m][store_idx] = self.lambda_[m].cpu()
                store_idx += 1

        # -------------- expose cached results ----------------------
        self.theta_samples: Tensor = theta_samps  # (S × C × K)
        self.lambda_samples: dict[str, Tensor] = lambda_samps  # mod → (S × K × G_m)
        self.ll_history: list[float] = ll_history

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_cell_topic_dist(self, normalised: bool = True) -> np.ndarray:
        """Return E[θ] across retained samples (cells × K)."""
        assert hasattr(self, "theta_samples"), "Run .fit() first."
        theta = self.theta_samples.mean(0)  # (C × K)
        if normalised:
            theta = theta / theta.sum(1, keepdim=True)
        return theta.cpu().numpy()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_feature_topic_dist(self, modality: str) -> np.ndarray:
        """Return posterior mean Λₖ,₉ / Φₖ,ₚ (topics × features)."""
        assert hasattr(self, "lambda_samples"), "Run .fit() first."
        if modality not in self.lambda_samples:
            raise ValueError(f"Modality {modality!r} not found.")
        return self.lambda_samples[modality].mean(0).cpu().numpy()

    # ------------------------------------------------------------------
    #                       internal machinery
    # ------------------------------------------------------------------
    def _ensure_int_tensor(self, X) -> Tensor:
        """Convert AnnData.X to *integer* Tensor (dense) – sparse supported."""
        if isinstance(X, Tensor):
            return X.long()
        if hasattr(X, "A"):
            X = X.A  # handle numpy.matrix
        if torch.is_tensor(X):
            return X.long()
        import scipy.sparse as sp

        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.integer):
            if not np.allclose(X, np.rint(X)):
                raise ValueError("Counts must be integers.")
            X = np.rint(X).astype(int)
        return torch.from_numpy(X).long()

    # ------------------------------------------------------------------
    def _init_priors(self, kappa: float = 1.0, eps: float = 1e-3):
        """Empirical‑Bayes modality‑specific priors (Γ/Dirichlet)."""
        self.priors: dict[str, dict[str, Tensor]] = {}
        for m, X in self.data_dict.items():
            if m == "rna":
                mean = X.float().mean()
                var = X.float().var(unbiased=False).clamp_min(1e-6)
                gamma = mean**2 / var
                tau = mean / var
                self.priors[m] = {
                    "gamma": torch.tensor(gamma, device=self.device),
                    "tau": torch.tensor(tau, device=self.device),
                }
            elif m in {"chromatin", "protein"}:
                peak_sum = X.sum(0).float()
                mean_prop = peak_sum / peak_sum.sum()
                beta = kappa * mean_prop + eps
                self.priors[m] = {"beta": beta.to(self.device)}
            else:
                raise ValueError(f"Unknown modality '{m}'.")

    # ------------------------------------------------------------------
    def _initialise_parameters(self, smart_init: bool = True, smart_init_mod: str = "rna"):
        """Sample initial Θ, Λ/Φ from the priors."""
        if smart_init:
            from sklearn.decomposition import NMF

            # ---- prepare RNA matrix as dense float ------------------------------
            X_mod = self.data_dict[smart_init_mod].cpu().float().numpy()  # C × G

            # ---- factorise -------------------------------------------------------
            nmf = NMF(n_components=self.K, max_iter=300)
            W = nmf.fit_transform(X_mod)

            # ---- seed theta ---------------------------------------------------
            theta0 = torch.tensor(W, device=self.device)
            theta0 /= theta0.sum(1, keepdim=True)
            self.theta = theta0

            self.lambda_: dict[str, Tensor] = {}
            for m, _ in self.data_dict.items():
                n_k_f = topic_feature_counts(theta0, self.data_dict[m])
                if m == "rna":
                    gamma0 = self.priors[m]["gamma"]
                    tau0 = self.priors[m]["tau"]
                    lambda_0 = (gamma0 + n_k_f) / (tau0 + theta0.sum(0).unsqueeze(1))
                    self.lambda_[m] = lambda_0

                else:
                    beta0 = self.priors[m]["beta"]  # length P
                    phi_0 = beta0.unsqueeze(0) + n_k_f
                    phi_0 /= phi_0.sum(1, keepdim=True)
                    self.lambda_[m] = phi_0
        else:
            # θ – C × K
            self.theta: Tensor = Dirichlet(self.alpha.repeat(self.C, 1)).sample()

            # modality‑specific Λ/Φ ---------------
            self.lambda_: dict[str, Tensor] = {}
            for m, X in self.data_dict.items():
                Gm = X.shape[1]
                if m == "rna":
                    gamma, tau = self.priors[m]["gamma"], self.priors[m]["tau"]
                    dist = Gamma(gamma, tau)
                    lam = dist.sample((self.K, Gm))  # K × Gm
                    self.lambda_[m] = lam.to(self.device)
                else:
                    beta = self.priors[m]["beta"]  # vector len Gm
                    phi = Dirichlet(beta).sample((self.K,))  # K × Gm
                    self.lambda_[m] = phi.to(self.device)

    def _log_likelihood(
        self,
        *,
        include_const: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Joint log‑likelihood."""
        ll = torch.tensor(0.0, device=self.device)  # data‑dependent part
        const_part = torch.tensor(0.0, device=self.device)  # parameter‑independent
        θ = self.theta

        for m, X in self.data_dict.items():
            Λ = self.lambda_[m]

            if m == "rna":  # ----- Poisson -----
                rate = θ @ Λ + 1e-8
                ll += torch.sum(X * torch.log(rate) - rate)

                # combinatorial term  −log(x!)
                const_part -= torch.sum(torch.lgamma(X + 1))

            else:  # --- chromatin ----
                prob = θ @ Λ + 1e-8
                ll += torch.sum(X * torch.log(prob))

                # Multinomial constant −log(x!) + log(N!) (optional)
                N = X.sum(1)
                const_part += torch.sum(torch.lgamma(N + 1) - torch.lgamma(X + 1).sum(1))

        if include_const:
            ll_total = ll + const_part
        else:
            ll_total = ll
        return ll_total

    # ------------------------------------------------------------------
    # ------------------ single Gibbs sweep ----------------------------
    # ------------------------------------------------------------------
    def _gibbs_epoch(self, batch_size: int):
        """Full Gibbs sweep: first gather global counts, then resample Λ/Φ, finally θ."""
        # -----------------------------------------------------------------
        # 1) reset global (K×G) and (C×K) accumulators
        suff_stats = {m: torch.zeros_like(self.lambda_[m]) for m in self.modalities}
        n_ck = torch.zeros(self.C, self.K, device=self.device)

        perm = torch.randperm(self.C, device=self.device)
        for start in range(0, self.C, batch_size):
            idx = perm[start : start + batch_size]
            θ_b = self.theta[idx]

            # ---- modality loop ------------------------------------------
            for m in self.modalities:
                X_b = self.data_dict[m][idx]
                Λ_m = self.lambda_[m]

                if m == "rna":
                    n_ck_b, n_k_g_b = self._exp_counts_poisson(X_b, θ_b, Λ_m)
                else:
                    n_ck_b, n_k_g_b = self._exp_counts_multinom(X_b, θ_b, Λ_m)

                n_ck[idx] += n_ck_b
                suff_stats[m] += n_k_g_b  # accumulate across batches

        # -----------------------------------------------------------------
        # 2) resample Λ/Φ **once per modality** from the full sufficient stats
        θ_sum = self.theta.sum(0).unsqueeze(1)  # K×1  (all cells!)
        for m in self.modalities:
            if m == "rna":
                γ0, τ0 = self.priors[m]["gamma"], self.priors[m]["tau"]
                shape = γ0 + suff_stats[m]
                rate = τ0 + θ_sum
                self.lambda_[m] = Gamma(shape, rate).sample()
            else:
                β0 = self.priors[m]["beta"]
                α_post = β0.unsqueeze(0) + suff_stats[m]
                self.lambda_[m] = Dirichlet(α_post).sample()

        # -----------------------------------------------------------------
        # 3) sample θ cell-wise given the full n_ck
        α_post = self.alpha + n_ck
        self.theta = Dirichlet(α_post).sample()

    # ------------------------------------------------------------------
    def _gibbs_batch(self, idx: Tensor):
        """Gibbs update for a mini‑batch of cells (indices *idx*)."""
        theta_batch = self.theta[idx]  # view (B × K)
        n_ck = torch.zeros_like(theta_batch)  # will accumulate expected counts

        # ------- iterate over modalities -----------------------------
        for m in self.modalities:
            X = self.data_dict[m][idx]  # B × G_m  (int counts)
            lam = self.lambda_[m]  # K × G_m

            if m == "rna":
                n_ck_m, m_k_g, new_lam = self._gamma_poisson_update(X, theta_batch, lam, self.priors[m])
                self.lambda_[m][:] = new_lam  # update in‑place
            else:  # chromatin
                n_ck_m, m_k_g, new_phi = self._dirichlet_multinomial_update(X, theta_batch, lam, self.priors[m])
                self.lambda_[m][:] = new_phi  # update φ

            n_ck += n_ck_m  # accumulate across modalities

        # ---------------- θ update (Dirichlet) ------------------------
        alpha_post = self.alpha + n_ck  # B × K
        self.theta[idx] = Dirichlet(alpha_post).sample()

    # ------------------------------------------------------------------
    # ---------------------- update helpers ---------------------------
    # ------------------------------------------------------------------
    def _exp_counts_poisson(self, x: Tensor, θ: Tensor, Λ: Tensor) -> tuple[Tensor, Tensor]:
        rate = θ @ Λ + 1e-12  # B×G
        p = (θ.unsqueeze(2) * Λ.unsqueeze(0)) / rate.unsqueeze(1)
        exp_n = x.unsqueeze(1) * p  # B×K×G
        return exp_n.sum(2), exp_n.sum(0)  # n_ck_b, n_k_g_b

    def _exp_counts_multinom(self, x: Tensor, θ: Tensor, Φ: Tensor) -> tuple[Tensor, Tensor]:
        rate = θ @ Φ + 1e-12  # B×P
        p = (θ.unsqueeze(2) * Φ.unsqueeze(0)) / rate.unsqueeze(1)
        exp_n = x.unsqueeze(1) * p  # B×K×P
        return exp_n.sum(2), exp_n.sum(0)

    def _gamma_poisson_update(
        self,
        x: Tensor,  # B × G
        theta: Tensor,  # B × K
        lam: Tensor,  # K × G
        priors: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update Λ (Gamma) and latent counts for a *Gamma–Poisson* modality."""
        B, G = x.shape

        # ---------------- expected topic allocations ----------------
        #   p_{ctk} ∝ θ_{ck} Λ_{kg}
        rate = theta @ lam  # B × G   (μ)
        # avoid zero division for empty genes
        p = (theta.unsqueeze(2) * lam.unsqueeze(0)) / (rate.unsqueeze(1) + 1e-12)
        exp_n_tcg = x.unsqueeze(1) * p  # B × K × G

        n_ck = exp_n_tcg.sum(2)  # B × K
        n_k_g = exp_n_tcg.sum(0)  # K × G

        # ---------------- Λ posterior -------------------------------
        gamma0, tau0 = priors["gamma"], priors["tau"]
        shape_post = gamma0 + n_k_g  # K × G
        rate_post = tau0 + theta.sum(0).unsqueeze(1)  # K × 1  broadcast
        new_lam = Gamma(shape_post, rate_post).sample()

        return n_ck, n_k_g, new_lam

    # ------------------------------------------------------------------
    def _dirichlet_multinomial_update(
        self,
        x: Tensor,  # B × P        (peak counts – usually binary)
        theta: Tensor,  # B × K
        phi: Tensor,  # K × P
        priors: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update Φ (Dirichlet) for ATAC / chromatin modality."""
        B, P = x.shape

        # expected topic allocations -------------------------------
        prob = theta @ phi  # B × P
        prob = prob.clamp_min(1e-12)
        p = (theta.unsqueeze(2) * phi.unsqueeze(0)) / prob.unsqueeze(1)  # B × K × P
        exp_n_tcp = x.unsqueeze(1) * p

        n_ck = exp_n_tcp.sum(2)  # B × K
        n_k_p = exp_n_tcp.sum(0)  # K × P

        # Dirichlet posterior --------------------------------------
        beta0 = priors["beta"]  # vector P
        alpha_post = beta0.unsqueeze(0) + n_k_p  # K × P
        new_phi = Dirichlet(alpha_post).sample()

        return n_ck, n_k_p, new_phi


def topic_feature_counts(
    theta: torch.Tensor,
    X_counts: torch.Tensor,
) -> torch.Tensor:
    """Return expected topic–feature counts.

    Parameters
    ----------
    theta : torch.Tensor
        C × K matrix of topic proportions (float32, CPU or GPU).
    X_counts : torch.Tensor
        C × G count matrix (int64).

    Returns
    -------
    torch.Tensor
        K × G matrix of expected counts.
    """
    Xf = X_counts.to(theta.device, dtype=torch.float32)  # cast + move
    return theta.T @ Xf
