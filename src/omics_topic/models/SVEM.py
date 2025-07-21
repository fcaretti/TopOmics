from __future__ import annotations

import math

import muon as mu
import torch
from anndata import AnnData
from scipy import sparse
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_model import BaseModel

MuDataType = mu.MuData  # type: ignore

# -----------------------------------------------------------------------------
# tiny helper ------------------------------------------------------------------
# -----------------------------------------------------------------------------

digamma = torch.digamma

# -----------------------------------------------------------------------------
# likelihood registry ----------------------------------------------------------
# -----------------------------------------------------------------------------

MOD_REG = {
    "rna": ("_gamma_poisson_init", "gamma_poisson"),
    "protein": ("_gamma_poisson_init", "gamma_poisson"),
    "chromatin": ("_dirichlet_init", "dirichlet_multinomial"),
}


# -----------------------------------------------------------------------------
class SVEM_LDA_Multi(BaseModel):
    """
    Stochastic Variational EM for MuData with shared topics across modalities.

    Args:
        mdata: MuDataType | dict[str, AnnData] | dict[str, Tensor]
            Multi-modal data container.
        modalities: list[str] | None
            Names corresponding to each AnnData in a list input.
        n_topics: int
            Number of topics to learn.
        batch_size: int
            Batch size for training.
        feature_frac: float
            Fraction of features to use for training.
        alpha: float
            Dirichlet prior parameter for topic distributions.
        device: str
            Device to run the model on ('cuda' or 'cpu').
        entropy_penalty: float
            Entropy penalty for regularization.
        mod_weights: dict[str, float] | None
            Weights for each modality.
    """

    def __init__(
        self,
        mdata: MuDataType | dict[str, AnnData] | dict[str, Tensor],
        modalities=None,
        n_topics: int = 20,
        batch_size: int = 512,
        feature_frac: float = 1.0,
        alpha: float = 0.1,
        device: str = "cuda",
        entropy_penalty: float = 0.0,
        mod_weights: dict[str, float] = None,
    ):
        """
        Initialize the SVEM_LDA_Multi model.

        Calls the super() constructor to initialize the base model with the provided data and modalities.
        Then, loads data to tensors and initializes model parameters.
        """
        super().__init__(mdata, modalities)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.K = n_topics
        self.batch_size = batch_size
        self.feature_frac = feature_frac
        self.alpha_scalar = alpha
        self.entropy_penalty = entropy_penalty

        # sanity: same number of cells across modalities
        n_cells_set = {v.shape[0] for v in self.data_dict.values()}

        if len(n_cells_set) != 1:
            raise ValueError("All modalities must share the same cells / order")
        self.C = n_cells_set.pop()

        # Running SVEM requires torch
        for mod, X in list(self.data_dict.items()):
            if isinstance(X, torch.Tensor):
                self.data_dict[mod] = X.long()
                continue
            if sparse.issparse(X):  # catches csr, csc, coo, …
                X = torch.as_tensor(X.toarray())
            else:
                X = torch.as_tensor(X)

            self.data_dict[mod] = X.long()  # keep counts as integers

        # ---------------- initialise per‑modality gamma‑params --------------
        self.A: dict[str, Tensor] = {}
        self.B: dict[str, Tensor] = {}
        for mod in self.modalities:
            init_fn_name, _ = MOD_REG.get(mod, (None, None))
            if init_fn_name is None:
                raise ValueError(f"Modality '{mod}' not recognised in MOD_REG")
            init_fn = getattr(self, init_fn_name)
            A, B = init_fn(self.data_dict[mod])
            self.A[mod] = A.to(self.device)
            self.B[mod] = B.to(self.device)

        self.mod_weights = dict.fromkeys(self.modalities, 1.0) if mod_weights is None else mod_weights

        # ---------------- Dirichlet gamma  (cells × K) ----------------------
        self.gamma = torch.full((self.C, self.K), alpha, device=self.device)

        # ---------------- DataLoader: concatenate features -------------
        indices = torch.arange(self.C)
        concat = torch.cat(list(self.data_dict.values()), dim=1)
        self.feature_offsets: dict[str, slice] = {}
        off = 0
        for mod in self.modalities:
            Gm = self.data_dict[mod].shape[1]
            self.feature_offsets[mod] = slice(off, off + Gm)
            off += Gm

        self.loader = DataLoader(
            TensorDataset(indices, concat),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # modality‑specific prior initialisation ---------------------------
    # ------------------------------------------------------------------
    def _gamma_poisson_init(self, X: Tensor) -> tuple[Tensor, Tensor]:
        mean, var = X.float().mean(0), X.float().var(0, unbiased=False).clamp_min(1e-6)
        a_gene, b_gene = (mean**2) / var, mean / var
        A = torch.stack([torch.distributions.Gamma(a_gene, b_gene).sample() for _ in range(self.K)])
        B = b_gene.expand(self.K, -1).clone()
        return A, B

    def _dirichlet_init(self, X: Tensor) -> tuple[Tensor, Tensor]:
        peak_sum = X.sum(0).float()
        alpha_vec = 0.1 + peak_sum / peak_sum.sum()
        A = alpha_vec.expand(self.K, -1).clone()
        B = torch.ones_like(A)  # placeholder
        return A, B

    # ------------------------------------------------------------------
    # training loop ----------------------------------------------------
    # ------------------------------------------------------------------
    def fit(
        self,
        n_epochs: int = 100,
        inner_iters: int = 2,
        kappa: float = 0.6,
        tau0: float = 1024.0,
        lr_mult=1.0,
        verbose=True,
    ):
        """
        Fits the model to the data using stochastic variational EM.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train.
        inner_iters : int
            Number of inner iterations to re-estimate γ.
        kappa : float
            Learning rate decay parameter.
        tau0 : float
            Learning rate offset.
        lr_mult : float
            Learning rate multiplier.
        verbose : bool
            Whether to print progress and metrics.
        """
        step = 0  # <-- η, try 0.05–0.2
        alpha_vec = torch.full((1, self.K), self.alpha_scalar, device=self.device)

        for epoch in range(1, n_epochs + 1):
            elbo_epoch = 0.0
            token_epoch = 0
            iterator = tqdm(self.loader, desc=f"epoch {epoch}")
            for idx_cpu, megaX_cpu in iterator:
                idx = idx_cpu.to(self.device, non_blocking=True)
                megaX = megaX_cpu.to(self.device, non_blocking=True)
                B = megaX.shape[0]
                # holder for batch‑wise topic counts across all modalities
                n_ck_batch = torch.zeros(B, self.K, device=self.device)

                # -------- loop over modalities ------------------
                for mod in self.modalities:
                    sl = self.feature_offsets[mod]
                    x_full = megaX[:, sl]  # B × G_m
                    Gm = x_full.shape[1]
                    # subsample genes if feature_frac < 1
                    if self.feature_frac < 1.0:
                        g_idx = torch.randperm(Gm, device=self.device)[: max(1, int(self.feature_frac * Gm))]
                        x = x_full[:, g_idx]
                        A_mod = self.A[mod][:, g_idx]
                        B_mod = self.B[mod][:, g_idx]
                        scale_features = Gm / g_idx.numel()
                    else:
                        x = x_full
                        A_mod, B_mod = self.A[mod], self.B[mod]
                        scale_features = 1.0

                    gamma_batch = self.gamma[idx]  # B × K view
                    w = self.mod_weights[mod]
                    x = x * w

                    if MOD_REG[mod][1] == "gamma_poisson":
                        n_ck_local, m_k_g = self._e_gamma_poisson(x, gamma_batch, A_mod, B_mod)
                    elif MOD_REG[mod][1] == "dirichlet_multinomial":
                        n_ck_local, m_k_g = self._e_dirichlet_multinomial(x, gamma_batch, A_mod)
                    else:
                        raise ValueError(f"Unknown likelihood for modality '{mod}'")
                    # ---------------------------------------

                    n_ck_batch += n_ck_local  # accumulate across modalities

                    # ---------- global Λ_m update (natural gradient) -----
                    # scale by cells & features then apply Robbins–Monro
                    n_k = n_ck_local.sum(0) * (self.C / B)
                    m_k_g *= (self.C / B) * scale_features
                    step += 1
                    rho_t = lr_mult * (tau0 + step) ** (-kappa)
                    # rho_t = math.pow(tau0 + step, -kappa)
                    self.A[mod][:, g_idx if self.feature_frac < 1.0 else ...].mul_(1 - rho_t).add_(
                        m_k_g + 0.1, alpha=rho_t
                    )  # 0.1 prior stub
                    self.B[mod].mul_(1 - rho_t).add_(n_k.unsqueeze(1) + 0.1, alpha=rho_t)

                # -------------- update γ (shared) ------------------------
                gamma_batch = alpha_vec + n_ck_batch
                if self.entropy_penalty > 0:
                    theta = gamma_batch / gamma_batch.sum(1, keepdim=True)
                    p = 1.0 / (1.0 - self.entropy_penalty)
                    sharp = theta.clamp_min(1e-12).pow(p)
                    sharp = sharp / sharp.sum(1, keepdim=True)
                    gamma_batch = sharp * gamma_batch.sum(1, keepdim=True)
                self.gamma[idx] = gamma_batch.detach()

            # --------- monitoring (rough perplexity) -----------------
            with torch.no_grad():
                theta = gamma_batch / gamma_batch.sum(1, keepdim=True)  # B × K

                for mod in self.modalities:
                    sl = self.feature_offsets[mod]  # slice of megaX occupied by this modality
                    x_full = megaX[:, sl]  # B × G_m (no feature-subsampling)
                    if MOD_REG[mod][1] == "gamma_poisson":
                        rate = theta @ (self.A[mod] / self.B[mod])  # B × G_m
                        w = self.mod_weights[mod]
                        elbo_epoch += w * (x_full * torch.log(rate.clamp(min=1e-8))).sum().item()
                        token_epoch += int((w * x_full).sum())
                    elif MOD_REG[mod][1] == "dirichlet_multinomial":
                        phi = self.A[mod] / self.A[mod].sum(1, keepdim=True)  # K × P
                        prob = theta @ phi  # B × P
                        w = self.mod_weights[mod]
                        elbo_epoch += w * (x_full * torch.log(prob.clamp(min=1e-8))).sum().item()
                        token_epoch += int((w * x_full).sum())

            if verbose:
                ppl = math.exp(-elbo_epoch / max(token_epoch, 1))
                print(f"epoch {epoch:3d} | perplexity ≈ {ppl:.4f} | elbo ≈ {elbo_epoch:.4f} | ")

    @staticmethod
    def _e_gamma_poisson(
        x: Tensor,  # B × G
        gamma_batch: Tensor,  # B × K (view of global γ rows)
        A: Tensor,
        B: Tensor,  # K × G (variational Γ params for λ)
    ) -> tuple[Tensor, Tensor]:
        """
        E-step for a Gamma–Poisson / Negative-Binomial likelihood.

        Returns
        -------
            n_ck : B × K   expected topic counts per cell
            m_k_g: K × G   expected gene counts per topic
        """
        log_theta = digamma(gamma_batch) - digamma(gamma_batch.sum(1, keepdim=True))  # B×K
        log_lambda = digamma(A) - torch.log(B)  # K×G
        exp_lambda = A / B  # K×G

        log_w = (
            log_theta.unsqueeze(2)  # B × K × 1
            + x.unsqueeze(1) * log_lambda.unsqueeze(0)  # B × K × G
            - exp_lambda.unsqueeze(0)  # B × K × G
        )
        rho = torch.softmax(log_w, dim=1)  # B × K × G
        n_ck = rho.sum(2)  # B × K
        m_k_g = (rho * x.unsqueeze(1)).sum(0)  # K × G
        return n_ck, m_k_g

    @staticmethod
    def _e_dirichlet_multinomial(
        x: Tensor,  # B × P   (peaks / features)
        gamma_batch: Tensor,  # B × K
        alpha: Tensor,  # K × P   Dirichlet concentration (A-matrix)
    ) -> tuple[Tensor, Tensor]:
        """
        E-step for a Dirichlet–Multinomial likelihood used for ATAC / chromatin.

        `alpha` plays the role of φ’s Dirichlet parameters.
        """
        log_theta = digamma(gamma_batch) - digamma(gamma_batch.sum(1, keepdim=True))  # B×K
        phi = alpha / alpha.sum(1, keepdim=True)  # K×P
        log_phi = torch.log(phi.clamp_min(1e-32))  # K×P

        log_w = log_theta.unsqueeze(2) + x.unsqueeze(1) * log_phi.unsqueeze(0)  # B×K×P
        rho = torch.softmax(log_w, dim=1)  # B×K×P
        n_ck = rho.sum(2)  # B×K
        m_k_p = (rho * x.unsqueeze(1)).sum(0)  # K×P
        return n_ck, m_k_p

    # ------------------------------------------------------------------
    # accessors --------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def topic_by_feature(self, mod: str) -> Tensor:
        """Returns the topic-by-feature matrix for the specified modality."""
        return (self.A[mod] / self.B[mod]).cpu()

    @torch.inference_mode()
    def cell_topic_distribution(self, normalised: bool = True):
        """
        Returns the cell-topic distribution (γ) for all cells.

        If `normalised` is True, returns the normalised distribution.
        """
        gamma = self.gamma.detach().cpu()
        return gamma / gamma.sum(1, keepdim=True) if normalised else gamma

    @torch.inference_mode()
    def diagnostics(
        self,
        loader: DataLoader | None = None,
        mod_weights: dict[str, float] | None = None,
        inner_iters: int = 3,
    ) -> dict[str, float]:
        """
        Computes corpus log-likelihood, token-level perplexity and mean entropy of θ̂_c for the current variational parameters.

        Parameters
        ----------
        loader       : DataLoader that yields (idx, megaX) pairs.
                    If None, reuse self.loader (full data, no shuffle).
        mod_weights  : optional per-modality weights to match training.
        inner_iters  : number of coordinate-ascent steps to re-estimate γ
                    for held-out cells (default 3).

        Returns
        -------
        dict( log_lik = … , perplexity = … , entropy = … )
        """
        if loader is None:
            loader = self.loader
        if mod_weights is None:
            mod_weights = getattr(self, "mod_weights", dict.fromkeys(self.modalities, 1.0))

        log_lik_tot, token_tot, entropy_tot, C_seen = 0.0, 0, 0.0, 0
        for _, megaX_cpu in loader:
            megaX = megaX_cpu.to(self.device, non_blocking=True)
            B = megaX.shape[0]

            # ---- infer θ̂ for these cells (no global updates) ------------
            gamma = torch.full((B, self.K), self.alpha_scalar, device=self.device)
            log_lambda_cache = {
                m: digamma(self.A[m]) - torch.log(self.B[m]) if MOD_REG[m][1] == "gamma_poisson" else None
                for m in self.modalities
            }
            exp_lambda_cache = {
                m: (self.A[m] / self.B[m]).to(self.device) if MOD_REG[m][1] == "gamma_poisson" else None
                for m in self.modalities
            }
            for _ in range(inner_iters):
                log_theta = digamma(gamma) - digamma(gamma.sum(1, keepdim=True))
                n_ck = torch.zeros_like(gamma)
                off = 0
                for m in self.modalities:
                    Gm = self.data_dict[m].shape[1]
                    x = megaX[:, off : off + Gm]
                    off += Gm

                    if MOD_REG[m][1] == "gamma_poisson":
                        log_w = (
                            log_theta.unsqueeze(2)
                            + x.unsqueeze(1) * log_lambda_cache[m].unsqueeze(0)
                            - exp_lambda_cache[m].unsqueeze(0)
                        )
                        rho = torch.softmax(log_w, 1)
                        n_ck += rho.sum(2)
                    else:  # dirichlet_multinomial
                        phi = (self.A[m] / self.A[m].sum(1, keepdim=True)).to(self.device)
                        log_phi = torch.log(phi.clamp_min(1e-32))
                        log_w = log_theta.unsqueeze(2) + x.unsqueeze(1) * log_phi.unsqueeze(0)
                        rho = torch.softmax(log_w, 1)
                        n_ck += rho.sum(2)
                gamma = self.alpha_scalar + n_ck
            theta = gamma / gamma.sum(1, keepdim=True)  # B × K

            # ---- compute log-likelihood over modalities ------------------
            off = 0
            for m in self.modalities:
                w = mod_weights.get(m, 1.0)
                Gm = self.data_dict[m].shape[1]
                x = megaX[:, off : off + Gm]
                off += Gm

                if MOD_REG[m][1] == "gamma_poisson":
                    rate = theta @ exp_lambda_cache[m]  # B × Gm
                    log_lik_tot += w * (x * torch.log(rate.clamp(min=1e-8))).sum().item()
                    token_tot += int((w * x).sum())
                else:  # dirichlet_multinomial
                    phi = (self.A[m] / self.A[m].sum(1, keepdim=True)).to(self.device)
                    prob = theta @ phi
                    log_lik_tot += w * (x * torch.log(prob.clamp(min=1e-8))).sum().item()
                    token_tot += int((w * x).sum())

            # entropy of θ
            entropy_tot += (-theta * torch.log(theta.clamp(min=1e-8))).sum().item()
            C_seen += B

        perplexity = math.exp(-log_lik_tot / max(token_tot, 1))
        mean_entropy = entropy_tot / C_seen
        return {"log_lik": log_lik_tot, "perplexity": perplexity, "entropy": mean_entropy}
