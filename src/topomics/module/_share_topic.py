"""Gibbs sampler internals for SHARE-Topic multimodal LDA.

This module provides the computational machinery for the ShareTopic_LDA_Multi
model: sparse ATAC batching, per-modality Gibbs updates (RNA Gamma-Poisson,
ATAC sparse Bernoulli, Protein NB or Multinomial), automatic burn-in detection,
and the main sampler loop.

No model-level logic lives here — only pure functions and helper dataclasses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Dirichlet, Gamma
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-12

# ---------------------------------------------------------------------------
# A.  Data conversion
# ---------------------------------------------------------------------------


def anndata_to_sparse_coords(X) -> Tensor:
    """Convert a sparse binary matrix to ``[2, nnz]`` coordinate tensor.

    Parameters
    ----------
    X : scipy.sparse matrix or np.ndarray
        Binary (cells x regions) matrix.

    Returns
    -------
    torch.LongTensor, shape ``[2, nnz]``
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        coo = X.tocoo()
        rows = coo.row.copy()
        cols = coo.col.copy()
    else:
        arr = np.asarray(X)
        rows, cols = arr.nonzero()
    return torch.stack([
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
    ])


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Safely convert a torch tensor to numpy."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        return np.array(tensor.detach().cpu().tolist())


# ---------------------------------------------------------------------------
# B.  Cell / region batching  (ported from SHARE-Topic _batching.py)
# ---------------------------------------------------------------------------


@dataclass
class CellBatches:
    """Partitioning of cells into batches with ATAC coordinate boundaries."""
    cell_boundaries: Tensor
    atac_cell_boundaries: Tensor
    region_counts_per_cell: Tensor
    cell_indices_expanded: Tensor
    cell_array: Tensor


@dataclass
class RegionBatches:
    """Region grouping per cell-batch with sort/unsort index arrays."""
    regions: Tensor
    region_batch_boundaries: Tensor
    region_rep: Tensor
    region_rep_expanded: Tensor
    sort_indices: Tensor
    unsort_indices: Tensor


def create_cell_batches(atac: Tensor, batch_size: int, n_cells: int) -> CellBatches:
    """Partition cells into batches and compute ATAC coordinate boundaries."""
    n_batches = n_cells + batch_size - (n_cells % batch_size)
    cell_boundaries = torch.arange(0, n_batches, batch_size)
    cell_boundaries = torch.hstack((cell_boundaries, torch.tensor(n_cells)))

    _, _, region_counts_per_cell = torch.unique(
        atac[0, :], return_inverse=True, return_counts=True,
    )
    region_counts_per_cell = region_counts_per_cell.long()
    cell_array = torch.arange(n_cells)
    cell_indices_expanded = torch.repeat_interleave(
        cell_array, region_counts_per_cell, dim=0,
    )

    atac_cell_boundaries = torch.zeros(cell_boundaries.shape[0], dtype=torch.long)
    q = 1
    t = 0
    t_ = 0
    for i in torch.arange(batch_size, n_batches, batch_size):
        atac_cell_boundaries[q] = t + torch.sum(region_counts_per_cell[t_:i])
        t = int(atac_cell_boundaries[q].item())
        t_ = i
        q += 1
    atac_cell_boundaries[q] = t + torch.sum(region_counts_per_cell[t_:])

    return CellBatches(
        cell_boundaries=cell_boundaries,
        atac_cell_boundaries=atac_cell_boundaries,
        region_counts_per_cell=region_counts_per_cell,
        cell_indices_expanded=cell_indices_expanded,
        cell_array=cell_array,
    )


def create_region_batches(atac: Tensor, cell_batches: CellBatches) -> RegionBatches:
    """Group regions by cell batch with sorted indices."""
    acb = cell_batches.atac_cell_boundaries
    nnz = atac.shape[1]

    sort_indices = torch.zeros(nnz, dtype=torch.long)
    unsort_indices = torch.zeros(nnz, dtype=torch.long)
    regions = torch.tensor([], dtype=torch.long)
    region_rep = torch.tensor([], dtype=torch.long)
    region_rep_expanded = torch.tensor([], dtype=torch.long)
    region_batch_boundaries = torch.zeros(acb.shape[0], dtype=torch.long)

    for i in torch.arange(1, acb.shape[0]):
        lo = int(acb[i - 1].item())
        hi = int(acb[i].item())
        if lo == hi:
            region_batch_boundaries[i] = region_batch_boundaries[i - 1]
            continue

        sorted_vals, si = torch.sort(atac[1, lo:hi])
        sort_indices[lo:hi] = si
        _, ui = torch.sort(si)
        unsort_indices[lo:hi] = ui

        unique_regions, _, counts = torch.unique(
            sorted_vals, return_inverse=True, return_counts=True,
        )
        regions = torch.cat((regions, unique_regions), 0)
        region_batch_boundaries[i] = unique_regions.shape[0] + region_batch_boundaries[i - 1]
        region_rep = torch.cat((region_rep, counts), 0)

        local_array = torch.arange(unique_regions.shape[0])
        rep_expanded = torch.repeat_interleave(local_array, counts.long(), dim=0)
        region_rep_expanded = torch.cat((region_rep_expanded, rep_expanded), 0)

    return RegionBatches(
        regions=regions,
        region_batch_boundaries=region_batch_boundaries,
        region_rep=region_rep,
        region_rep_expanded=region_rep_expanded,
        sort_indices=sort_indices,
        unsort_indices=unsort_indices,
    )


def _move_batching_to_device(
    cell_batches: CellBatches,
    region_batches: RegionBatches,
    device: torch.device,
) -> tuple:
    """Transfer batching tensors to *device*. Returns a flat tuple."""
    cb = cell_batches
    rb = region_batches
    return (
        cb.cell_boundaries.to(device),
        cb.region_counts_per_cell.to(device),
        cb.cell_indices_expanded.to(device),
        cb.atac_cell_boundaries.to(device),
        rb.regions.to(device),
        rb.region_batch_boundaries,        # kept on CPU (python indexing)
        rb.region_rep.to(device),
        rb.region_rep_expanded.to(device),
        rb.sort_indices.to(device),
        rb.unsort_indices.to(device),
    )


# ---------------------------------------------------------------------------
# C.  Parameter initialisation
# ---------------------------------------------------------------------------


def initialize_parameters(
    n_topics: int,
    n_cells: int,
    priors: dict[str, dict[str, Tensor]],
    modalities: list[str],
    feature_dims: dict[str, int],
    protein_likelihood: str,
    alpha_vec: Tensor,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sample initial theta and per-modality lambda/phi from priors."""
    theta = Dirichlet(alpha_vec.squeeze()).sample([n_cells]).to(device)

    lambda_dict: dict[str, Tensor] = {}
    for m in modalities:
        Gm = feature_dims[m]
        if m == "rna" or (m == "protein" and protein_likelihood == "nb"):
            gamma = priors[m]["gamma"]
            tau = priors[m]["tau"]
            lambda_dict[m] = Gamma(gamma, tau).sample((n_topics, Gm)).to(device)
        else:
            # chromatin or protein-multinomial
            beta = priors[m]["beta"]
            lambda_dict[m] = Dirichlet(beta).sample((n_topics,)).to(device)

    return theta, lambda_dict


def smart_initialize(
    dense_data: dict[str, Tensor],
    n_topics: int,
    priors: dict[str, dict[str, Tensor]],
    modalities: list[str],
    feature_dims: dict[str, int],
    protein_likelihood: str,
    smart_init_mod: str,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    """NMF-seeded initialisation (same idea as Gibbs_LDA_Multi).

    Parameters
    ----------
    dense_data : dict[str, Tensor]
        Dense (C x Gm) matrices for *every* modality (including chromatin).
    """
    from sklearn.decomposition import NMF

    X_mod = dense_data[smart_init_mod]
    if isinstance(X_mod, Tensor):
        X_np = X_mod.cpu().float().numpy()
    else:
        X_np = np.asarray(X_mod, dtype=np.float32)

    nmf = NMF(n_components=n_topics, max_iter=300)
    W = nmf.fit_transform(X_np)

    theta = torch.tensor(W, dtype=torch.float32, device=device)
    theta = theta / (theta.sum(1, keepdim=True) + _EPS)

    lambda_dict: dict[str, Tensor] = {}
    for m in modalities:
        n_k_f = theta.T @ dense_data[m].to(device).float()  # K × Gm
        if m == "rna" or (m == "protein" and protein_likelihood == "nb"):
            gamma0 = priors[m]["gamma"]
            tau0 = priors[m]["tau"]
            lam = (gamma0 + n_k_f) / (tau0 + theta.sum(0).unsqueeze(1))
            lambda_dict[m] = lam
        else:
            beta0 = priors[m]["beta"]  # length Gm
            phi = beta0.unsqueeze(0) + n_k_f
            phi = phi / (phi.sum(1, keepdim=True) + _EPS)
            lambda_dict[m] = phi

    return theta, lambda_dict


# ---------------------------------------------------------------------------
# D.  Burn-in monitor
# ---------------------------------------------------------------------------


class BurninMonitor:
    """Track log-likelihood values and detect thermalization."""

    def __init__(
        self,
        window: int = 20,
        rtol: float = 1e-4,
        patience: int = 3,
        min_iters: int = 50,
    ):
        self.window = window
        self.rtol = rtol
        self.patience = patience
        self.min_iters = min_iters
        self._buf: deque[float] = deque(maxlen=2 * window)
        self._pass_count = 0
        self._n_updates = 0
        self._converged = False

    def update(self, ll: float) -> None:
        self._buf.append(ll)
        self._n_updates += 1
        if self._n_updates < self.min_iters:
            return
        if len(self._buf) < 2 * self.window:
            return
        buf = list(self._buf)
        old_mean = np.mean(buf[: self.window])
        new_mean = np.mean(buf[self.window :])
        rel_change = abs(new_mean - old_mean) / (abs(old_mean) + _EPS)
        if rel_change < self.rtol:
            self._pass_count += 1
        else:
            self._pass_count = 0
        if self._pass_count >= self.patience:
            self._converged = True

    @property
    def converged(self) -> bool:
        return self._converged


# ---------------------------------------------------------------------------
# E.  Per-modality expected-count and resampling helpers
# ---------------------------------------------------------------------------


def _exp_counts_poisson(
    x: Tensor,       # B × G
    theta: Tensor,   # B × K
    lam: Tensor,     # K × G
) -> tuple[Tensor, Tensor]:
    """Expected topic allocations under a Poisson (Gamma-Poisson) likelihood."""
    rate = theta @ lam + _EPS   # B × G
    p = (theta.unsqueeze(2) * lam.unsqueeze(0)) / rate.unsqueeze(1)   # B × K × G
    exp_n = x.unsqueeze(1) * p  # B × K × G
    return exp_n.sum(2), exp_n.sum(0)  # n_ck (B×K), n_kg (K×G)


def _exp_counts_multinomial(
    x: Tensor,       # B × P
    theta: Tensor,   # B × K
    phi: Tensor,     # K × P
) -> tuple[Tensor, Tensor]:
    """Expected topic allocations under a Multinomial (Dirichlet) likelihood."""
    prob = theta @ phi + _EPS   # B × P
    p = (theta.unsqueeze(2) * phi.unsqueeze(0)) / prob.unsqueeze(1)   # B × K × P
    exp_n = x.unsqueeze(1) * p  # B × K × P
    return exp_n.sum(2), exp_n.sum(0)


def _resample_gamma(suff: Tensor, gamma: Tensor, tau: Tensor, theta_sum: Tensor) -> Tensor:
    """Gamma posterior for rate parameters (RNA / protein-NB)."""
    shape = gamma + suff                  # K × G
    rate = tau + theta_sum.unsqueeze(1)   # K × 1  broadcast
    return Gamma(shape, rate).sample()


def _resample_dirichlet(suff: Tensor, beta: Tensor) -> Tensor:
    """Dirichlet posterior for proportion parameters (chromatin / protein-multinomial)."""
    alpha_post = beta.unsqueeze(0) + suff  # K × P
    return Dirichlet(alpha_post).sample()


# ---------------------------------------------------------------------------
# E2. Sparse ATAC batch update  (ported from SHARE-Topic _gibbs.py)
# ---------------------------------------------------------------------------


def _atac_sparse_batch(
    theta_batch: Tensor,     # B × K
    phi_atac: Tensor,        # K × R
    n_topics: int,
    batch_size: int,
    # batching metadata (sliced for this cell batch)
    c_lo: int, c_hi: int,
    rep_c: Tensor,           # region_counts_per_cell  (full, on device)
    rep_c_expanded: Tensor,  # cell_indices_expanded   (full, on device)
    acb_lo: int, acb_hi: int,
    regions_slice: Tensor,
    region_rep_slice: Tensor,
    region_rep_expanded_slice: Tensor,
    sort_indices_slice: Tensor,
    unsort_indices_slice: Tensor,
    device: torch.device,
    t_range: Tensor,         # arange(n_topics) on device, shape (K, 1)
) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
    """Sparse Bernoulli ATAC update for one cell batch.

    Returns
    -------
    n_ck_atac : (B, K)  topic counts per cell from ATAC
    atac_suff : (regions_slice, n_kr_local) or None
        regions_slice : (n_unique,) region indices into R
        n_kr_local    : (n_unique, K) region-topic counts
    """
    B = c_hi - c_lo
    if B == 0 or acb_lo == acb_hi:
        return torch.zeros(B, n_topics, device=device), None

    # Expand theta by number of ATAC entries per cell
    z_atac = torch.repeat_interleave(
        theta_batch, rep_c[c_lo:c_hi], dim=0,
    )  # (nnz_batch, K)

    # Expand phi for the relevant regions
    phi_ = phi_atac[:, regions_slice]                                  # K × n_unique
    phi_ = torch.repeat_interleave(phi_, region_rep_slice, dim=1)      # K × nnz_sorted
    phi_ = torch.index_select(phi_, 1, unsort_indices_slice)           # K × nnz_batch

    # z ∝ theta * phi  (element-wise per ATAC entry)
    z_atac = z_atac.T * phi_                                           # K × nnz_batch
    z_atac = z_atac / (z_atac.sum(0, keepdim=True) + _EPS)

    # Sample topic assignment via inverse-CDF
    z_atac_cum = torch.cumsum(z_atac, dim=0)
    u = torch.rand(1, z_atac.shape[1], device=device)
    z_idx = torch.searchsorted(
        z_atac_cum.T.contiguous(), u.T.contiguous(),
    ).contiguous()                                                     # (nnz_batch, 1)

    # One-hot encode
    z_onehot = (z_idx == t_range.T).int()                              # (nnz_batch, K)

    # --- Accumulate n_ck for this batch (scatter_add by cell) ---
    h_cell = rep_c_expanded[acb_lo:acb_hi] % batch_size                # local cell idx
    n_ck_atac = torch.zeros(B, n_topics, device=device)
    n_ck_atac.scatter_add_(0, h_cell.unsqueeze(1).expand_as(z_onehot), z_onehot.float())

    # --- Accumulate n_kr for this batch (scatter_add by region) ---
    z_sorted = torch.index_select(z_onehot, 0, sort_indices_slice)
    n_unique = regions_slice.shape[0]
    n_kr_local = torch.zeros(n_unique, n_topics, device=device)
    n_kr_local.scatter_add_(
        0,
        region_rep_expanded_slice.unsqueeze(1).expand_as(z_sorted),
        z_sorted.float(),
    )

    return n_ck_atac, (regions_slice, n_kr_local)


# ---------------------------------------------------------------------------
# F.  Log-likelihood
# ---------------------------------------------------------------------------


def compute_log_likelihood(
    dense_data: dict[str, Tensor],
    theta: Tensor,
    lambda_dict: dict[str, Tensor],
    modalities: list[str],
    protein_likelihood: str,
) -> dict[str, float]:
    """Joint log-likelihood per modality + total.

    Parameters
    ----------
    dense_data : dict[str, Tensor]
        Dense (C × Gm) matrices for *all* modalities (including chromatin).
    """
    ll_per_mod: dict[str, float] = {}

    for m in modalities:
        lam = lambda_dict[m]
        X = dense_data[m]
        if m == "rna" or (m == "protein" and protein_likelihood == "nb"):
            rate = theta @ lam + _EPS
            ll = torch.sum(X * torch.log(rate) - rate).item()
        elif m == "chromatin":
            # Bernoulli LL on binary data
            prob = (theta @ lam).clamp(_EPS, 1.0 - _EPS)
            ll = torch.sum(X * torch.log(prob) + (1 - X) * torch.log(1 - prob)).item()
        else:
            # protein multinomial
            prob = theta @ lam + _EPS
            ll = torch.sum(X * torch.log(prob)).item()
        ll_per_mod[m] = ll

    ll_per_mod["total"] = sum(ll_per_mod.values())
    return ll_per_mod


# ---------------------------------------------------------------------------
# G.  Main Gibbs sampler loop
# ---------------------------------------------------------------------------


def run_gibbs_sampler(
    dense_data: dict[str, Tensor],
    atac_coords: Tensor | None,
    modalities: list[str],
    n_topics: int,
    feature_dims: dict[str, int],
    # priors
    alpha_vec: Tensor,
    priors: dict[str, dict[str, Tensor]],
    protein_likelihood: str,
    # ATAC batching (may be None when chromatin absent)
    cell_batches: CellBatches | None,
    region_batches: RegionBatches | None,
    # initial parameters
    theta: Tensor,
    lambda_dict: dict[str, Tensor],
    device: torch.device,
    # sampling config
    n_samples: int = 500,
    thin: int = 10,
    burnin: int = 1_000,
    initial_burnin: int = 500,
    batch_size: int = 512,
    progress: bool = True,
    ll_every: int = 1,
    # auto burn-in
    auto_burnin: bool = True,
    min_initial_burnin: int = 50,
    burnin_window: int = 20,
    burnin_rtol: float = 1e-4,
    burnin_patience: int = 3,
) -> dict:
    """Run the full Gibbs sampler and return posterior samples.

    Parameters
    ----------
    dense_data : dict[str, Tensor]
        Dense (C × Gm) count matrices for every modality on *device*.
        Used for dense-modality Gibbs updates (RNA, protein) and for LL.
    atac_coords : Tensor or None
        Sparse ``[2, nnz]`` coordinate tensor for chromatin.  Only needed
        when ``"chromatin"`` is in *modalities*.

    Returns
    -------
    dict with keys:
        theta_samples       : (S, C, K)
        lambda_samples      : {mod: (S, K, Gm)}
        ll_history          : list[float]
        burnin_ll_trace     : list[float]
        burnin_converged_at : int | None
    """
    C = theta.shape[0]
    K = n_topics
    alpha_dev = alpha_vec.to(device)

    # Prepare ATAC batching tensors on device
    has_atac = "chromatin" in modalities and cell_batches is not None
    if has_atac:
        (c_bounds, rep_c, rep_c_exp, acb,
         regions_all, region_batching, region_rep_all, region_rep_exp_all,
         sort_idx_all, unsort_idx_all) = _move_batching_to_device(
            cell_batches, region_batches, device,
        )
        t_range = torch.arange(K, device=device).reshape(K, 1)
    else:
        c_bounds = acb = None  # not used

    # Dense modalities (everything except chromatin which uses sparse path)
    dense_mods = [m for m in modalities if m != "chromatin"]

    # ---------------------------------------------------------------
    # Helper: single Gibbs epoch
    # ---------------------------------------------------------------
    def gibbs_epoch():
        nonlocal theta
        n_ck = torch.zeros(C, K, device=device)
        suff_stats: dict[str, Tensor] = {
            m: torch.zeros_like(lambda_dict[m]) for m in modalities
        }

        # --- dense modalities (RNA, protein) via random mini-batches ---
        perm = torch.randperm(C, device=device)
        for start in range(0, C, batch_size):
            idx = perm[start: start + batch_size]
            theta_b = theta[idx]

            for m in dense_mods:
                X_b = dense_data[m][idx]
                lam = lambda_dict[m]

                if m == "rna" or (m == "protein" and protein_likelihood == "nb"):
                    n_ck_b, n_kg_b = _exp_counts_poisson(X_b, theta_b, lam)
                else:
                    n_ck_b, n_kg_b = _exp_counts_multinomial(X_b, theta_b, lam)

                n_ck[idx] += n_ck_b
                suff_stats[m] += n_kg_b

        # --- sparse ATAC (sequential cell-batches from SHARE-Topic) ---
        if has_atac:
            n_regions = feature_dims["chromatin"]
            for i in range(1, c_bounds.shape[0]):
                c_lo = int(c_bounds[i - 1].item())
                c_hi = int(c_bounds[i].item())
                if c_hi == c_lo:
                    continue

                acb_lo = int(acb[i - 1].item())
                acb_hi = int(acb[i].item())
                rb_lo = int(region_batching[i - 1].item())
                rb_hi = int(region_batching[i].item())

                n_ck_atac, atac_suff = _atac_sparse_batch(
                    theta[c_lo:c_hi],
                    lambda_dict["chromatin"],
                    K, batch_size,
                    c_lo, c_hi,
                    rep_c, rep_c_exp,
                    acb_lo, acb_hi,
                    regions_all[rb_lo:rb_hi],
                    region_rep_all[rb_lo:rb_hi],
                    region_rep_exp_all[acb_lo:acb_hi],
                    sort_idx_all[acb_lo:acb_hi],
                    unsort_idx_all[acb_lo:acb_hi],
                    device, t_range,
                )
                n_ck[c_lo:c_hi] += n_ck_atac
                if atac_suff is not None:
                    reg_idx, n_kr_local = atac_suff
                    # suff_stats["chromatin"] is K × R
                    # n_kr_local is n_unique × K  →  transpose to K × n_unique
                    suff_stats["chromatin"].index_add_(1, reg_idx, n_kr_local.T)

        # --- Resample lambda/phi ---
        theta_sum = theta.sum(0)  # (K,)
        for m in modalities:
            if m == "rna" or (m == "protein" and protein_likelihood == "nb"):
                lambda_dict[m] = _resample_gamma(
                    suff_stats[m], priors[m]["gamma"], priors[m]["tau"], theta_sum,
                )
            else:
                lambda_dict[m] = _resample_dirichlet(suff_stats[m], priors[m]["beta"])

        # --- Resample theta ---
        alpha_post = alpha_dev + n_ck
        theta = Dirichlet(alpha_post).sample()

    # ---------------------------------------------------------------
    # PHASE 1:  Initial burn-in (with optional auto-detection)
    # ---------------------------------------------------------------
    monitor = BurninMonitor(
        window=burnin_window, rtol=burnin_rtol,
        patience=burnin_patience, min_iters=min_initial_burnin,
    ) if auto_burnin else None
    burnin_ll_trace: list[float] = []
    burnin_converged_at: int | None = None

    iterator = range(initial_burnin)
    if progress:
        iterator = tqdm(iterator, total=initial_burnin, desc="Burn-in (initial)")
    for it in iterator:
        gibbs_epoch()
        if it % ll_every == 0:
            ll = compute_log_likelihood(
                dense_data, theta, lambda_dict, modalities, protein_likelihood,
            )["total"]
            burnin_ll_trace.append(ll)
            if progress:
                iterator.set_postfix({"ll": f"{ll:.2e}"})
            if monitor is not None:
                monitor.update(ll)
                if monitor.converged:
                    burnin_converged_at = it
                    if progress:
                        tqdm.write(f"Burn-in converged at iteration {it}")
                    break

    # ---------------------------------------------------------------
    # PHASE 2:  Post-initial-burnin discard
    # ---------------------------------------------------------------
    if burnin > 0:
        iterator2 = range(burnin)
        if progress:
            iterator2 = tqdm(iterator2, total=burnin, desc="Burn-in (discard)")
        for _ in iterator2:
            gibbs_epoch()

    # ---------------------------------------------------------------
    # PHASE 3:  Posterior sampling
    # ---------------------------------------------------------------
    S = n_samples
    total_sampling_iters = n_samples * thin
    theta_samps = torch.empty(S, C, K, dtype=torch.float32)
    lambda_samps: dict[str, Tensor] = {
        m: torch.empty(S, K, feature_dims[m], dtype=torch.float32)
        for m in modalities
    }
    ll_history: list[float] = []
    store_idx = 0

    sampling_iter = range(total_sampling_iters)
    if progress:
        sampling_iter = tqdm(sampling_iter, total=total_sampling_iters, desc="Sampling")
    for it in sampling_iter:
        gibbs_epoch()

        if it % ll_every == 0:
            ll = compute_log_likelihood(
                dense_data, theta, lambda_dict, modalities, protein_likelihood,
            )["total"]
            ll_history.append(ll)
            if progress:
                sampling_iter.set_postfix({"ll": f"{ll:.2e}"})

        if it % thin == 0:
            theta_samps[store_idx] = theta.cpu()
            for m in modalities:
                lambda_samps[m][store_idx] = lambda_dict[m].cpu()
            store_idx += 1

    return {
        "theta_samples": theta_samps,
        "lambda_samples": lambda_samps,
        "ll_history": ll_history,
        "burnin_ll_trace": burnin_ll_trace,
        "burnin_converged_at": burnin_converged_at,
    }
