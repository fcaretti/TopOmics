"""Data / graph helpers for the :class:`MultimodalAmortizedLDA` model.

Factored out of ``omics_topic.models.amortizedLDA`` so they can be reused
without importing the full model class.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from mudata import MuData


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
