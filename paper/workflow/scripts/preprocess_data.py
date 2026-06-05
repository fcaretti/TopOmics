"""
Shared data loading and preprocessing for all datasets.

Every training script imports `load_dataset(dataset_name, data_dir, config)`
which returns the appropriate data object (MuData or AnnData) ready for model
creation, with counts layers, HVG filtering, spatial graphs, etc. applied.
"""

import os
import warnings

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(dataset_name, data_dir, datasets_config):
    """Load and preprocess a dataset by name.

    Returns
    -------
    data : MuData | AnnData
        Ready for model creation.
    """
    loader = _LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds_cfg = datasets_config[dataset_name]
    return loader(data_dir, ds_cfg)


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------


def _load_teaseq(data_dir, ds_cfg):
    import muon as mu

    path = os.path.join(data_dir, ds_cfg["data_path"])
    mdata = mu.read_h5mu(path)

    _ensure_counts(mdata.mod["rna"])
    _ensure_counts(mdata.mod["atac"])
    _ensure_counts(mdata.mod["prot"])

    # Binarize ATAC
    mdata.mod["atac"].layers["counts"] = (mdata.mod["atac"].layers["counts"] > 0).astype(int)

    # HVG filtering
    prep = ds_cfg.get("preprocessing", {})
    _filter_hvg(mdata.mod["rna"], prep.get("rna_hvgs", 2000))
    _filter_hvg(mdata.mod["atac"], prep.get("atac_hvgs", 10000))
    mdata.update()
    return mdata


def _load_atac_rna(data_dir, ds_cfg):
    import muon as mu

    path = os.path.join(data_dir, ds_cfg["data_path"])
    mdata = mu.read_h5mu(path)

    _ensure_counts(mdata.mod["rna"])
    _ensure_counts(mdata.mod["atac"])

    # Binarize ATAC
    mdata.mod["atac"].layers["counts"] = (mdata.mod["atac"].layers["counts"] > 0).astype(int)

    prep = ds_cfg.get("preprocessing", {})
    _filter_hvg(mdata.mod["rna"], prep.get("rna_hvgs", 2000))
    _filter_hvg(mdata.mod["atac"], prep.get("atac_hvgs", 20000))
    mdata.update()
    return mdata


def _load_visium(data_dir, ds_cfg):
    import squidpy as sq

    adata = sq.datasets.visium_hne_adata()
    adata.X = adata.raw.X
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic")
    return adata


def _load_mouse_brain_spatial(data_dir, ds_cfg):
    paths = ds_cfg["data_paths"]
    adata_rna = sc.read_h5ad(os.path.join(data_dir, paths["rna"]))
    adata_atac = sc.read_h5ad(os.path.join(data_dir, paths["atac"]))

    _ensure_counts(adata_rna)
    _binarize_atac(adata_atac)

    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    n_neighbors = ds_cfg.get("preprocessing", {}).get("n_neighbors", 5)
    _build_spatial_graph(mdata, "rna", list(mdata.mod.keys()), n_neighbors)
    return mdata


def _load_colon_citeseq(data_dir, ds_cfg):
    paths = ds_cfg["data_paths"]

    rna_df = pd.read_csv(os.path.join(data_dir, paths["rna"]), sep="\t", index_col=0)
    adata_rna = ad.AnnData(
        X=sp.csr_matrix(rna_df.values.astype(np.float32)),
        obs=pd.DataFrame(index=rna_df.index),
        var=pd.DataFrame(index=rna_df.columns),
    )
    adata_rna.layers["counts"] = adata_rna.X.copy()

    prot_df = pd.read_csv(os.path.join(data_dir, paths["protein"]), sep="\t", index_col=0)
    common = adata_rna.obs_names.intersection(prot_df.index)
    adata_rna = adata_rna[common].copy()
    prot_df = prot_df.loc[common]

    adata_prot = ad.AnnData(
        X=sp.csr_matrix(prot_df.values.astype(np.float32)),
        obs=pd.DataFrame(index=prot_df.index),
        var=pd.DataFrame(index=prot_df.columns),
    )
    adata_prot.layers["counts"] = adata_prot.X.copy()

    _filter_hvg(adata_rna, 2000, layer="counts")

    # Spatial coords from barcodes
    coords = np.array([list(map(int, b.split("x"))) for b in adata_rna.obs_names]).astype(np.float32)
    adata_rna.obsm["spatial"] = coords
    adata_prot.obsm["spatial"] = coords

    mdata = md.MuData({"rna": adata_rna, "prot": adata_prot})

    if ds_cfg.get("spatial", False):
        _build_spatial_graph(mdata, "rna", ["rna", "prot"], 5)

    return mdata


def _load_thymus(data_dir, ds_cfg):
    paths = ds_cfg["data_paths"]
    adata_rna = sc.read_h5ad(os.path.join(data_dir, paths["rna"]))
    adata_adt = sc.read_h5ad(os.path.join(data_dir, paths["adt"]))

    _ensure_counts(adata_rna)
    _ensure_counts(adata_adt)
    _to_sparse(adata_rna)
    _to_sparse(adata_adt)

    _filter_hvg(adata_rna, 2000, layer="counts")

    mdata = md.MuData({"rna": adata_rna, "adt": adata_adt})

    if ds_cfg.get("spatial", False):
        _build_spatial_graph(mdata, "rna", ["rna", "adt"], 5)

    return mdata


def _load_dbit_seq(data_dir, ds_cfg):
    import h5py

    path = os.path.join(data_dir, ds_cfg["data_path"])
    with h5py.File(path, "r") as f:
        X_gene = f["X_gene"][:].astype(np.float32)
        X_protein = f["X_protein"][:].astype(np.float32)
        cells = [c.decode() for c in f["cell"][:]]
        genes = [g.decode() for g in f["gene"][:]]
        proteins = [p.decode() for p in f["protein"][:]]
        pos = f["pos"][:].astype(np.float32)

    adata_rna = ad.AnnData(X=sp.csr_matrix(X_gene), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    adata_rna.layers["counts"] = adata_rna.X.copy()
    adata_rna.obsm["spatial"] = pos

    adata_prot = ad.AnnData(X=sp.csr_matrix(X_protein), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=proteins))
    adata_prot.layers["counts"] = adata_prot.X.copy()
    adata_prot.obsm["spatial"] = pos

    mdata = md.MuData({"rna": adata_rna, "prot": adata_prot})

    if ds_cfg.get("spatial", False):
        n_neighbors = ds_cfg.get("preprocessing", {}).get("n_neighbors", 5)
        _build_spatial_graph(mdata, "rna", ["rna", "prot"], n_neighbors)

    return mdata


def _load_gastrulation(data_dir, ds_cfg):
    import scvelo as scv

    adata = scv.datasets.gastrulation_erythroid()

    spliced = adata.layers["spliced"]
    unspliced = adata.layers["unspliced"]
    nonzero = (np.asarray(spliced.sum(0)).ravel() > 0) | (np.asarray(unspliced.sum(0)).ravel() > 0)
    adata = adata[:, nonzero].copy()

    adata_mature = adata.copy()
    adata_mature.X = adata.layers["spliced"]
    adata_mature.layers.clear()

    adata_immature = adata.copy()
    adata_immature.X = adata.layers["unspliced"]
    adata_immature.layers.clear()

    mdata = md.MuData({"mature": adata_mature, "immature": adata_immature})
    return mdata


def _load_retina(data_dir, ds_cfg):
    path = os.path.join(data_dir, ds_cfg["data_path"])
    adata = sc.read_h5ad(path)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_genes=200)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch", flavor="seurat_v3")
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_counts(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()


def _to_sparse(adata):
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
        adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])


def _binarize_atac(adata):
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.ones_like(X.data)
        X.eliminate_zeros()
        adata.layers["binary"] = X
        adata.layers["counts"] = X.copy()
    else:
        binary = (X != 0).astype(np.float32)
        adata.layers["binary"] = binary
        adata.layers["counts"] = binary.copy()


def _filter_hvg(adata, n_top_genes, layer=None):
    """Filter to highly variable genes with seurat_v3 fallback."""
    if layer is None:
        layer = "counts" if "counts" in adata.layers else None
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", layer=layer)
    except ValueError as e:
        if "near singularities" not in str(e):
            raise
        warnings.warn("seurat_v3 HVG failed; falling back to seurat.", RuntimeWarning)
        hvg_input = adata.copy()
        sc.pp.normalize_total(hvg_input, target_sum=1e4)
        sc.pp.log1p(hvg_input)
        sc.pp.highly_variable_genes(hvg_input, n_top_genes=n_top_genes, flavor="seurat")
        adata.var["highly_variable"] = hvg_input.var["highly_variable"].values


def _build_spatial_graph(mdata, ref_mod, all_mods, n_neighbors):
    """Build spatial neighbor graph from ref_mod and share to all modalities."""
    sc.pp.neighbors(
        mdata.mod[ref_mod],
        use_rep="spatial",
        n_neighbors=n_neighbors,
        metric="euclidean",
        key_added="spatial",
    )
    mdata.obsp["spatial_connectivities"] = mdata.mod[ref_mod].obsp["spatial_connectivities"]
    for mod_name in all_mods:
        if mod_name != ref_mod:
            mdata.mod[mod_name].obsp["spatial_connectivities"] = mdata.mod[ref_mod].obsp["spatial_connectivities"]
            mdata.mod[mod_name].obsp["spatial_distances"] = mdata.mod[ref_mod].obsp["spatial_distances"]


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

_LOADERS = {
    "teaseq": _load_teaseq,
    "atac_rna": _load_atac_rna,
    "visium": _load_visium,
    "mouse_brain_spatial": _load_mouse_brain_spatial,
    "colon_citeseq": _load_colon_citeseq,
    "thymus": _load_thymus,
    "dbit_seq": _load_dbit_seq,
    "gastrulation": _load_gastrulation,
    "retina": _load_retina,
}
