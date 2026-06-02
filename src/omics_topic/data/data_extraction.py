"""Data extraction and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData


def extract_from_mudata(
    mdata: MuData,
    modalities: list[str] | None = None,
    layers: dict[str, str | None] | str | None = None,
    spatial_keys: dict[str, str] | str | None = None,
) -> tuple[AnnData, dict]:
    """
    Extract and preprocess data from MuData.

    This function extracts modality-specific data from MuData, with support for:
    - Layer selection (per-modality or global)
    - Spatial graph extraction
    - Modality subsetting

    Parameters
    ----------
    mdata : MuData
        Input MuData object containing multiple modalities.
    modalities : list[str] | None
        Subset of modalities to use. If None, uses all modalities in mdata.mod.
    layers : dict[str, str | None] | str | None
        Layer specification:
        - dict: Per-modality layer specification, e.g. {"rna": "counts", "protein": None}
        - str: Same layer name for all modalities, e.g. "counts"
        - None: Use .X for all modalities (default)
    spatial_keys : dict[str, str] | str | None
        Spatial graph keys in .obsp:
        - dict: Per-modality spatial graph keys, e.g. {"rna": "spatial_connectivities"}
        - str: Same spatial key for all modalities
        - None: No spatial graphs (default)

    Returns
    -------
    adata_concat : AnnData
        Concatenated AnnData with features from all modalities.
        Features are concatenated horizontally in the order of `modalities`.
    metadata : dict
        Dictionary containing:
        - modality_names: list[str] - names of modalities in concatenation order
        - feature_counts: list[int] - feature counts per modality
        - spatial_info: dict | None - spatial graph information
        - layer_dict: dict[str, str | None] - layer used for each modality

    Examples
    --------
    >>> # Extract all modalities with default .X
    >>> adata, meta = extract_from_mudata(mdata)

    >>> # Extract specific modalities with layer selection
    >>> adata, meta = extract_from_mudata(
    ...     mdata,
    ...     modalities=["rna", "protein"],
    ...     layers={"rna": "counts", "protein": None}
    ... )

    >>> # Same layer for all modalities
    >>> adata, meta = extract_from_mudata(mdata, layers="counts")

    >>> # With spatial graphs
    >>> adata, meta = extract_from_mudata(
    ...     mdata,
    ...     layers="counts",
    ...     spatial_keys="spatial_connectivities"
    ... )
    """
    # Normalize modalities
    if modalities is None:
        modalities = list(mdata.mod.keys())

    # Normalize layers to dict
    if isinstance(layers, str):
        layer_dict = {mod: layers for mod in modalities}
    elif layers is None:
        layer_dict = {mod: None for mod in modalities}
    else:
        layer_dict = layers

    # Normalize spatial_keys to dict
    if isinstance(spatial_keys, str):
        spatial_dict = {mod: spatial_keys for mod in modalities}
    elif spatial_keys is None:
        spatial_dict = {}
    else:
        spatial_dict = spatial_keys

    # Extract data
    matrices = []
    feat_counts = []
    var_names = []
    n_cells_ref = mdata.n_obs

    for mod in modalities:
        if mod not in mdata.mod:
            raise ValueError(
                f"Modality '{mod}' not found in MuData. " f"Available: {list(mdata.mod.keys())}"
            )

        adata_mod = mdata.mod[mod]

        # Validate cell counts
        if adata_mod.n_obs != n_cells_ref:
            raise ValueError(
                f"Modality '{mod}' has {adata_mod.n_obs} cells, "
                f"but MuData has {n_cells_ref} cells. All modalities must be aligned."
            )

        # Extract from layer if specified, otherwise use .X
        layer_key = layer_dict.get(mod)
        if layer_key is not None:
            if layer_key not in adata_mod.layers:
                raise KeyError(
                    f"Layer '{layer_key}' not found in modality '{mod}'. "
                    f"Available layers: {list(adata_mod.layers.keys())}"
                )
            X = adata_mod.layers[layer_key]
        else:
            X = adata_mod.X

        # Convert to appropriate format
        if sp.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        matrices.append(X)
        feat_counts.append(X.shape[1])
        var_names.extend(adata_mod.var_names)

    # Concatenate matrices
    if any(sp.issparse(M) for M in matrices):
        X_concat = sp.hstack(matrices, format="csr")
    else:
        X_concat = np.hstack(matrices)

    # Create concatenated AnnData
    adata_concat = AnnData(X_concat, obs=mdata.obs.copy())
    adata_concat.var_names = var_names

    # Extract spatial graphs if specified
    spatial_info = _extract_spatial_graphs(mdata, modalities, spatial_dict)

    # Build metadata
    metadata = {
        "modality_names": modalities,
        "feature_counts": feat_counts,
        "spatial_info": spatial_info,
        "layer_dict": layer_dict,
    }

    return adata_concat, metadata


def extract_from_adata_dict(
    adata_dict: dict[str, AnnData],
    layers: dict[str, str | None] | str | None = None,
    spatial_keys: dict[str, str] | str | None = None,
) -> tuple[AnnData, dict]:
    """
    Extract and preprocess data from dict of AnnData objects.

    Converts dict → MuData → uses extract_from_mudata()

    Parameters
    ----------
    adata_dict : dict[str, AnnData]
        Dictionary mapping modality names to AnnData objects.
    layers : dict[str, str | None] | str | None
        Layer specification (same as extract_from_mudata).
    spatial_keys : dict[str, str] | str | None
        Spatial graph keys (same as extract_from_mudata).

    Returns
    -------
    adata_concat : AnnData
        Concatenated AnnData.
    metadata : dict
        Metadata dictionary.

    Examples
    --------
    >>> adata_dict = {"rna": adata_rna, "protein": adata_protein}
    >>> adata, meta = extract_from_adata_dict(
    ...     adata_dict,
    ...     layers={"rna": "counts"}
    ... )
    """
    # Create MuData from dict
    mdata = MuData(adata_dict)
    modalities = list(adata_dict.keys())

    return extract_from_mudata(mdata, modalities, layers, spatial_keys)


def extract_from_anndata(
    adata: AnnData,
    modality_name: str = "rna",
    layer: str | None = None,
    spatial_key: str | None = None,
) -> tuple[AnnData, dict]:
    """
    Extract and preprocess data from single AnnData (single modality).

    For single modality, we can use the AnnData directly,
    but need to format metadata consistently.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    modality_name : str
        Name to assign to this modality (default: "rna").
    layer : str | None
        Layer to extract. If None, uses .X.
    spatial_key : str | None
        Spatial graph key in .obsp. If None, no spatial graph.

    Returns
    -------
    adata_processed : AnnData
        Processed AnnData (with layer extracted to .X if specified).
    metadata : dict
        Metadata dictionary.

    Examples
    --------
    >>> adata, meta = extract_from_anndata(
    ...     adata,
    ...     modality_name="rna",
    ...     layer="counts",
    ...     spatial_key="spatial_connectivities"
    ... )
    """
    # Extract from layer if specified
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found. " f"Available layers: {list(adata.layers.keys())}"
            )
        # Reuse the same AnnData object but move the selected layer into .X
        adata.X = adata.layers[layer]
    adata_processed = adata

    # Extract spatial graph if specified
    from omics_topic.utils.amortized_utils import _resolve_spatial_graph_from_adata

    spatial_info = _resolve_spatial_graph_from_adata(adata_processed, spatial_key)

    metadata = {
        "modality_names": [modality_name],
        "feature_counts": [adata_processed.n_vars],
        "spatial_info": spatial_info,
        "layer_dict": {modality_name: layer} if layer else {},
    }

    return adata_processed, metadata


def extract_from_spatialdata(
    sdata,  # SpatialData type
    table_key: str = "table",
    modalities: list[str] | None = None,
    layers: dict[str, str | None] | str | None = None,
    spatial_key: str | None = None,
) -> tuple[AnnData, dict]:
    """
    Extract and preprocess data from SpatialData.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object.
    table_key : str
        Which table to extract from sdata.tables (default: "table").
    modalities : list[str] | None
        Modalities to extract (if table is MuData-like).
    layers : dict[str, str | None] | str | None
        Layer specification.
    spatial_key : str | None
        Spatial graph key in the table.

    Returns
    -------
    adata_concat : AnnData
        Concatenated AnnData.
    metadata : dict
        Metadata dictionary.

    Examples
    --------
    >>> adata, meta = extract_from_spatialdata(
    ...     sdata,
    ...     table_key="table",
    ...     layers="counts",
    ...     spatial_key="spatial"
    ... )
    """
    # Extract table from SpatialData
    if table_key not in sdata.tables:
        raise KeyError(
            f"Table '{table_key}' not found in SpatialData. "
            f"Available tables: {list(sdata.tables.keys())}"
        )

    table = sdata.tables[table_key]

    # Check if table is AnnData or MuData
    try:
        is_mudata = isinstance(table, MuData)
    except NameError:
        # MuData not imported
        is_mudata = False

    if is_mudata:
        return extract_from_mudata(table, modalities, layers, spatial_key)
    elif isinstance(table, AnnData):
        modality_name = modalities[0] if modalities else "rna"
        layer = layers if isinstance(layers, str) else None
        return extract_from_anndata(table, modality_name, layer, spatial_key)
    else:
        raise TypeError(
            f"Table '{table_key}' is type {type(table)}, " "expected AnnData or MuData"
        )


def _extract_spatial_graphs(
    mdata: MuData,
    modalities: list[str],
    spatial_dict: dict[str, str],
) -> dict | None:
    """
    Extract spatial graphs from MuData modalities.

    Parameters
    ----------
    mdata : MuData
        Input MuData object.
    modalities : list[str]
        List of modality names.
    spatial_dict : dict[str, str]
        Mapping of modality names to spatial graph keys in .obsp.

    Returns
    -------
    dict | None
        Dictionary mapping modality names to spatial graph info dicts,
        or None if no spatial graphs found.
        Each spatial graph info dict contains:
        - adjacency: sparse matrix
        - key: str - the obsp key used
    """
    # Import here to avoid circular dependency
    from omics_topic.utils.amortized_utils import _resolve_spatial_graph_from_adata

    if not spatial_dict:
        return None

    spatial_graphs = {}
    for mod in modalities:
        spatial_key = spatial_dict.get(mod)
        if spatial_key:
            adata_mod = mdata.mod[mod]
            graph_info = _resolve_spatial_graph_from_adata(adata_mod, spatial_key)
            if graph_info:
                spatial_graphs[mod] = graph_info

    return spatial_graphs if spatial_graphs else None
