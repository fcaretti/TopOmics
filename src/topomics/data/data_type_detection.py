"""Type detection utilities for flexible data input."""

from __future__ import annotations

from anndata import AnnData


def detect_data_type(data) -> str:
    """
    Detect the type of input data.

    Parameters
    ----------
    data
        Input data to detect type for.

    Returns
    -------
    str
        One of: "anndata", "mudata", "spatialdata", "dict", "unknown"

    Examples
    --------
    >>> from anndata import AnnData
    >>> import numpy as np
    >>> adata = AnnData(np.random.rand(10, 20))
    >>> detect_data_type(adata)
    'anndata'
    """
    # Check for dict first (most specific)
    if isinstance(data, dict):
        # Check if it's a dict of AnnData objects
        if all(isinstance(v, AnnData) for v in data.values()):
            return "dict"
        return "unknown"

    # Check for AnnData
    if isinstance(data, AnnData):
        return "anndata"

    # Check for MuData (conditional import to avoid hard dependency)
    try:
        from mudata import MuData

        if isinstance(data, MuData):
            return "mudata"
    except ImportError:
        pass

    # Check for SpatialData (conditional import)
    try:
        from spatialdata import SpatialData

        if isinstance(data, SpatialData):
            return "spatialdata"
    except ImportError:
        pass

    return "unknown"


def validate_data_type(data) -> None:
    """
    Validate that data is a supported type, raise clear error if not.

    Parameters
    ----------
    data
        Input data to validate.

    Raises
    ------
    TypeError
        If data type is not supported.

    Examples
    --------
    >>> from anndata import AnnData
    >>> import numpy as np
    >>> adata = AnnData(np.random.rand(10, 20))
    >>> validate_data_type(adata)  # No error
    >>> validate_data_type("invalid")  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: Unsupported data type: <class 'str'>. Supported types: AnnData, MuData, SpatialData, or dict[str, AnnData]
    """
    data_type = detect_data_type(data)
    if data_type == "unknown":
        raise TypeError(
            f"Unsupported data type: {type(data)}. Supported types: AnnData, MuData, SpatialData, or dict[str, AnnData]"
        )
