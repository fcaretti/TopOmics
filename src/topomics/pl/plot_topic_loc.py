from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt

# import textwrap
import numpy as np
import scanpy as sc

# import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes

# import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# import pandas as pd
from matplotlib.patches import Patch

# from upsetplot import plot, from_contents
# from itertools import chain
from scanpy._utils import Empty, _empty
from scanpy.pl._tools.scatterplots import (
    _check_crop_coord,
    _check_img,
    _check_na_color,
    _check_scale_factor,
    _check_spatial_data,
    _check_spot_size,
)
from scanpy.pl._utils import (
    ColorLike,
)


def spatial(
    adata,
    color=None,
    cmap=None,
    frameon=None,
    title=None,
    wspace=None,
    hspace=0.25,
    palette=None,
    colorbar_loc="right",
    size=1,
    basis="spatial",
    vmax=None,
    ncols=4,
    layer=None,
    show=True,
    *args,
    **kwargs,
):
    """A faster simple function that uses sc.pl.embedding to plot for non-visium data
    so it dont take too long. ~sleep. Very inflexible.

    Args:
        adata (_type_): Annotated data matrix.
        color (_type_): Keys for annotations of observations/cells or variables/genes
        size (int, optional): size of spots. Defaults to 1.
        basis (str, optional): basis in obsm. Defaults to "spatial".
        vmax (str, optional): The value representing the upper limit of the color scale. Defaults to "p99".
        show (bool, optional): Show the plot, do not return axis. Defaults to True.

    Returns:
        _type_: A plot
    """
    ax = sc.pl.embedding(
        adata,
        basis=basis,
        show=False,
        color=color,
        wspace=wspace,
        hspace=hspace,
        palette=palette,
        vmax=vmax,
        size=size,
        ncols=ncols,
        cmap=cmap,
        frameon=frameon,
        colorbar_loc=colorbar_loc,
        title=title,
        layer=layer,
        *args,
        **kwargs,
    )
    if isinstance(ax, list):
        [axs.invert_yaxis() for axs in ax]
        [axs.set_aspect("equal") for axs in ax]
    else:
        ax.invert_yaxis()
        ax.set_aspect("equal")
    if show is False:
        return ax