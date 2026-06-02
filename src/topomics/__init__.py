from importlib.metadata import PackageNotFoundError, version

from . import models, pl, pp, tl
from .models import BaseTopicModel, MultimodalAmortizedLDA, ShareTopic_LDA_Multi, SVEM_LDA_Multi

__all__ = ["pl", "pp", "tl", "models", "SVEM_LDA_Multi", "BaseTopicModel", "MultimodalAmortizedLDA", "ShareTopic_LDA_Multi"]

# Package was renamed from "omics-topic" -> "topomics"; tolerate stale editable installs.
try:
    __version__ = version("topomics")
except PackageNotFoundError:
    try:
        __version__ = version("omics-topic")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
