from importlib.metadata import version

from . import models, pl, pp, tl
from .models import BaseModel, SVEM_LDA_Multi

__all__ = ["pl", "pp", "tl", "models", "SVEM_LDA_Multi", "BaseModel"]

__version__ = version("omics-topic")
