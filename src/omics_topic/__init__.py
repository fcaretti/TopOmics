from importlib.metadata import version

from . import models, pl, pp, tl
from .models import BaseTopicModel, MultimodalAmortizedLDA, ShareTopic_LDA_Multi, SVEM_LDA_Multi

__all__ = ["pl", "pp", "tl", "models", "SVEM_LDA_Multi", "BaseTopicModel", "MultimodalAmortizedLDA", "ShareTopic_LDA_Multi"]

__version__ = version("omics-topic")
