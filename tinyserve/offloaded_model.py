"""Deprecated: use tinyserve._model_hooks instead."""
import warnings

warnings.warn(
    "tinyserve.offloaded_model is deprecated, use tinyserve._model_hooks",
    DeprecationWarning,
    stacklevel=2,
)
from tinyserve._model_hooks import *  # noqa: F401,F403
