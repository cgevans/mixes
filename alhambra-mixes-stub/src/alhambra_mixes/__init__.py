"""
DEPRECATED: The 'alhambra-mixes' package has been renamed to 'riverine'.

Please update your dependencies and imports:
    - Install: pip install riverine (or uv add riverine)
    - Import: import riverine instead of import alhambra

This stub package will be maintained for a transition period but may be
removed in the future.
"""

import warnings

warnings.warn(
    "The 'alhambra-mixes' package has been renamed to 'riverine'. "
    "Please update your code to 'import riverine' instead of 'import alhambra'. "
    "This compatibility package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

import riverine
from riverine import *

try:
    from riverine import __version__
except ImportError:
    __version__ = "unknown"

if hasattr(riverine, '__all__'):
    __all__ = riverine.__all__

