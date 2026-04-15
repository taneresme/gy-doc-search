"""gy-doc-search package."""

from importlib.metadata import PackageNotFoundError, version as distribution_version

__all__ = ["__version__"]

_DIST_NAME = "gy-doc-search"
_FALLBACK_VERSION = "0.1.2"

try:
    __version__ = distribution_version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = _FALLBACK_VERSION
