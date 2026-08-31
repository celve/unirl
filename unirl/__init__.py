"""Canonical Python package for this repository; import from the defining modules, this package re-exports nothing."""

try:
    from unirl._version import __version__, __version_tuple__
except ImportError:  # source tree that has never been built
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")
