"""Compatibility shim for imbalanced-learn on top of Sklong's sklearn fork.

`scikit-lexicographical-trees` (the sklearn fork that scikit-longitudinal
ships) reports ``sklearn.__version__ == "0.0.5"``. ``imbalanced-learn`` gates
many private helper imports on the parsed sklearn version, so it picks the
pre-1.5 import paths even though the fork mirrors a newer sklearn layout
(helpers live in ``sklearn.utils._indexing``, ``sklearn.utils._user_interface``
etc.). Forcing the reported version to a recent value before imblearn is
imported makes both libraries cohabit cleanly without forking either.
"""

from __future__ import annotations

# Sklong's sklearn fork is API-compatible with sklearn ~1.5; advertising 1.5.0
# steers imbalanced-learn (0.12.x and 0.13+) into its modern import branches.
_FAKE_SKLEARN_VERSION = "1.5.0"


def _patch_sklearn_version() -> None:
    try:
        import sklearn
    except ImportError:
        return
    if sklearn.__version__ == _FAKE_SKLEARN_VERSION:
        return
    if not sklearn.__version__.startswith("0."):
        return
    sklearn.__version__ = _FAKE_SKLEARN_VERSION


_patch_sklearn_version()
