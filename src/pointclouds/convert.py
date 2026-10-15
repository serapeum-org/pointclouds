"""The pyramids boundary: importing pyramids happens here and nowhere else.

This is the ONLY module in the package permitted to import ``pyramids``, and it does so lazily inside
:func:`require_pyramids` behind a guarded import. Importing this module with the core dependency set alone must
succeed; only *calling into* the GIS stack requires the ``[gis]`` extra.

The converters themselves - ``to_dataset`` (grid a cloud to a pyramids ``Dataset``) and ``to_feature_collection``
(a decimated cloud to a point ``FeatureCollection``) - land in Phase 2 (plan task T-10), on top of the core
``ops.grid.to_grid``. They are deliberately absent here rather than shipped as raising stubs.
"""

from __future__ import annotations

from typing import Any

__all__ = ["require_pyramids"]

_GIS_HINT = "This needs the optional GIS stack (pyramids-gis). Install it with: pip install pointclouds[gis]"


def require_pyramids() -> tuple[Any, Any]:
    """Return ``(Dataset, GeoReference)`` from pyramids, or raise a hint naming the extra.

    Returns:
        The ``pyramids.dataset.Dataset`` and ``GeoReference`` classes.

    Raises:
        ModuleNotFoundError: when pyramids is not installed, with a message that names
            ``pip install pointclouds[gis]`` rather than surfacing the bare import error.
    """
    try:
        from pyramids.dataset import Dataset, GeoReference
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ModuleNotFoundError(_GIS_HINT) from exc
    return Dataset, GeoReference
