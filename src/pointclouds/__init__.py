"""pointclouds - point cloud (LAS/LAZ/COPC/EPT) I/O, filtering, decimation and gridding.

The public entry point is :class:`~pointclouds.points.PointCloud`. The GIS converters in
:mod:`pointclouds.convert` require the optional ``[gis]`` extra; importing the package needs the core
dependency set only (numpy / laspy / pyproj / scipy) and never GDAL.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from pointclouds.points import PointCloud

try:
    __version__ = _version("pointclouds")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["PointCloud", "__version__"]
