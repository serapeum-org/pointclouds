"""laspy-backed LAS/LAZ engine: header-first read, chunked record read, write. (Phase 1 - T-8).

Two entry points, deliberately separate, because the split is the point of the design:

- :func:`read_header` opens the file and returns metadata only. No point record is decompressed, so the cost is
  constant whatever the file's size - a 40 GB tile answers as fast as a 40 KB one.
- :func:`read_points` decompresses the record arrays, optionally only the dimensions asked for.

Everything here imports with the core dependency set (numpy, laspy, pyproj). Spatially-indexed windowed reads
live in :mod:`pointclouds.io.copc` (T-16) and writing lands with T-13; neither belongs in this module. The
record-array contract the two readers share is in :mod:`pointclouds.io._records`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

from pointclouds.io import _records

__all__ = ["LasHeaderInfo", "read_header", "read_points"]

_SUFFIXES = {".las", ".laz"}


@dataclass(frozen=True)
class LasHeaderInfo:
    """Everything the LAS/LAZ header can answer without touching a point record.

    Attributes:
        count: the header's point count. On a lazy cloud this is known before any record is read, which is why it
            is not simply ``len(cloud)``.
        bounds: ``(xmin, ymin, xmax, ymax)`` as recorded in the header.
        crs: the file's CRS as a :class:`pyproj.CRS`, or ``None`` when the header carries no SRS. Never a GDAL or
            pyramids object - that is the dependency boundary this package exists to keep.
        dimensions: every dimension name the point format declares, in file order.
        version: the LAS version string, e.g. ``"1.4"``.
        point_format: the point format id, e.g. ``3``.
    """

    count: int
    bounds: tuple[float, float, float, float]
    crs: CRS | None
    dimensions: tuple[str, ...]
    version: str
    point_format: int


def _resolve(path: str | Path) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path`, rejecting what this engine cannot read.

    Args:
        path: the LAS/LAZ file to open.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: when the path does not exist, naming the path rather than surfacing laspy's error.
        ValueError: when the suffix is neither ``.las`` nor ``.laz``.
    """
    resolved = Path(path)
    if resolved.suffix.lower() not in _SUFFIXES:
        raise ValueError(f"{resolved.name!r} is not a LAS/LAZ file; expected one of {sorted(_SUFFIXES)}")
    if not resolved.is_file():
        raise FileNotFoundError(f"no such point cloud file: {resolved}")
    return resolved


def read_header(path: str | Path) -> LasHeaderInfo:
    """Read the header of a LAS/LAZ file, touching no point records.

    Args:
        path: the file to open.

    Returns:
        A :class:`LasHeaderInfo` with count, bounds, CRS, dimensions and format metadata.

    Raises:
        FileNotFoundError: when the path does not exist.
        ValueError: when the suffix is neither ``.las`` nor ``.laz``.
    """
    resolved = _resolve(path)
    with laspy.open(resolved) as reader:
        header = reader.header
        return LasHeaderInfo(
            count=int(header.point_count),
            bounds=(float(header.x_min), float(header.y_min), float(header.x_max), float(header.y_max)),
            crs=header.parse_crs(),
            dimensions=tuple(str(name) for name in header.point_format.dimension_names),
            version=str(header.version),
            point_format=int(header.point_format.id),
        )


def read_points(path: str | Path, dimensions: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Read the point records of a LAS/LAZ file.

    Args:
        path: the file to open.
        dimensions: dimension names to read, case-insensitive. ``None`` reads every dimension this engine maps
            onto a :class:`~pointclouds.points.PointCloud` field.

    Returns:
        A mapping of lower-case dimension name to array. A dimension the file's point format does not declare
        comes back as a zero-length array of the right dtype, never as a missing key.

    Raises:
        FileNotFoundError: when the path does not exist.
        ValueError: when the suffix is neither ``.las`` nor ``.laz``.
        KeyError: when a requested dimension is not one this engine knows.
    """
    resolved = _resolve(path)
    requested = _records.resolve(dimensions)
    return _records.extract(laspy.read(resolved), requested)
