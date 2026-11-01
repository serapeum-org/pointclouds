"""laspy-backed LAS/LAZ engine: header-first read, chunked record read, write.

Entry points, deliberately separate, because the split is the point of the design:

- :func:`read_header` opens the file and returns metadata only. No point record is decompressed, so the cost is
  constant whatever the file's size - a 40 GB tile answers as fast as a 40 KB one.
- :func:`read_points` decompresses the record arrays whole, optionally only the dimensions asked for.
- :func:`read_points_chunked` streams the record arrays a fixed number of points at a time, for a plain (not
  COPC/EPT-indexed) tile too large to materialise at once.
- :func:`write` writes a cloud back out as ``.las`` / ``.laz``, round-tripping the attribute table and CRS.

Everything here imports with the core dependency set (numpy, laspy, pyproj). Spatially-indexed windowed reads
live in :mod:`pointclouds.io.copc` (COPC) and :mod:`pointclouds.io.ept` (EPT). The record-array contract every
reader shares is in :mod:`pointclouds.io._records`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

from pointclouds.io import _records

__all__ = ["LasHeaderInfo", "read_header", "read_points", "read_points_chunked", "write"]

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
            # prefer_wkt=True (also laspy's default): an OGC-WKT VLR wins over the legacy GeoTIFF-key VLR when a
            # file carries both. Stated explicitly so the choice survives a laspy default change, and so COPC and
            # EPT CRS resolution visibly match this one.
            crs=header.parse_crs(prefer_wkt=True),
            dimensions=tuple(str(name) for name in header.point_format.dimension_names),
            version=str(header.version),
            point_format=int(header.point_format.id),
        )


def read_points(path: str | Path, dimensions: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Read the point records of a LAS/LAZ file.

    Args:
        path: the file to open.
        dimensions: dimension names to read, case-insensitive. ``None`` reads every standard dimension this
            engine maps (the canonical attribute table) **plus** every extra/``ExtraBytes`` dimension the file
            carries. A given list reads exactly those standard dimensions.

    Returns:
        A mapping of lower-case dimension name to array. A standard dimension the file's point format does not
        declare comes back as a zero-length array of the right dtype, never as a missing key; extra dimensions
        (only included when ``dimensions`` is ``None``) keep their own native dtype.

    Raises:
        FileNotFoundError: when the path does not exist.
        ValueError: when the suffix is neither ``.las`` nor ``.laz``.
        KeyError: when a requested dimension is not one this engine knows.
    """
    resolved = _resolve(path)
    requested = _records.resolve(dimensions)
    las = laspy.read(resolved)
    arrays = _records.extract(las, requested)
    if dimensions is None:
        arrays.update(_records.extract_extra(las))
    return arrays


def read_points_chunked(
    path: str | Path,
    points_per_iteration: int = 1_000_000,
    dimensions: tuple[str, ...] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream a LAS/LAZ file's records in fixed-size chunks, without materialising the whole cloud.

    This is the linear-streaming counterpart to :func:`read_points`, for a plain ``.laz`` tile of 10^7-10^8
    points that is *not* COPC/EPT-indexed: COPC/EPT windowing gives spatial access, but a non-indexed tile still
    needs a way to be read a slice at a time. Built on ``laspy.open(...).chunk_iterator``.

    Args:
        path: the file to open.
        points_per_iteration: how many points each yielded chunk holds (the last is whatever remains).
        dimensions: dimension names to read, case-insensitive. ``None`` reads every standard dimension this
            engine maps; a list reads exactly those. Extra dimensions are not streamed.

    Yields:
        One record mapping per chunk, under the same contract as :func:`read_points`: lower-case dimension name
        to array, a dimension the format omits present as a zero-length array.

    Raises:
        FileNotFoundError: when the path does not exist.
        ValueError: when the suffix is neither ``.las`` nor ``.laz``, or ``points_per_iteration`` is not positive.
        KeyError: when a requested dimension is not one this engine knows.
    """
    resolved = _resolve(path)
    if points_per_iteration <= 0:
        raise ValueError(f"points_per_iteration must be positive; got {points_per_iteration!r}")
    requested = _records.resolve(dimensions)
    with laspy.open(resolved) as reader:
        for chunk in reader.chunk_iterator(points_per_iteration):
            yield _records.extract(chunk, requested)


def _point_format_for(attributes: dict[str, np.ndarray]) -> int:
    """Pick the smallest LAS 1.4 point format that carries the attributes present.

    Args:
        attributes: the non-coordinate arrays to write; an empty array counts as absent.

    Returns:
        A point format id: ``8`` when ``nir`` is present, ``7`` when RGB is present, else ``6``. All three are
        the LAS 1.4 formats with ``gps_time`` and the wide (16-bit) class/return fields, so the canonical
        attribute table round-trips without truncation.
    """
    present = {name for name, values in attributes.items() if getattr(values, "size", 0)}
    if "nir" in present:
        return 8
    if {"red", "green", "blue"} & present:
        return 7
    return 6


def write(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    attributes: dict[str, np.ndarray] | None = None,
    crs: CRS | None = None,
    scales: tuple[float, float, float] = (0.01, 0.01, 0.01),
    offsets: tuple[float, float, float] | None = None,
) -> None:
    """Write a point cloud to ``path`` as ``.las`` / ``.laz``, round-tripping its attributes and CRS.

    The suffix picks the container: ``.laz`` is LAZ-compressed (via the core ``lazrs`` backend), ``.las`` is
    uncompressed. The point format is chosen by :func:`_point_format_for` from the attributes present, so a
    cloud carrying RGB or NIR is written to a format that can hold it. Standard canonical dimensions go to their
    LAS fields; any other attribute is written as an ``ExtraBytes`` dimension of its own dtype.

    **A written file carries every dimension its point format declares.** The output is always a LAS 1.4 format
    (6, 7 or 8), all of which mandate ``gps_time`` (7/8 add RGB, 8 adds NIR). A dimension the cloud does not
    carry is written **zero-filled**, so reading the file back shows that dimension present (length ``N``) rather
    than absent - the "empty means absent" convention describes a *fresh* read, not a round-trip, since the file
    on disk genuinely has the field. Only the dimensions the cloud populated carry meaningful values.

    Args:
        path: destination file; the suffix (``.las`` / ``.laz``) decides compression.
        x: ``(N,)`` x-coordinates.
        y: ``(N,)`` y-coordinates.
        z: ``(N,)`` z-coordinates.
        attributes: a mapping of lower-case dimension name to ``(N,)`` array for every non-coordinate dimension
            to write; ``None`` or an empty array for an absent one.
        crs: the CRS to record, as a :class:`pyproj.CRS`, or ``None`` to write no SRS.
        scales: the per-axis quantisation step stored in the header (default centimetre precision).
        offsets: the per-axis offset stored in the header; ``None`` uses each axis's minimum, which keeps the
            scaled integer coordinates in range for a large extent.

    Raises:
        ValueError: when the suffix is neither ``.las`` nor ``.laz``, or an attribute's length is neither 0 nor
            ``len(x)``.
        OverflowError: propagated from laspy when a value does not fit the stored representation - a coordinate
            outside the scaled ``int32`` range for the given ``scales`` / ``offsets``, or a field beyond its LAS
            bit width (e.g. ``return_number`` above 15 on a point format with a 4-bit return field). The write
            path does no range clamping; pick ``scales`` / ``offsets`` (or values) that fit.
    """
    resolved = Path(path)
    if resolved.suffix.lower() not in _SUFFIXES:
        raise ValueError(f"{resolved.name!r} is not a LAS/LAZ file; expected one of {sorted(_SUFFIXES)}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = x.shape[0]
    attributes = {name.lower(): np.asarray(values) for name, values in (attributes or {}).items()}
    for name, values in attributes.items():
        # Point-axis count: the first axis for a 1-D/vector array, 1 for a 0-d scalar - so a vector (N, k) extra
        # is validated on N (not N*k) and a scalar gives a clean ValueError, not len()'s TypeError.
        count = values.shape[0] if values.ndim else 1
        if values.size and count != n:
            raise ValueError(f"{name} has length {count}, expected 0 or {n}")

    header = laspy.LasHeader(version="1.4", point_format=_point_format_for(attributes))
    header.scales = np.asarray(scales, dtype=np.float64)
    # Default each axis offset to its own minimum, z included, so a large-magnitude band is representable on
    # every axis alike (a hard-coded z offset of 0.0 would overflow the scaled int32 for a large z while x/y,
    # offset by their min, would not).
    header.offsets = np.asarray(
        offsets
        if offsets is not None
        else (float(x.min()) if n else 0.0, float(y.min()) if n else 0.0, float(z.min()) if n else 0.0),
        dtype=np.float64,
    )
    if crs is not None:
        header.add_crs(crs)

    standard = {name for name in attributes if name in _records.CANONICAL}
    extra = {name for name in attributes if name not in _records.CANONICAL}
    for name in extra:
        values = attributes[name]
        if values.ndim > 2:
            raise ValueError(
                f"{name!r} extra dimension has {values.ndim} axes; write supports a scalar (N,) or vector "
                f"(N, k) ExtraBytes dimension only"
            )
        # A vector (N, k) extra is declared as a k-element field; a scalar (N,) extra as its plain dtype.
        extra_type = np.dtype((values.dtype, values.shape[1])) if values.ndim > 1 else values.dtype
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=extra_type))

    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    for name in standard:
        if attributes[name].size:
            # write targets point formats 6-8, which declare every canonical standard dimension by its canonical
            # name (scan_angle, not the legacy scan_angle_rank), so each name is the right field here.
            setattr(las, name, attributes[name])
    for name in extra:
        setattr(las, name, attributes[name])
    las.write(str(resolved))
