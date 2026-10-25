"""COPC spatially-indexed windowed reads: pull back a bounding box, or a coarser level of detail. (T-16).

A COPC file is a LAZ 1.4 whose chunks are laid out as an octree, with the tree's index stored in the file. That
layout is what makes a *windowed* read possible: given a bounding box, only the chunks whose octree voxels
overlap it are fetched and decompressed, and given a target resolution, only the octree levels coarse enough to
satisfy it are visited. The alternative - :func:`pointclouds.io.las.read_points` followed by a mask - reads and
decompresses every record first, which is the cost COPC exists to avoid.

Two entry points, mirroring :mod:`pointclouds.io.las`:

- :func:`read_header` answers ``count`` / ``bounds`` / ``crs`` plus the octree's root ``spacing``, touching no
  point record - the same header-first contract the LAS engine keeps.
- :func:`read_window` runs the query and returns record arrays, under the record contract in
  :mod:`pointclouds.io._records`, so a window maps onto :class:`~pointclouds.points.PointCloud` exactly as a
  full read does.

``laspy`` provides the octree traversal (:class:`laspy.CopcReader`) and ``lazrs`` the per-chunk decompression;
both are already core dependencies, so nothing here needs an extra. Everything imports with the core set
(numpy, laspy, pyproj) and never touches GDAL.

A COPC source may be a local path or any file-like object, which is how a remote cloud is read without this
package taking on an HTTP client: open the object with ``fsspec`` (the ``[cloud]`` extra) and pass it in.

**EPT is not implemented here yet.** An Entwine tileset is a directory of files behind an ``ept.json`` rather
than a single indexed file, and ``laspy`` has no reader for it, so it needs its own traversal and fetch path -
the second half of the issue this module closes the first half of.
"""

from __future__ import annotations

import io
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from laspy.copc import Bounds
from laspy.errors import LaspyException

from pointclouds.io import _records
from pointclouds.io.las import LasHeaderInfo

__all__ = ["CopcHeaderInfo", "is_copc", "read_header", "read_window"]

#: A COPC file is LAZ-compressed by definition (the octree is a chunk layout, and chunks only exist in LAZ), so
#: an uncompressed ``.las`` can never be one.
_SUFFIXES = {".laz"}

#: What a source may be: a path, or an already-open file object (an ``fsspec`` handle, a ``BytesIO``, ...).
Source = str | os.PathLike[str] | io.IOBase


@dataclass(frozen=True)
class CopcHeaderInfo(LasHeaderInfo):
    """A COPC file's header, plus the one octree property a caller needs to choose a ``resolution``.

    Everything :class:`~pointclouds.io.las.LasHeaderInfo` carries is answered the same way here, from the
    header alone. This subclass adds:

    Attributes:
        spacing: the root node's point spacing, in the data's own units (metres for a projected CRS). The octree
            halves it at every level, so a level ``n`` node has spacing ``spacing / 2**n``. This is the number
            ``resolution=`` is measured against - ask for ``resolution=spacing`` and only the root is read.

    Examples:
        ``spacing`` is what turns a target resolution into an octree depth, so a caller can work out how deep a
        view will have to go before paying for the read:

        >>> info = CopcHeaderInfo(
        ...     count=1600,
        ...     bounds=(155000.0, 463000.0, 155078.0, 463078.0),
        ...     crs=None,
        ...     dimensions=("X", "Y", "Z"),
        ...     version="1.4",
        ...     point_format=6,
        ...     spacing=7.8,
        ... )
        >>> info.spacing / 2**2  # the spacing two levels down
        1.95

        Everything the LAS header answers is here too, so the file's extent and size are known before any
        record is read:

        >>> info.count, info.bounds[:2]
        (1600, (155000.0, 463000.0))
    """

    spacing: float


def _resolve(source: Source) -> Source:
    """Validate ``source`` if it is a path, and pass a file object through untouched.

    Args:
        source: the COPC file to open, as a path or an open file object.

    Returns:
        The resolved :class:`~pathlib.Path`, or ``source`` itself when it is a file object.

    Raises:
        FileNotFoundError: when a path does not exist, naming the path rather than surfacing laspy's error.
        ValueError: when a path's suffix is not ``.laz``.
    """
    if isinstance(source, io.IOBase):
        return source
    resolved = Path(source)
    if resolved.suffix.lower() not in _SUFFIXES:
        raise ValueError(
            f"{resolved.name!r} is not a COPC file; COPC is always LAZ-compressed, so the suffix must be "
            f"'.laz' (conventionally '.copc.laz')"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"no such point cloud file: {resolved}")
    return resolved


@contextmanager
def _reader(source: Source) -> Iterator[laspy.CopcReader]:
    """Open ``source`` as a COPC reader, translating laspy's "not a COPC" error into a ``ValueError``.

    Args:
        source: the COPC file to open, as a path or an open file object.

    Yields:
        An open :class:`laspy.CopcReader`.

    Raises:
        FileNotFoundError: when a path does not exist.
        ValueError: when the file carries no COPC octree - a plain LAS/LAZ, which :func:`read_window` cannot
            window and which :func:`pointclouds.io.las.read_points` should read instead.
    """
    resolved = _resolve(source)
    if isinstance(resolved, io.IOBase):
        # Rewind: laspy reads the header from wherever the handle happens to sit, so a second read of the same
        # handle - a header read followed by a window read - would otherwise start mid-file.
        resolved.seek(0)
    try:
        if isinstance(resolved, io.IOBase):
            # close_fd=False: the caller opened this handle, so the caller closes it. laspy's own default would
            # close it on exit, which would break a caller reading several windows out of one fsspec handle.
            reader = laspy.CopcReader(resolved, close_fd=False)
        else:
            reader = laspy.CopcReader.open(resolved)
    except LaspyException as exc:
        label = resolved.name if isinstance(resolved, Path) else "the given file object"
        raise ValueError(
            f"{label} carries no COPC octree, so it cannot be read through a window; read it whole with "
            f"PointCloud.read_file() instead ({exc})"
        ) from exc
    with reader:
        yield reader


def is_copc(source: Source) -> bool:
    """Report whether ``source`` carries a COPC octree.

    This is the cheap question: it reads the header and its first VLR, never a point record.

    Args:
        source: the file to inspect, as a path or an open file object.

    Returns:
        ``True`` when the file is a valid COPC, ``False`` when it is a plain LAS/LAZ.

    Raises:
        FileNotFoundError: when a path does not exist.
        ValueError: when a path's suffix is not ``.laz`` - an uncompressed ``.las`` is never COPC.

    Examples:
        An indexed cloud answers yes, and a plain LAZ answers no rather than raising - that is the question:

        >>> is_copc("examples/data/sample.copc.laz")  # doctest: +SKIP
        True
        >>> is_copc("examples/data/sample.laz")  # doctest: +SKIP
        False

        Which makes it the way to choose a read path for a file of unknown provenance:

        >>> path = "examples/data/sample.laz"  # doctest: +SKIP
        >>> read_window(path, bbox=(0, 0, 1, 1)) if is_copc(path) else "read it whole instead"  # doctest: +SKIP
        'read it whole instead'
    """
    resolved = _resolve(source)  # a bad suffix or a missing file is an error, not a "no" answer
    try:
        with _reader(resolved):
            return True
    except ValueError:
        return False


def read_header(source: Source) -> CopcHeaderInfo:
    """Read a COPC file's header and octree root properties, touching no point records.

    Args:
        source: the COPC file to open, as a path or an open file object.

    Returns:
        A :class:`CopcHeaderInfo` with count, bounds, CRS, dimensions, format metadata and the root ``spacing``.

    Raises:
        FileNotFoundError: when a path does not exist.
        ValueError: when a path's suffix is not ``.laz``, or the file carries no COPC octree.

    Examples:
        The size and extent of a cloud are answerable before deciding whether to read any of it:

        >>> info = read_header("examples/data/sample.copc.laz")  # doctest: +SKIP
        >>> info.count, info.spacing  # header only - no records touched  # doctest: +SKIP
        (1600, 7.8)

        ``spacing`` says what a coarse read will cost: asking for it reads the root node alone:

        >>> info = read_header("examples/data/sample.copc.laz")  # doctest: +SKIP
        >>> read_window("examples/data/sample.copc.laz", resolution=info.spacing)["x"].size  # doctest: +SKIP
        100
    """
    with _reader(source) as reader:
        header = reader.header
        return CopcHeaderInfo(
            count=int(header.point_count),
            bounds=(float(header.x_min), float(header.y_min), float(header.x_max), float(header.y_max)),
            crs=header.parse_crs(),
            dimensions=tuple(str(name) for name in header.point_format.dimension_names),
            version=str(header.version),
            point_format=int(header.point_format.id),
            spacing=float(reader.copc_info.spacing),
        )


def _as_bounds(bbox: tuple[float, float, float, float], header: laspy.LasHeader) -> Bounds:
    """Turn a 2-D bounding box into the laspy ``Bounds`` the octree query takes, clamped to the file.

    The box stays 2-D on purpose: laspy extends it over the file's full z range before querying, so no point is
    ever dropped for its elevation. Filtering in z is a cloud operation, not a read window.

    The box is clamped to the file's own x/y extent before it reaches laspy. laspy clips in scaled ``int32``
    space (``np.round((mins - offset) / scale)``), which overflows for coordinates past roughly ``±2.1e7`` at a
    centimetre scale and silently returns zero points. Clamping first means a generously large or sentinel box
    reads the whole cloud, which is what "give me everything in this area" has to mean.

    Args:
        bbox: ``(xmin, ymin, xmax, ymax)`` in the file's CRS.
        header: the open file's header, read for its ``x_min`` / ``y_min`` / ``x_max`` / ``y_max`` extent.

    Returns:
        The equivalent :class:`laspy.copc.Bounds`, clamped to the file's extent.

    Raises:
        ValueError: when ``bbox`` is not four numbers, holds a non-finite value, or either axis is inverted.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (xmin, ymin, xmax, ymax); got {len(bbox)} value(s): {bbox!r}")
    xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    # nan slips the inversion test below (nan > nan is False), so reject non-finite corners explicitly first.
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"bbox corners must be finite; got (xmin, ymin, xmax, ymax) = {(xmin, ymin, xmax, ymax)!r}")
    if xmin > xmax or ymin > ymax:
        raise ValueError(f"bbox is inverted: (xmin, ymin, xmax, ymax) = {(xmin, ymin, xmax, ymax)!r}")
    return Bounds(
        mins=np.array([max(xmin, header.x_min), max(ymin, header.y_min)]),
        maxs=np.array([min(xmax, header.x_max), min(ymax, header.y_max)]),
    )


def _clip_to_box(arrays: dict[str, np.ndarray], bbox: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    """Drop points outside ``bbox`` exactly, to the float edge.

    laspy rounds the box edges to the file's coordinate scale before clipping, so its result can include points
    up to half a scale unit outside the requested box - a whole boundary row, when an edge falls between grid
    lines. This re-applies the box as an exact float mask so adjacent, non-overlapping windows never both claim
    a boundary point. The arrays are already in memory, so the mask is cheap.

    Args:
        arrays: the mapping from :func:`pointclouds.io._records.extract`; must contain ``x`` and ``y``.
        bbox: ``(xmin, ymin, xmax, ymax)`` in the file's CRS.

    Returns:
        The same mapping with every array masked to the points inside the box.
    """
    xmin, ymin, xmax, ymax = bbox
    x, y = arrays["x"], arrays["y"]
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    # An undeclared dimension is a zero-length array (the record contract); leave those untouched rather than
    # index them with a full-length mask.
    return {name: (values[keep] if values.size == keep.size else values) for name, values in arrays.items()}


def read_window(
    source: Source,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    level: int | range | None = None,
    dimensions: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Read only the points a window selects, using the file's octree rather than scanning every record.

    With no argument at all this reads the whole cloud, which is the honest default but also the expensive one -
    the point of the function is to pass a window.

    Args:
        source: the COPC file to open, as a path or an open file object.
        bbox: ``(xmin, ymin, xmax, ymax)`` in the file's CRS. Only chunks whose octree voxel overlaps the box are
            fetched, and the points are then clipped exactly to the box, to the float edge. 2-D: the full z
            range is always kept. A box larger than the file is clamped to the file's extent; a non-finite
            corner is rejected.
        resolution: target point spacing in the data's units. The octree is visited only down to the level that
            satisfies it, measured against :attr:`CopcHeaderInfo.spacing`, so a wide view stays cheap. Mutually
            exclusive with ``level``.
        level: an exact octree level, or a ``range`` of them, when a caller would rather name levels than a
            spacing. Mutually exclusive with ``resolution``.
        dimensions: dimension names to read, case-insensitive. ``None`` reads every dimension the package maps
            onto a :class:`~pointclouds.points.PointCloud` field.

    Returns:
        A mapping of lower-case dimension name to array, under the same contract as
        :func:`pointclouds.io.las.read_points`: a dimension the point format does not declare comes back as a
        zero-length array, never as a missing key. A window that selects nothing gives zero-length arrays
        throughout.

    Raises:
        FileNotFoundError: when a path does not exist.
        ValueError: when a path's suffix is not ``.laz``; when the file carries no COPC octree; when ``bbox`` is
            malformed; when ``resolution`` is not positive; when ``level`` is negative; or when ``resolution``
            and ``level`` are both given.
        KeyError: when a requested dimension is not one this package reads.

    Examples:
        A bounding box reads a corner of the cloud, fetching only the chunks that reach into it:

        >>> arrays = read_window(  # doctest: +SKIP
        ...     "examples/data/sample.copc.laz", bbox=(155000, 463000, 155020, 463020)
        ... )
        >>> arrays["x"].size  # 121 of the file's 1600 points  # doctest: +SKIP
        121

        A resolution reads the whole extent sparsely instead, which is what a zoomed-out view wants:

        >>> coarse = read_window("examples/data/sample.copc.laz", resolution=7.8)  # doctest: +SKIP
        >>> coarse["x"].size, coarse["classification"].size  # doctest: +SKIP
        (100, 100)

        The two compose, and a dimension list trims what is decompressed per point:

        >>> corner = read_window(  # doctest: +SKIP
        ...     "examples/data/sample.copc.laz",
        ...     bbox=(155000, 463000, 155020, 463020),
        ...     resolution=7.8,
        ...     dimensions=("X", "Y"),
        ... )
        >>> sorted(corner)  # doctest: +SKIP
        ['x', 'y']
    """
    if resolution is not None and level is not None:
        raise ValueError("pass either resolution= or level=, not both; they are two ways to say the same thing")
    if resolution is not None and resolution <= 0:
        raise ValueError(f"resolution must be positive; got {resolution!r}")
    if isinstance(level, int) and level < 0:
        raise ValueError(f"level must be a non-negative octree level; got {level!r}")

    requested = _records.resolve(dimensions)
    # An exact bbox clip needs x and y even when the caller did not ask for them; read them, then drop them
    # again if they were not requested.
    needed = requested if bbox is None else tuple(dict.fromkeys(("x", "y", *requested)))
    with _reader(source) as reader:
        bounds = None if bbox is None else _as_bounds(bbox, reader.header)
        points = reader.query(bounds=bounds, resolution=resolution, level=level)
        arrays = _records.extract(points, needed)
        if bbox is not None:
            arrays = _clip_to_box(arrays, bbox)
            arrays = {name: arrays[name] for name in requested}
        return arrays
