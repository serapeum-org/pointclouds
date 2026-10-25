"""The record-array contract shared by every reader in :mod:`pointclouds.io`.

Both engines - the whole-file one in :mod:`pointclouds.io.las` and the windowed one in
:mod:`pointclouds.io.copc` - hand :class:`~pointclouds.points.PointCloud` the same thing: a mapping of
lower-case dimension name to array, with one entry per requested dimension and no missing keys. That contract
lives here rather than in either engine, so the two cannot drift: a dimension added below is readable through
both paths, and a window read maps onto ``PointCloud`` exactly as a full read does.

Private to the package. Callers use the engines, not this module.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

__all__ = ["CANONICAL", "declared_names", "extract", "resolve"]

#: Record dimensions the engines map onto :class:`~pointclouds.points.PointCloud` fields, with the dtype each
#: array is cast to. Keys are lower-case; callers may pass any case (``"X"`` and ``"x"`` are the same dimension).
CANONICAL: dict[str, type] = {
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
    "intensity": np.uint16,
    "classification": np.uint8,
    "return_number": np.uint8,
}

#: The coordinate dimensions. laspy always exposes these as scaled doubles, whatever the point format declares,
#: so they are never treated as absent.
_COORDINATES = ("x", "y", "z")


class _RecordSource(Protocol):
    """What :func:`extract` needs of a record source.

    Both ``laspy.LasData`` (a whole-file read) and ``laspy.ScaleAwarePointRecord`` (a COPC query result) satisfy
    it: a ``point_format`` that can list its dimension names, plus attribute access per dimension.
    """

    point_format: Any


def resolve(dimensions: tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalise a caller's dimension list to lower-case names this package reads.

    Args:
        dimensions: dimension names, case-insensitive, or ``None`` for every canonical dimension.

    Returns:
        The requested names, lower-cased, in the order given.

    Raises:
        KeyError: when a requested dimension is not one the engines know, listing what they do read.

    Examples:
        LAS spells dimensions inconsistently, so the request is normalised and the caller's order kept:

        >>> resolve(("Z", "Intensity"))
        ('z', 'intensity')

        ``None`` is the "everything this package maps" request, which is what both engines pass by default:

        >>> resolve(None)
        ('x', 'y', 'z', 'intensity', 'classification', 'return_number')

        A dimension the package does not map is refused rather than quietly returned empty:

        >>> resolve(("x", "gps_time"))
        Traceback (most recent call last):
            ...
        KeyError: "unknown dimension(s) ['gps_time']; this engine reads ['classification', 'intensity', \
'return_number', 'x', 'y', 'z']"
    """
    requested = tuple(CANONICAL) if dimensions is None else tuple(name.lower() for name in dimensions)
    unknown = [name for name in requested if name not in CANONICAL]
    if unknown:
        raise KeyError(f"unknown dimension(s) {unknown}; this engine reads {sorted(CANONICAL)}")
    return requested


def declared_names(source: _RecordSource) -> set[str]:
    """Return the lower-case dimension names ``source``'s point format declares.

    Args:
        source: a laspy record source (``LasData`` or ``ScaleAwarePointRecord``).

    Returns:
        The declared dimension names, lower-cased.

    Examples:
        Point format 6 declares the attributes this package reads, whatever a given file happens to populate:

        >>> import laspy
        >>> cloud = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
        >>> names = declared_names(cloud)
        >>> {"x", "y", "z", "intensity", "classification", "return_number"} <= names
        True

        The names come back lower-cased, so membership tests work against the canonical keys:

        >>> "X" in declared_names(cloud), "x" in declared_names(cloud)
        (False, True)
    """
    return {str(name).lower() for name in source.point_format.dimension_names}


def extract(source: _RecordSource, requested: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Pull ``requested`` out of a laspy record source as typed arrays.

    Args:
        source: a laspy record source (``LasData`` or ``ScaleAwarePointRecord``).
        requested: lower-case dimension names, as returned by :func:`resolve`.

    Returns:
        A mapping of dimension name to array. A dimension the source's point format does not declare comes back
        as a zero-length array of the right dtype, never as a missing key - so every caller can index the result
        by any requested name.

    Examples:
        Only the requested dimensions are read, which is what makes a partial read cheaper than a full one:

        >>> import laspy
        >>> import numpy as np
        >>> cloud = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
        >>> cloud.x = np.array([10.0, 20.0])
        >>> cloud.y = np.array([0.0, 5.0])
        >>> cloud.z = np.array([1.0, 2.0])
        >>> arrays = extract(cloud, ("x", "z"))
        >>> sorted(arrays)
        ['x', 'z']
        >>> arrays["z"].tolist()
        [1.0, 2.0]

        Each array is cast to its canonical dtype, so a caller never has to check what the file used:

        >>> cloud = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
        >>> cloud.x = np.array([1.0])
        >>> cloud.y = np.array([2.0])
        >>> cloud.z = np.array([3.0])
        >>> cloud.classification = np.array([5])
        >>> arrays = extract(cloud, ("classification", "x"))
        >>> arrays["classification"].dtype.name, arrays["x"].dtype.name
        ('uint8', 'float64')
        >>> arrays["classification"].tolist()
        [5]
    """
    declared = declared_names(source)
    arrays: dict[str, np.ndarray] = {}
    for name in requested:
        dtype = CANONICAL[name]
        if name in _COORDINATES or name in declared:
            # Always a copy, never a view. np.asarray would alias when the dtype already matches, and on the
            # COPC path the source is one packed record buffer - a view would keep the whole buffer alive behind
            # a single-column result and let a caller mutate the source through it. The two engines must agree,
            # and the LAS path copies, so this one does too.
            arrays[name] = np.array(getattr(source, name), dtype=dtype, copy=True)
        else:
            arrays[name] = np.empty(0, dtype=dtype)
    return arrays
