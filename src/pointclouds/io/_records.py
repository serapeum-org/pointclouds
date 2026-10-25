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
    """
    declared = declared_names(source)
    arrays: dict[str, np.ndarray] = {}
    for name in requested:
        dtype = CANONICAL[name]
        if name in _COORDINATES or name in declared:
            arrays[name] = np.asarray(getattr(source, name), dtype=dtype)
        else:
            arrays[name] = np.empty(0, dtype=dtype)
    return arrays
