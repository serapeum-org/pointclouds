"""The record-array contract shared by every reader in :mod:`pointclouds.io`.

Every engine - the whole-file one in :mod:`pointclouds.io.las` and the windowed ones in
:mod:`pointclouds.io.copc` (COPC) and :mod:`pointclouds.io.ept` (EPT) - hands
:class:`~pointclouds.points.PointCloud` the same thing: a mapping of lower-case dimension name to array, with
one entry per requested dimension and no missing keys. That contract lives here rather than in any engine, so
they cannot drift: a dimension added below is readable through every path, and a window read maps onto
``PointCloud`` exactly as a full read does. The exact-box clip the two windowed engines share lives here too,
for the same reason.

Private to the package. Callers use the engines, not this module.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

__all__ = [
    "CANONICAL",
    "clip_to_box",
    "declared_names",
    "extra_dimension_names",
    "extract",
    "extract_extra",
    "resolve",
]

#: Standard LAS dimensions the engines map onto :class:`~pointclouds.points.PointCloud` fields, with the dtype
#: each array is cast to. Keys are lower-case; callers may pass any case (``"X"`` and ``"x"`` are one dimension).
#: This is the full standard attribute table; a given file's point format carries only some of these, and a
#: dimension the format omits reads back as a zero-length array (the "empty means absent" convention).
CANONICAL: dict[str, type] = {
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
    "intensity": np.uint16,
    "return_number": np.uint8,
    "number_of_returns": np.uint8,
    "classification": np.uint8,
    "scan_angle": np.int16,
    "user_data": np.uint8,
    "point_source_id": np.uint16,
    "gps_time": np.float64,
    "red": np.uint16,
    "green": np.uint16,
    "blue": np.uint16,
    "nir": np.uint16,
}

#: The coordinate dimensions. laspy always exposes these as scaled doubles, whatever the point format declares,
#: so they are never treated as absent.
_COORDINATES = ("x", "y", "z")

#: A few canonical names laspy spells differently depending on the point format; the first source attribute that
#: the format actually declares wins. The scan angle is ``scan_angle`` on point formats 6-10 (a scaled int16) and
#: ``scan_angle_rank`` on formats 0-5 (an int8); both are read into the canonical ``scan_angle`` field.
_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "scan_angle": ("scan_angle", "scan_angle_rank"),
}


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

        >>> resolve(None)[:4]
        ('x', 'y', 'z', 'intensity')

        A dimension the package does not map is refused rather than quietly returned empty:

        >>> resolve(("x", "elevation_percentile"))
        Traceback (most recent call last):
            ...
        KeyError: "unknown standard dimension(s) ['elevation_percentile']; this engine reads \
['blue', 'classification', 'gps_time', 'green', 'intensity', 'nir', 'number_of_returns', 'point_source_id', \
'red', 'return_number', 'scan_angle', 'user_data', 'x', 'y', 'z']"
    """
    if dimensions is None:
        return tuple(CANONICAL)
    # Lower-case and de-duplicate, preserving first-seen order, so ("Z", "z") - or an upstream forced x/y/z that
    # overlaps the request - does not carry a repeated name into the record contract.
    requested = tuple(dict.fromkeys(name.lower() for name in dimensions))
    unknown = [name for name in requested if name not in CANONICAL]
    if unknown:
        raise KeyError(f"unknown standard dimension(s) {unknown}; this engine reads {sorted(CANONICAL)}")
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
        Only the requested dimensions come back in the mapping (laspy still decompresses the whole point; this
        selects the result, not the read):

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
        source_name = _source_attr(name, declared)
        if name in _COORDINATES or source_name is not None:
            # Always a copy, never a view. np.asarray would alias when the dtype already matches, and on the
            # COPC path the source is one packed record buffer - a view would keep the whole buffer alive behind
            # a single-column result and let a caller mutate the source through it. The two engines must agree,
            # and the LAS path copies, so this one does too.
            arrays[name] = np.array(getattr(source, source_name or name), dtype=dtype, copy=True)
        else:
            arrays[name] = np.empty(0, dtype=dtype)
    return arrays


def _source_attr(canonical: str, declared: set[str]) -> str | None:
    """Return the source attribute name that feeds ``canonical``, or ``None`` when the format omits it.

    Args:
        canonical: a canonical dimension name (a key of :data:`CANONICAL`).
        declared: the lower-case names the source's point format declares (from :func:`declared_names`).

    Returns:
        The declared source attribute to read - ``canonical`` itself, or an alias (e.g. ``scan_angle_rank`` for
        ``scan_angle`` on an older point format) - or ``None`` when neither is declared.
    """
    for candidate in _SOURCE_ALIASES.get(canonical, (canonical,)):
        if candidate in declared:
            return candidate
    return None


def extra_dimension_names(source: _RecordSource) -> tuple[str, ...]:
    """Return the names of any extra (``ExtraBytes``) dimensions the source carries, in file order.

    These are the non-standard, per-file custom dimensions - anything the point format declares that is not one
    of the canonical LAS fields. They pass through read and write by their own name and native dtype.

    Args:
        source: a laspy record source (``LasData`` or ``ScaleAwarePointRecord``).

    Returns:
        The extra dimension names (lower-case), or an empty tuple when the file has none.

    Examples:
        A plain point format carries no extras:

        >>> import laspy
        >>> cloud = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
        >>> extra_dimension_names(cloud)
        ()

        A custom dimension added to the format shows up, lower-cased:

        >>> import laspy
        >>> header = laspy.LasHeader(version="1.4", point_format=6)
        >>> header.add_extra_dim(laspy.ExtraBytesParams(name="Amplitude", type="f4"))
        >>> extra_dimension_names(laspy.LasData(header))
        ('amplitude',)
    """
    extra = getattr(source.point_format, "extra_dimension_names", None)
    if extra is None:
        return ()
    return tuple(str(name).lower() for name in extra)


def extract_extra(source: _RecordSource) -> dict[str, np.ndarray]:
    """Pull every extra (``ExtraBytes``) dimension out of a source as owning copies of its native dtype.

    Args:
        source: a laspy record source (``LasData`` or ``ScaleAwarePointRecord``).

    Returns:
        A mapping of extra dimension name to array, keeping each dimension's own numpy dtype (unlike the
        canonical fields, extras are not cast). Empty when the file declares no extra dimensions.

    Examples:
        An extra dimension comes back by its own name and dtype, as an owning copy:

        >>> import laspy
        >>> import numpy as np
        >>> header = laspy.LasHeader(version="1.4", point_format=6)
        >>> header.add_extra_dim(laspy.ExtraBytesParams(name="amplitude", type="f4"))
        >>> cloud = laspy.LasData(header)
        >>> cloud.x = np.array([0.0, 1.0])
        >>> cloud.amplitude = np.array([1.5, 2.5], dtype=np.float32)
        >>> extras = extract_extra(cloud)
        >>> extras["amplitude"].tolist()
        [1.5, 2.5]
        >>> extras["amplitude"].dtype.name
        'float32'

        A file with no extras yields an empty mapping:

        >>> import laspy
        >>> extract_extra(laspy.LasData(laspy.LasHeader(version="1.4", point_format=6)))
        {}
    """
    return {name: np.array(getattr(source, name), copy=True) for name in extra_dimension_names(source)}


def clip_to_box(arrays: dict[str, np.ndarray], bbox: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    """Drop points outside ``bbox`` exactly, to the float edge. Shared by the COPC and EPT windowed engines.

    Both windowed engines fetch at chunk / node granularity, so their raw result over-includes: COPC because
    laspy rounds the box edges to the file's coordinate scale, EPT because a node's whole data file is fetched
    when its cube overlaps the box. This re-applies the box as an exact float mask, so the result matches the
    requested box to the float edge. The arrays are already in memory, so the mask is cheap.

    **Both edges are inclusive** (``xmin <= x <= xmax``), so a point exactly on a boundary is kept - which is
    what a zero-width point-query box needs. The consequence for tiling: two windows that share an edge each
    include the points on that shared edge, so a caller stitching adjacent tiles must make its own boxes
    half-open (or de-duplicate) to avoid counting the boundary twice.

    Args:
        arrays: a record mapping from :func:`extract`; must contain ``x`` and ``y``.
        bbox: ``(xmin, ymin, xmax, ymax)`` in the data's CRS.

    Returns:
        The same mapping with every array masked to the points inside the box. A zero-length (undeclared)
        dimension is left untouched rather than indexed with a full-length mask.

    Examples:
        Only the points inside the (inclusive) box survive, every array masked in step:

        >>> import numpy as np
        >>> arrays = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 1.0, 2.0]),
        ...           "classification": np.array([2, 5, 2], dtype=np.uint8)}
        >>> kept = clip_to_box(arrays, (0.0, 0.0, 1.0, 1.0))
        >>> kept["x"].tolist()
        [0.0, 1.0]
        >>> kept["classification"].tolist()
        [2, 5]

        An undeclared (zero-length) dimension is left alone, never indexed by the mask:

        >>> import numpy as np
        >>> arrays = {"x": np.array([0.0, 9.0]), "y": np.array([0.0, 9.0]),
        ...           "nir": np.empty(0, dtype=np.uint16)}
        >>> clip_to_box(arrays, (-1.0, -1.0, 1.0, 1.0))["nir"].size
        0
    """
    xmin, ymin, xmax, ymax = bbox
    x, y = arrays["x"], arrays["y"]
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return {name: (values[keep] if values.size == keep.size else values) for name, values in arrays.items()}
