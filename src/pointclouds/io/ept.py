"""EPT (Entwine Point Tile) spatially-indexed windowed reads: a bounding box, or a coarser level of detail.

Where COPC packs an octree into a single LAZ file (:mod:`pointclouds.io.copc`), EPT spreads the same idea across
a **directory tree**: an ``ept.json`` manifest, a ``ept-hierarchy/`` of octree index pages, and a ``ept-data/``
of one file per node. That shape is what a windowed read exploits - given a bounding box, only the nodes whose
octree cube overlaps it are fetched; given a target resolution, only the shallow octree depths are visited. The
alternative, reading every node and masking, defeats the point of the tiling.

``laspy`` has no EPT reader, so this module walks the tree itself:

- :func:`read_header` reads ``ept.json`` alone - ``count`` / ``bounds`` / ``crs`` / ``span`` - touching no node
  data, the same header-first contract the LAS and COPC engines keep.
- :func:`read_window` selects nodes from the hierarchy, fetches and decodes each, and returns record arrays
  under the contract in :mod:`pointclouds.io._records`, so an EPT window maps onto
  :class:`~pointclouds.points.PointCloud` exactly as every other read does.

**Data types.** EPT stores each node as ``laszip`` (a standalone LAZ, read with ``laspy``), ``binary`` (the
schema's dimensions packed little-endian, read with ``numpy``), or ``zstandard`` (binary under zstd framing).
The first two are supported; ``zstandard`` raises a clear error naming what is missing, rather than returning
wrong data.

**Local and remote.** A source may be a path to an ``ept.json`` or to the directory holding it, or an
``http(s)://`` / ``s3://`` URL to either. Remote access goes through ``fsspec`` behind the ``[cloud]`` extra;
the core imports with ``numpy`` / ``laspy`` / ``pyproj`` alone and never touches GDAL. A node's data is fetched
through one indirection (:class:`_Store`), so the local and remote paths differ only in how bytes are read.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from pyproj import CRS

from pointclouds.io import _records

__all__ = ["EptInfo", "is_ept", "read_header", "read_window"]

#: EPT schema dimension names (case-insensitive) mapped onto this package's canonical lower-case names. A name
#: EPT carries but this package does not map (``Red``, ``GpsTime``, ...) is simply not read.
_SCHEMA_ALIASES: dict[str, str] = {
    "x": "x",
    "y": "y",
    "z": "z",
    "intensity": "intensity",
    "classification": "classification",
    "returnnumber": "return_number",
}

#: NumPy dtype for each EPT schema ``{type, size}`` pair. EPT types are ``signed`` / ``unsigned`` / ``float``.
_NUMPY_TYPES: dict[tuple[str, int], type] = {
    ("signed", 1): np.int8,
    ("signed", 2): np.int16,
    ("signed", 4): np.int32,
    ("signed", 8): np.int64,
    ("unsigned", 1): np.uint8,
    ("unsigned", 2): np.uint16,
    ("unsigned", 4): np.uint32,
    ("unsigned", 8): np.uint64,
    ("float", 4): np.float32,
    ("float", 8): np.float64,
}

#: The manifest filename at the root of every EPT dataset.
_MANIFEST = "ept.json"

#: A source may be a path or a URL to the manifest or its directory.
Source = str | os.PathLike[str]


class _Store:
    """Read-only access to an EPT dataset's files, local or remote, by path relative to the dataset root.

    The one indirection the local and remote paths share: :func:`read_header` and :func:`read_window` ask for
    ``ept.json``, ``ept-hierarchy/<key>.json`` and ``ept-data/<key>.<ext>`` by relative path, and this resolves
    each against the root - a local directory, or a URL through ``fsspec``.
    """

    def __init__(self, root: str, fs: Any | None) -> None:
        """Build a store.

        Args:
            root: the dataset root - a local directory path, or a URL (with a trailing-slash-insensitive join).
            fs: an ``fsspec`` filesystem for a remote root, or ``None`` for the local filesystem.
        """
        self._root = root.rstrip("/")
        self._fs = fs

    def read_bytes(self, relative: str) -> bytes:
        """Return the bytes of the file at ``relative`` under the root.

        Args:
            relative: a path relative to the dataset root, e.g. ``"ept-data/0-0-0-0.laz"``.

        Returns:
            The file's raw bytes.
        """
        if self._fs is None:
            return Path(self._root, relative).read_bytes()
        with self._fs.open(f"{self._root}/{relative}", "rb") as handle:
            return handle.read()

    def exists(self, relative: str) -> bool:
        """Whether the file at ``relative`` exists.

        Args:
            relative: a path relative to the dataset root.

        Returns:
            ``True`` if it exists.
        """
        if self._fs is None:
            return Path(self._root, relative).is_file()
        return bool(self._fs.exists(f"{self._root}/{relative}"))


def _store(source: Source) -> _Store:
    """Resolve a source to a :class:`_Store` rooted at the dataset directory.

    Args:
        source: a path or URL to an ``ept.json`` or to the directory containing it.

    Returns:
        A store whose root is the dataset directory.

    Raises:
        FileNotFoundError: when a local path has no ``ept.json``.
        ModuleNotFoundError: when a remote URL is given but the ``[cloud]`` extra is not installed.
    """
    text = str(source)
    if "://" in text:
        try:
            import fsspec
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise ModuleNotFoundError(
                "reading a remote EPT needs the optional cloud stack (fsspec). Install it with: "
                "pip install pointclouds[cloud]"
            ) from exc
        root = text[: -len(_MANIFEST)] if text.endswith(_MANIFEST) else text
        fs, _, (path,) = fsspec.get_fs_token_paths(root.rstrip("/"))
        return _Store(path, fs)

    path = Path(source)
    base = path.parent if path.name == _MANIFEST else path
    if not (base / _MANIFEST).is_file():
        raise FileNotFoundError(f"no EPT manifest at {base / _MANIFEST}")
    return _Store(str(base), None)


@dataclass(frozen=True)
class EptInfo:
    """Everything an EPT ``ept.json`` answers without reading a single node.

    Attributes:
        count: the dataset's total point count, from the manifest.
        bounds: ``(xmin, ymin, xmax, ymax)`` of the *conforming* (tight) bounds, in the dataset's CRS.
        crs: the dataset's CRS as a :class:`pyproj.CRS`, or ``None`` when the manifest carries no SRS.
        dimensions: every dimension name the schema declares, in schema order.
        version: the EPT version string, e.g. ``"1.0.0"``.
        span: the octree's points-per-edge resolution (e.g. ``128``); the number ``resolution=`` is measured
            against. A node at depth ``d`` has point spacing roughly ``cube_side / (span * 2**d)``.
        data_type: the node encoding - ``"laszip"``, ``"binary"`` or ``"zstandard"``.
    """

    count: int
    bounds: tuple[float, float, float, float]
    crs: CRS | None
    dimensions: tuple[str, ...]
    version: str
    span: int
    data_type: str


def _parse_crs(manifest: dict[str, Any]) -> CRS | None:
    """Build a :class:`pyproj.CRS` from an EPT ``srs`` block, or ``None`` when absent.

    Args:
        manifest: the parsed ``ept.json``.

    Returns:
        The CRS, or ``None``. A WKT string is preferred; else an ``authority``/``horizontal`` code pair.
    """
    srs = manifest.get("srs") or {}
    if srs.get("wkt"):
        return CRS.from_wkt(srs["wkt"])
    if srs.get("authority") and srs.get("horizontal"):
        return CRS.from_authority(srs["authority"], srs["horizontal"])
    return None


def _read_manifest(store: _Store) -> dict[str, Any]:
    """Read and parse ``ept.json``.

    Args:
        store: the dataset store.

    Returns:
        The parsed manifest.
    """
    return json.loads(store.read_bytes(_MANIFEST).decode("utf-8"))


def is_ept(source: Source) -> bool:
    """Report whether ``source`` is an EPT dataset.

    Cheap: it checks for an ``ept.json``, reading no node data.

    Args:
        source: a path or URL to inspect.

    Returns:
        ``True`` when an ``ept.json`` is present at or under ``source``, ``False`` otherwise.

    Examples:
        >>> is_ept("examples/data/sample-ept")  # doctest: +SKIP
        True
    """
    text = str(source)
    if "://" in text:
        try:
            return _store(source).exists(_MANIFEST)
        except (ModuleNotFoundError, FileNotFoundError, ValueError):
            return False
    path = Path(source)
    if path.name == _MANIFEST:
        return path.is_file()
    return (path / _MANIFEST).is_file()


def read_header(source: Source) -> EptInfo:
    """Read an EPT dataset's manifest, touching no node data.

    Args:
        source: a path or URL to an ``ept.json`` or its directory.

    Returns:
        An :class:`EptInfo` with count, bounds, CRS, dimensions, version, span and data type.

    Raises:
        FileNotFoundError: when there is no ``ept.json`` at the source.
        ModuleNotFoundError: for a remote URL without the ``[cloud]`` extra.

    Examples:
        >>> info = read_header("examples/data/sample-ept")  # doctest: +SKIP
        >>> info.count, info.span  # doctest: +SKIP
        (1600, 128)
    """
    manifest = _read_manifest(_store(source))
    conforming = manifest.get("boundsConforming") or manifest["bounds"]
    schema = manifest.get("schema", [])
    return EptInfo(
        count=int(manifest.get("points", 0)),
        bounds=(float(conforming[0]), float(conforming[1]), float(conforming[3]), float(conforming[4])),
        crs=_parse_crs(manifest),
        dimensions=tuple(str(entry["name"]) for entry in schema),
        version=str(manifest.get("version", "")),
        span=int(manifest.get("span", 0)),
        data_type=str(manifest["dataType"]),
    )


@dataclass(frozen=True)
class _Cube:
    """An octree node's cubic bounds, carried as min/max corners in the dataset's CRS."""

    mins: np.ndarray
    maxs: np.ndarray

    def child(self, dx: int, dy: int, dz: int) -> _Cube:
        """Return the child sub-cube in octant ``(dx, dy, dz)`` (each 0 or 1).

        Args:
            dx: x octant, 0 (low) or 1 (high).
            dy: y octant.
            dz: z octant.

        Returns:
            The child cube, half the side in every axis.
        """
        mid = (self.mins + self.maxs) / 2.0
        lo = np.where([dx, dy, dz], mid, self.mins)
        hi = np.where([dx, dy, dz], self.maxs, mid)
        return _Cube(lo, hi)

    def overlaps_xy(self, bbox: tuple[float, float, float, float]) -> bool:
        """Whether this cube's x/y footprint overlaps a 2-D bounding box.

        Args:
            bbox: ``(xmin, ymin, xmax, ymax)``.

        Returns:
            ``True`` if the footprints intersect (touching edges count).
        """
        xmin, ymin, xmax, ymax = bbox
        return not (self.maxs[0] < xmin or self.mins[0] > xmax or self.maxs[1] < ymin or self.mins[1] > ymax)


def _root_cube(manifest: dict[str, Any]) -> _Cube:
    """The octree root cube, from the manifest's cubic ``bounds``.

    Args:
        manifest: the parsed ``ept.json``.

    Returns:
        The root :class:`_Cube`.
    """
    b = manifest["bounds"]
    return _Cube(np.array([b[0], b[1], b[2]], dtype=float), np.array([b[3], b[4], b[5]], dtype=float))


def _max_depth(manifest: dict[str, Any], resolution: float | None, depth: int | None) -> int | None:
    """Translate a ``resolution`` or an explicit ``depth`` into the deepest octree level to visit.

    Args:
        manifest: the parsed ``ept.json`` (for ``span`` and the cube side).
        resolution: target point spacing in CRS units, or ``None``.
        depth: an explicit maximum depth, or ``None``.

    Returns:
        The maximum depth (inclusive) to descend to, or ``None`` for "no limit - the full dataset".
    """
    if depth is not None:
        return depth
    if resolution is None:
        return None
    b = manifest["bounds"]
    side = float(b[3]) - float(b[0])
    span = int(manifest.get("span", 0)) or 1
    # Spacing at depth d is side / (span * 2**d); descend until it is <= resolution.
    ratio = side / (span * resolution)
    return max(0, math.ceil(math.log2(ratio))) if ratio > 1 else 0


class _Hierarchy:
    """The EPT octree index, loaded page by page on demand.

    The root page (``ept-hierarchy/0-0-0-0.json``) maps ``"d-x-y-z"`` keys to point counts; a count of ``-1``
    means that node's subtree lives in its own page, fetched lazily the first time the walk reaches it.
    """

    def __init__(self, store: _Store) -> None:
        """Build a hierarchy over ``store``, loading only the root page.

        Args:
            store: the dataset store.
        """
        self._store = store
        self._counts: dict[str, int] = {}
        self._load_page("0-0-0-0")

    def _load_page(self, key: str) -> None:
        """Merge the hierarchy page named ``key`` into the known counts.

        Args:
            key: the ``"d-x-y-z"`` key whose page to load.
        """
        page = json.loads(self._store.read_bytes(f"ept-hierarchy/{key}.json").decode("utf-8"))
        self._counts.update({str(k): int(v) for k, v in page.items()})

    def count(self, key: str) -> int:
        """Return the point count for ``key``, loading a nested page if the index points to one.

        Args:
            key: the ``"d-x-y-z"`` node key.

        Returns:
            The node's point count, ``0`` when the node is absent. A ``-1`` placeholder is resolved by loading
            the node's own page and re-reading the count.
        """
        value = self._counts.get(key, 0)
        if value == -1:
            self._load_page(key)
            value = self._counts.get(key, 0)
        return value


def _select_nodes(
    hierarchy: _Hierarchy,
    root: _Cube,
    bbox: tuple[float, float, float, float] | None,
    max_depth: int | None,
) -> list[str]:
    """Walk the octree and return the keys of every node to read.

    Args:
        hierarchy: the loaded hierarchy index.
        root: the root cube.
        bbox: a 2-D bounding box to prune against, or ``None`` for the whole extent.
        max_depth: the deepest level to descend to, or ``None`` for no limit.

    Returns:
        The ``"d-x-y-z"`` keys of the non-empty nodes whose cube overlaps ``bbox`` and whose depth is within
        ``max_depth``.
    """
    selected: list[str] = []
    stack: list[tuple[int, int, int, int, _Cube]] = [(0, 0, 0, 0, root)]
    while stack:
        d, x, y, z, cube = stack.pop()
        if bbox is not None and not cube.overlaps_xy(bbox):
            continue
        if hierarchy.count(f"{d}-{x}-{y}-{z}") <= 0:
            continue
        selected.append(f"{d}-{x}-{y}-{z}")
        if max_depth is not None and d >= max_depth:
            continue
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    stack.append((d + 1, 2 * x + dx, 2 * y + dy, 2 * z + dz, cube.child(dx, dy, dz)))
    return selected


def _decode_laszip(data: bytes, requested: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Decode a ``laszip`` node (a standalone LAZ) into the requested record arrays.

    Args:
        data: the node file's bytes.
        requested: the resolved canonical dimension names.

    Returns:
        A record mapping, via the shared :func:`pointclouds.io._records.extract`.
    """
    return _records.extract(laspy.read(io.BytesIO(data)), requested)


def _binary_dtype(schema: list[dict[str, Any]]) -> np.dtype:
    """Build the structured dtype a ``binary`` node is packed as, from the EPT schema.

    Args:
        schema: the manifest's ``schema`` list of ``{name, type, size}`` entries.

    Returns:
        A little-endian structured :class:`numpy.dtype`, one field per schema entry in order.

    Raises:
        ValueError: when an entry's ``{type, size}`` is not an EPT numeric type.
    """
    fields = []
    for entry in schema:
        key = (str(entry["type"]), int(entry["size"]))
        if key not in _NUMPY_TYPES:
            raise ValueError(f"unsupported EPT schema type {key} for dimension {entry.get('name')!r}")
        fields.append((str(entry["name"]), np.dtype(_NUMPY_TYPES[key]).newbyteorder("<")))
    return np.dtype(fields)


def _decode_binary(data: bytes, schema: list[dict[str, Any]], requested: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Decode a ``binary`` node into the requested record arrays, applying each dimension's scale/offset.

    Args:
        data: the node file's bytes - the schema's dimensions packed little-endian, one record per point.
        schema: the manifest's ``schema`` list, which defines the packing and any per-dimension scale/offset.
        requested: the resolved canonical dimension names.

    Returns:
        A record mapping under the shared contract: every requested name present, a dimension the schema does
        not carry coming back as a zero-length array of the canonical dtype.
    """
    records = np.frombuffer(data, dtype=_binary_dtype(schema))
    # canonical name -> the schema entry that feeds it (first match wins).
    by_canonical: dict[str, dict[str, Any]] = {}
    for entry in schema:
        canonical = _SCHEMA_ALIASES.get(str(entry["name"]).lower())
        if canonical is not None:
            by_canonical.setdefault(canonical, entry)

    arrays: dict[str, np.ndarray] = {}
    for name in requested:
        dtype = _records.CANONICAL[name]
        feed = by_canonical.get(name)
        if feed is None:
            arrays[name] = np.empty(0, dtype=dtype)
            continue
        column = records[str(feed["name"])]
        scale, offset = feed.get("scale"), feed.get("offset")
        if scale is not None or offset is not None:
            factor = float(scale) if scale is not None else 1.0
            shift = float(offset) if offset is not None else 0.0
            column = column * factor + shift
        arrays[name] = np.array(column, dtype=dtype, copy=True)
    return arrays


def _decode(
    data: bytes, data_type: str, schema: list[dict[str, Any]], requested: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Decode one node's bytes according to the dataset's ``dataType``.

    Args:
        data: the node file's bytes.
        data_type: the manifest's ``dataType`` - ``"laszip"`` or ``"binary"``.
        schema: the manifest's ``schema`` (used by the binary path).
        requested: the resolved canonical dimension names.

    Returns:
        A record mapping for that node.

    Raises:
        NotImplementedError: for ``"zstandard"`` - recognised but not decoded here.
        ValueError: for an unknown data type.
    """
    if data_type == "laszip":
        return _decode_laszip(data, requested)
    if data_type == "binary":
        return _decode_binary(data, schema, requested)
    if data_type == "zstandard":
        raise NotImplementedError(
            "EPT 'zstandard' node encoding is not supported yet; re-export the dataset as 'laszip' or 'binary'"
        )
    raise ValueError(f"unknown EPT dataType {data_type!r}; expected 'laszip', 'binary' or 'zstandard'")


def _node_extension(data_type: str) -> str:
    """The data-file extension for a given ``dataType``.

    Args:
        data_type: the manifest's ``dataType``.

    Returns:
        ``"laz"`` for laszip, else ``"bin"``.
    """
    return "laz" if data_type == "laszip" else "bin"


def _concatenate(parts: list[dict[str, np.ndarray]], requested: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Join per-node record mappings into one, preserving the no-missing-key contract.

    Args:
        parts: one record mapping per node read.
        requested: the resolved canonical dimension names.

    Returns:
        A single mapping, each dimension the concatenation of that dimension across nodes. An empty selection
        gives zero-length arrays throughout.
    """
    if not parts:
        return {name: np.empty(0, dtype=_records.CANONICAL[name]) for name in requested}
    return {name: np.concatenate([part[name] for part in parts]) for name in requested}


def _check_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Validate a bounding box's shape, finiteness and orientation. Needs no dataset, so it runs before any read.

    Args:
        bbox: ``(xmin, ymin, xmax, ymax)`` in the dataset's CRS.

    Raises:
        ValueError: when ``bbox`` is not four numbers, holds a non-finite value, or either axis is inverted.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (xmin, ymin, xmax, ymax); got {len(bbox)} value(s): {bbox!r}")
    xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"bbox corners must be finite; got (xmin, ymin, xmax, ymax) = {(xmin, ymin, xmax, ymax)!r}")
    if xmin > xmax or ymin > ymax:
        raise ValueError(f"bbox is inverted: (xmin, ymin, xmax, ymax) = {(xmin, ymin, xmax, ymax)!r}")


def read_window(
    source: Source,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    depth: int | None = None,
    dimensions: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Read only the points a window selects from an EPT dataset, using its octree rather than every node.

    With no argument this reads the whole dataset - the honest default, and the expensive one; the point is to
    pass a window.

    Args:
        source: a path or URL to an ``ept.json`` or its directory.
        bbox: ``(xmin, ymin, xmax, ymax)`` in the dataset's CRS. Only nodes whose octree cube overlaps the box
            are fetched, and the points are then clipped exactly to the box, to the float edge. 2-D: the full z
            range is kept.
        resolution: target point spacing in CRS units, measured against :attr:`EptInfo.span`. The octree is
            visited only to the depth that satisfies it, so a wide view stays cheap. Mutually exclusive with
            ``depth``.
        depth: an explicit maximum octree depth (``0`` is the root), inclusive, when a caller would rather name
            a depth than a spacing. Mutually exclusive with ``resolution``.
        dimensions: dimension names to return, case-insensitive. ``None`` returns every dimension the package
            maps onto a :class:`~pointclouds.points.PointCloud` field. Selects what comes back, not what is
            decoded.

    Returns:
        A mapping of lower-case dimension name to array, under the same contract as every other reader: a
        dimension the schema does not carry comes back as a zero-length array, never as a missing key. A window
        that selects nothing gives zero-length arrays throughout.

    Raises:
        FileNotFoundError: when there is no ``ept.json`` at the source.
        ModuleNotFoundError: for a remote URL without the ``[cloud]`` extra.
        ValueError: when ``bbox`` is malformed, non-finite or inverted; when ``resolution`` is not positive and
            finite; when ``depth`` is negative; or when ``resolution`` and ``depth`` are both given.
        KeyError: when a requested dimension is not one this package reads.
        NotImplementedError: when the dataset's ``dataType`` is ``"zstandard"``.

    Examples:
        >>> arrays = read_window(  # doctest: +SKIP
        ...     "examples/data/sample-ept", bbox=(155000, 463000, 155020, 463020)
        ... )
        >>> arrays["x"].size  # doctest: +SKIP
        121
    """
    requested = _validate_window(bbox, resolution, depth, dimensions)  # before any read: no I/O on bad input
    store = _store(source)
    manifest = _read_manifest(store)
    return _window_from(store, manifest, bbox=bbox, resolution=resolution, depth=depth, requested=requested)


def _validate_window(
    bbox: tuple[float, float, float, float] | None,
    resolution: float | None,
    depth: int | None,
    dimensions: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Run every source-independent window check and resolve the dimension list, before any file is read.

    None of these needs the dataset, so running them first lets a malformed window be rejected before paying
    the manifest, hierarchy and node fetches - which matters on a remote source.

    Args:
        bbox: see :func:`read_window`.
        resolution: see :func:`read_window`.
        depth: see :func:`read_window`.
        dimensions: see :func:`read_window`.

    Returns:
        The resolved, lower-cased, de-duplicated dimension names.

    Raises:
        ValueError: on a bad bbox, resolution, depth, or both resolution and depth.
        KeyError: when a requested dimension is not one this package reads.
    """
    if resolution is not None and depth is not None:
        raise ValueError("pass either resolution= or depth=, not both; they are two ways to say the same thing")
    if resolution is not None and not (math.isfinite(resolution) and resolution > 0):
        raise ValueError(f"resolution must be a positive, finite number; got {resolution!r}")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 0):
        raise ValueError(f"depth must be a non-negative octree depth; got {depth!r}")
    if bbox is not None:
        _check_bbox(bbox)
    return _records.resolve(dimensions)


def _window_from(
    store: _Store,
    manifest: dict[str, Any],
    *,
    bbox: tuple[float, float, float, float] | None,
    resolution: float | None,
    depth: int | None,
    requested: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Run a window against an already-loaded store and manifest, with pre-validated arguments.

    Shared by :func:`read_window` and :func:`_read_window_with_crs`; both call :func:`_validate_window` first,
    so this does no validation.

    Args:
        store: the dataset store.
        manifest: the parsed ``ept.json``.
        bbox: see :func:`read_window`.
        resolution: see :func:`read_window`.
        depth: see :func:`read_window`.
        requested: the resolved dimension names from :func:`_validate_window`.

    Returns:
        The record-array mapping, as :func:`read_window` documents.
    """
    # An exact bbox clip needs x and y even when the caller did not ask for them; read them, then drop them.
    needed = requested if bbox is None else tuple(dict.fromkeys(("x", "y", *requested)))

    data_type = str(manifest["dataType"])
    schema = manifest.get("schema", [])
    extension = _node_extension(data_type)
    hierarchy = _Hierarchy(store)
    keys = _select_nodes(hierarchy, _root_cube(manifest), bbox, _max_depth(manifest, resolution, depth))

    parts = [_decode(store.read_bytes(f"ept-data/{key}.{extension}"), data_type, schema, needed) for key in keys]
    arrays = _concatenate(parts, needed)
    if bbox is not None:
        arrays = _records.clip_to_box(arrays, bbox)
        arrays = {name: arrays[name] for name in requested}
    return arrays


def _read_window_with_crs(
    source: Source,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    depth: int | None = None,
    dimensions: tuple[str, ...] | None = None,
) -> tuple[CRS | None, dict[str, np.ndarray]]:
    """Read a window and the dataset's CRS from a single manifest read.

    :class:`~pointclouds.points.PointCloud` needs both to build a windowed cloud; reading the manifest once
    avoids a second fetch of ``ept.json`` on a remote source.

    Args:
        source: a path or URL to an ``ept.json`` or its directory.
        bbox: see :func:`read_window`.
        resolution: see :func:`read_window`.
        depth: see :func:`read_window`.
        dimensions: see :func:`read_window`.

    Returns:
        ``(crs, arrays)`` - the dataset's CRS (or ``None``) and the record-array mapping.
    """
    requested = _validate_window(bbox, resolution, depth, dimensions)  # before any read: no I/O on bad input
    store = _store(source)
    manifest = _read_manifest(store)
    crs = _parse_crs(manifest)
    arrays = _window_from(store, manifest, bbox=bbox, resolution=resolution, depth=depth, requested=requested)
    return crs, arrays
