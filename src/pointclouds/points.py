"""The :class:`PointCloud` container.

A point cloud is held as parallel 1-D arrays - index ``i`` selects the i-th point across every field - so a filter
or decimation stays aligned across coordinates and attributes. This module is the nucleus of the package and must
import with the core dependency set alone (numpy / laspy / pyproj); it must never import GDAL or pyramids.

Reading is **header-first** (T-8). :meth:`PointCloud.read_file` opens a file and, by default, reads only its
header: ``crs``, ``bounds``, ``count`` and ``dimensions`` answer immediately, at a cost independent of the file's
size. The record arrays decompress on first access to ``x`` / ``y`` / ``z`` or to any attribute array, or eagerly
via :meth:`PointCloud.load`. The distinction that matters, and the reason ``count`` is not an alias for ``len``:

- ``cloud.count`` is the header's point count - free, no records read.
- ``len(cloud)`` is how many points are **in memory**, so on a lazy cloud it forces the read.

Reading can also be **windowed** (T-16). :meth:`PointCloud.read_window` - and ``read_file(bbox=, resolution=)``,
which delegates to it - asks a COPC file's octree for only the points inside a bounding box, or only a coarse
enough level of detail, instead of decompressing the whole cloud and masking it afterwards. A windowed cloud is
eager and carries no header, so its ``count`` and ``bounds`` describe the window rather than the file.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

from pointclouds.io import copc as _copc
from pointclouds.io import ept as _ept
from pointclouds.io import las as _las
from pointclouds.ops import clip as _clip
from pointclouds.ops import decimate as _decimate
from pointclouds.ops import filter as _filter
from pointclouds.ops import grid as _grid
from pointclouds.ops import merge as _merge

__all__ = ["PointCloud"]

#: Attribute arrays, in the order :meth:`PointCloud.dimensions` reports them, with the dtype each is held as.
_ATTRIBUTE_DTYPES: dict[str, type] = {
    "intensity": np.uint16,
    "classification": np.uint8,
    "return_number": np.uint8,
}


class PointCloud:
    """In-memory point cloud: parallel per-dimension arrays plus a CRS.

    The constructor builds an eager cloud from arrays already in hand. To read a file, use
    :meth:`read_file`, which is header-first and loads records only when they are asked for, or
    :meth:`read_window` to pull back only part of a COPC cloud.

    Attributes:
        x: ``(N,)`` float64 x-coordinates.
        y: ``(N,)`` float64 y-coordinates.
        z: ``(N,)`` float64 z-coordinates.
        intensity: ``(N,)`` uint16 return intensity, or empty when the file/caller carried none.
        classification: ``(N,)`` uint8 ASPRS class codes, or empty.
        return_number: ``(N,)`` uint8 return-number-within-pulse, or empty.
        crs: a :class:`pyproj.CRS`, or ``None`` when no SRS is known.

    Examples:
        >>> import numpy as np
        >>> cloud = PointCloud(x=np.arange(3.0), y=np.zeros(3), z=np.array([1.0, 2.0, 3.0]))
        >>> len(cloud), cloud.count
        (3, 3)
        >>> cloud.to_array(("X", "Z")).shape
        (3, 2)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        intensity: np.ndarray | None = None,
        classification: np.ndarray | None = None,
        return_number: np.ndarray | None = None,
        crs: CRS | str | None = None,
    ) -> None:
        """Build a cloud from arrays, rejecting coordinate arrays that disagree in length.

        Args:
            x: ``(N,)`` x-coordinates.
            y: ``(N,)`` y-coordinates.
            z: ``(N,)`` z-coordinates.
            intensity: ``(N,)`` return intensity, or ``None`` for an empty array.
            classification: ``(N,)`` ASPRS class codes, or ``None`` for an empty array.
            return_number: ``(N,)`` return numbers, or ``None`` for an empty array.
            crs: anything :class:`pyproj.CRS` accepts, or ``None``.

        Raises:
            ValueError: if ``x``, ``y`` and ``z`` do not all share one length, or if an attribute array is
                neither empty (absent) nor exactly one value per point.
        """
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._z = np.asarray(z, dtype=np.float64)
        self._attributes = {
            "intensity": _as_attribute(intensity, np.uint16),
            "classification": _as_attribute(classification, np.uint8),
            "return_number": _as_attribute(return_number, np.uint8),
        }
        self._crs = CRS.from_user_input(crs) if isinstance(crs, str) else crs
        self._header: _las.LasHeaderInfo | None = None
        self._source: Path | None = None
        self._loaded = True

        n = len(self._x)
        if not (len(self._y) == n and len(self._z) == n):
            raise ValueError(f"x/y/z must share a length; got {len(self._x)}, {len(self._y)}, {len(self._z)}")
        for name, values in self._attributes.items():
            if values.size and values.size != n:
                raise ValueError(f"{name} has length {values.size}, expected 0 or {n}")

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        lazy: bool | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        resolution: float | None = None,
        level: int | range | None = None,
    ) -> PointCloud:
        """Open a LAS/LAZ file, or a window onto a COPC one.

        Args:
            path: the file to read. A path only - to read a window from an open file object (an ``fsspec``
                handle), call :meth:`read_window` directly; the LAS/LAZ engine behind the non-windowed read
                takes a path.
            lazy: when left unset or ``True`` only the header is read and the record arrays decompress on first
                access; ``False`` reads the records immediately. It has no meaning for a windowed read, where
                the window *is* the read, so passing it together with ``bbox`` / ``resolution`` / ``level`` is
                refused rather than silently ignored.
            bbox: ``(xmin, ymin, xmax, ymax)`` in the file's CRS. Passing it makes this a windowed read, which
                requires a COPC file - see :meth:`read_window`.
            resolution: target point spacing; makes this a windowed read. Mutually exclusive with ``level``.
            level: an exact octree level or a ``range`` of them; makes this a windowed read. Mutually exclusive
                with ``resolution``.

        Returns:
            A cloud carrying the file's header metadata, and - unless ``lazy`` is ``False`` - its points. With a
            window, whatever :meth:`read_window` returns.

        Raises:
            FileNotFoundError: when the path does not exist.
            ValueError: when the suffix is neither ``.las`` nor ``.laz``; when ``lazy`` is passed together with a
                window argument; or, for a windowed read, anything :meth:`read_window` rejects.

        Examples:
            Without a window the read is header-first: the metadata costs nothing and the records wait:

            >>> cloud = PointCloud.read_file("examples/data/sample.laz")  # doctest: +SKIP
            >>> cloud.count, cloud.crs is not None  # header only - no records touched  # doctest: +SKIP
            (400, True)
            >>> cloud.loaded  # doctest: +SKIP
            False

            Passing a window makes it a COPC read instead, and the cloud comes back already populated:

            >>> window = PointCloud.read_file(  # doctest: +SKIP
            ...     "examples/data/sample.copc.laz", bbox=(155000, 463000, 155020, 463020)
            ... )
            >>> window.count, window.loaded  # doctest: +SKIP
            (121, True)
        """
        if bbox is not None or resolution is not None or level is not None:
            if lazy is not None:
                raise ValueError("lazy has no meaning with a window; a windowed read is always eager")
            return cls.read_window(path, bbox=bbox, resolution=resolution, level=level)
        header = _las.read_header(path)
        cloud = cls.__new__(cls)  # bypass __init__: a lazy cloud has a header but no arrays yet
        cloud._x = _EMPTY_F8
        cloud._y = _EMPTY_F8
        cloud._z = _EMPTY_F8
        cloud._attributes = {name: np.empty(0, dtype=dtype) for name, dtype in _ATTRIBUTE_DTYPES.items()}
        cloud._crs = header.crs
        cloud._header = header
        cloud._source = Path(path)
        cloud._loaded = False
        # lazy is None when unset, which means lazy; only an explicit lazy=False forces an eager load.
        return cloud.load() if lazy is False else cloud

    @classmethod
    def read_window(
        cls,
        source: _copc.Source | _ept.Source,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        resolution: float | None = None,
        level: int | range | None = None,
        dimensions: tuple[str, ...] | None = None,
    ) -> PointCloud:
        """Read only part of a cloud-optimised source - a COPC file or an EPT dataset - using its octree.

        This is the path a large cloud is meant to be read through: ``bbox`` prunes the octree to the nodes
        whose cube overlaps the box, and ``resolution`` stops the descent at the coarsest level that still
        satisfies it. Reading the whole thing and masking it afterwards costs the full decode; this does not.

        The source is dispatched by shape: an EPT dataset (a directory or URL holding an ``ept.json``) goes to
        :func:`pointclouds.io.ept.read_window`, anything else to :func:`pointclouds.io.copc.read_window`.

        The returned cloud is **eager and header-less**: :attr:`count` and :attr:`bounds` describe the points
        actually read, not the file they came from. Use :func:`pointclouds.io.copc.read_header` /
        :func:`pointclouds.io.ept.read_header` for the source's own count, bounds and resolution.

        Args:
            source: a COPC file (path or open file object) or an EPT dataset (path or URL to the directory or
                its ``ept.json``). A remote source goes through ``fsspec`` (the ``[cloud]`` extra).
            bbox: ``(xmin, ymin, xmax, ymax)`` in the source's CRS. 2-D: the full z range is kept.
            resolution: target point spacing, measured against the octree's root spacing (COPC) or ``span``
                (EPT). Mutually exclusive with ``level``.
            level: the level-of-detail knob. For COPC, an exact octree level or a ``range``. For EPT, an ``int``
                read as a maximum depth (``0`` = root), inclusive; a ``range`` is not accepted for EPT. Mutually
                exclusive with ``resolution``.
            dimensions: which attribute arrays to carry, case-insensitive. ``None`` reads every mapped
                dimension; ``x`` / ``y`` / ``z`` are always read whatever is asked for, since a cloud needs its
                coordinates. Selects what is returned, not what is decoded.

        Returns:
            A cloud holding only the selected points, with the source's CRS. A window that selects nothing gives
            an empty cloud, whose :attr:`count` is ``0``.

        Raises:
            FileNotFoundError: when the path does not exist.
            ValueError: when the source is malformed for its kind; when ``bbox`` is malformed; when
                ``resolution`` is not positive; when ``resolution`` and ``level`` are both given; or when
                ``level`` is a ``range`` against an EPT source.

        Examples:
            A box reads a corner of the cloud, and the resulting cloud describes the window it holds:

            >>> window = PointCloud.read_window(  # doctest: +SKIP
            ...     "examples/data/sample.copc.laz", bbox=(155000, 463000, 155020, 463020)
            ... )
            >>> window.count  # 121 of the file's 1600 points  # doctest: +SKIP
            121

            An EPT dataset reads the same way, by directory:

            >>> coarse = PointCloud.read_window("examples/data/sample-ept", resolution=50.0)  # doctest: +SKIP
            >>> coarse.count, coarse.crs.to_epsg()  # doctest: +SKIP
            (100, 28992)
        """
        # x/y/z are forced into the request: a cloud is built from its coordinates, so a caller asking only for
        # an attribute still needs them. The result is then splatted into the constructor, which is safe because
        # the record contract's keys are exactly the constructor's point parameters (asserted in the tests).
        requested = None if dimensions is None else tuple(dict.fromkeys(("x", "y", "z", *dimensions)))
        # EPT sources are only ever a path or URL (a directory holding ept.json); a file object is always COPC.
        if isinstance(source, (str, os.PathLike)) and _ept.is_ept(source):
            crs, arrays = _ept._read_window_with_crs(
                source, bbox=bbox, resolution=resolution, depth=_ept_depth(level), dimensions=requested
            )
        else:
            crs, arrays = _copc._read_window_with_crs(
                source, bbox=bbox, resolution=resolution, level=level, dimensions=requested
            )
        return cls(crs=crs, **arrays)

    def load(self) -> PointCloud:
        """Read this cloud's record arrays if they are not in memory yet.

        Returns:
            ``self``, so a read can be chained (``PointCloud.read_file(p).load()``).

        Raises:
            RuntimeError: when the cloud has no source file to load from, which cannot happen for a cloud built
                by :meth:`read_file` and always holds for one built from arrays (already loaded).
        """
        if self._loaded:
            return self
        if self._source is None:  # pragma: no cover - unreachable via the public API
            raise RuntimeError("cloud has no source file to load records from")
        arrays = _las.read_points(self._source)
        self._x = arrays["x"]
        self._y = arrays["y"]
        self._z = arrays["z"]
        self._attributes = {name: arrays[name] for name in _ATTRIBUTE_DTYPES}
        self._loaded = True
        return self

    @property
    def loaded(self) -> bool:
        """Whether the record arrays are in memory. Reading this never triggers a load."""
        return self._loaded

    @property
    def source(self) -> Path | None:
        """The file this cloud was read from, or ``None`` when it was built from arrays."""
        return self._source

    @property
    def x(self) -> np.ndarray:
        """``(N,)`` float64 x-coordinates, loading the records first if needed."""
        return self.load()._x

    @property
    def y(self) -> np.ndarray:
        """``(N,)`` float64 y-coordinates, loading the records first if needed."""
        return self.load()._y

    @property
    def z(self) -> np.ndarray:
        """``(N,)`` float64 z-coordinates, loading the records first if needed."""
        return self.load()._z

    @property
    def intensity(self) -> np.ndarray:
        """``(N,)`` uint16 return intensity (empty when absent), loading the records first if needed."""
        return self.load()._attributes["intensity"]

    @property
    def classification(self) -> np.ndarray:
        """``(N,)`` uint8 ASPRS class codes (empty when absent), loading the records first if needed."""
        return self.load()._attributes["classification"]

    @property
    def return_number(self) -> np.ndarray:
        """``(N,)`` uint8 return numbers (empty when absent), loading the records first if needed."""
        return self.load()._attributes["return_number"]

    @property
    def crs(self) -> CRS | None:
        """The cloud's CRS as a :class:`pyproj.CRS`, or ``None``. Answers from the header; never loads records."""
        return self._crs

    @property
    def count(self) -> int:
        """How many points the cloud holds, from the header when there is one.

        This is the cheap question: on a lazy cloud it answers without decompressing a single record, which is
        what ``len(cloud)`` cannot do. The two agree once the records are in memory.
        """
        if self._header is not None:
            return int(self._header.count)
        return int(self._x.shape[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)``, from the header when there is one, else computed from the arrays.

        Raises:
            ValueError: when the cloud is empty and has no header to take bounds from.
        """
        if self._header is not None:
            return self._header.bounds
        if self._x.size == 0:
            raise ValueError("an empty cloud with no header has no bounds")
        return (float(self._x.min()), float(self._y.min()), float(self._x.max()), float(self._y.max()))

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The dimension names available on this cloud.

        For a cloud read from a file this is what the point format declares, straight from the header. For one
        built from arrays it is the coordinates plus whichever attribute arrays are non-empty.
        """
        if self._header is not None:
            return self._header.dimensions
        present = [name for name, values in self._attributes.items() if values.size]
        return ("X", "Y", "Z", *present)

    def to_array(self, dimensions: tuple[str, ...] = ("X", "Y", "Z")) -> np.ndarray:
        """Stack the requested dimensions column-wise into an ``(N, k)`` array.

        Args:
            dimensions: dimension names, case-insensitive (``"X"``/``"Y"``/``"Z"`` map to the coordinate arrays).

        Returns:
            ``(N, k)`` float64 array, one column per requested dimension.

        Raises:
            KeyError: when a requested dimension is not one this container holds.
            ValueError: when a requested dimension is empty (the file/caller carried none of it) while others
                are populated, which would otherwise surface as an opaque numpy broadcast error.
        """
        self.load()
        lookup = {"x": self._x, "y": self._y, "z": self._z, **self._attributes}
        try:
            cols = [np.asarray(lookup[name.lower()], dtype=float) for name in dimensions]
        except KeyError as exc:
            raise KeyError(f"unknown dimension {exc.args[0]!r}; this cloud holds {sorted(lookup)}") from None
        width = len(self._x)
        empty = [name for name, col in zip(dimensions, cols, strict=True) if col.size != width]
        if empty:
            raise ValueError(
                f"cannot stack dimension(s) {empty} - this cloud carries none of them ({width} points, but "
                f"those arrays are empty); request only the dimensions it holds"
            )
        return np.column_stack(cols)

    def filter(
        self,
        *,
        classification: int | Iterable[int] | None = None,
        return_number: int | Iterable[int] | None = None,
        intensity: int | Iterable[int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> PointCloud:
        """Keep the points matching every given criterion, returning a new cloud.

        Each attribute criterion is a membership test - a scalar keeps points equal to it, an iterable keeps
        points whose value is in the set - and the criteria are ANDed together. Passing no criterion returns a
        copy of the whole cloud.

        Args:
            classification: an ASPRS class code, or an iterable of them, to keep (e.g. ``2`` for ground).
            return_number: a return number, or an iterable of them, to keep.
            intensity: an intensity value, or an iterable of them, to keep.
            bbox: ``(xmin, ymin, xmax, ymax)`` extent to keep, edges included.

        Returns:
            A new cloud holding only the matching points, every array subset in step. The original is untouched.

        Raises:
            ValueError: when a criterion names an attribute this cloud does not carry, or ``bbox`` is malformed.

        Examples:
            >>> import numpy as np
            >>> cloud = PointCloud(
            ...     x=np.arange(4.0), y=np.zeros(4), z=np.zeros(4),
            ...     classification=np.array([2, 5, 2, 6], dtype=np.uint8),
            ... )
            >>> len(cloud.filter(classification=2))
            2
        """
        mask = _filter.build_mask(
            self, classification=classification, return_number=return_number, intensity=intensity, bbox=bbox
        )
        return self._take(mask)

    def decimate(
        self,
        *,
        target: int | None = None,
        fraction: float | None = None,
        method: str = "random",
        voxel_size: float | None = None,
        seed: int | None = None,
    ) -> PointCloud:
        """Thin the cloud, returning a new cloud with fewer points.

        Args:
            target: for ``method="random"``, how many points to keep; never more than the cloud holds.
            fraction: for ``method="random"``, the share of points to keep, in ``(0, 1]``. A positive fraction
                over a non-empty cloud keeps at least one point.
            method: ``"random"`` for a uniform sample without replacement, ``"voxel"`` for one point per cube.
            voxel_size: for ``method="voxel"``, the cube edge length in CRS units.
            seed: seeds the random sample so a run is reproducible; ignored by ``"voxel"``.

        Returns:
            A new, thinned cloud; the original is untouched. Point order is preserved.

        Raises:
            ValueError: when the arguments for the chosen method are missing, ambiguous, or out of range, or when
                ``method`` is neither ``"random"`` nor ``"voxel"``.

        Examples:
            >>> import numpy as np
            >>> cloud = PointCloud(x=np.arange(10.0), y=np.zeros(10), z=np.zeros(10))
            >>> len(cloud.decimate(target=3, seed=0))
            3
        """
        indices = _decimate.select_indices(
            self, target=target, fraction=fraction, method=method, voxel_size=voxel_size, seed=seed
        )
        return self._take(indices)

    def clip(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        polygon: np.ndarray | None = None,
    ) -> PointCloud:
        """Keep the points inside an extent or a polygon, returning a new cloud.

        Exactly one of ``bbox`` or ``polygon`` must be given. Both forms are pure numpy, so clipping needs no GIS
        stack; see :mod:`pointclouds.ops.clip` for the scope of the polygon form (a single ring, no holes).

        Args:
            bbox: ``(xmin, ymin, xmax, ymax)`` extent, edges included.
            polygon: an ``(M, 2)`` array of ``(x, y)`` vertices forming a single ring (closed or open).

        Returns:
            A new cloud holding only the points inside the region; the original is untouched.

        Raises:
            ValueError: when neither or both of ``bbox`` and ``polygon`` are given, or ``polygon`` is malformed.

        Examples:
            >>> import numpy as np
            >>> cloud = PointCloud(x=np.array([0.0, 5.0, 9.0]), y=np.zeros(3), z=np.zeros(3))
            >>> len(cloud.clip(bbox=(4.0, -1.0, 10.0, 1.0)))
            2
        """
        mask = _clip.clip_mask(self, bbox=bbox, polygon=polygon)
        return self._take(mask)

    @classmethod
    def merge(cls, clouds: Sequence[PointCloud]) -> PointCloud:
        """Concatenate several clouds into one, rejecting a mix of CRSs.

        A zero-point cloud (for example one that filtered to empty) is a no-op participant: it contributes no
        points and takes no part in resolving the CRS or which attributes are present.

        Args:
            clouds: the clouds to merge, in order. Each is loaded if its records are not in memory yet.

        Returns:
            A new cloud holding every point from every input, in input order.

        Raises:
            ValueError: when ``clouds`` is empty, the clouds disagree on CRS, or an attribute is present on some
                clouds but absent on others.

        Examples:
            >>> import numpy as np
            >>> a = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))
            >>> b = PointCloud(x=np.ones(3), y=np.zeros(3), z=np.zeros(3))
            >>> len(PointCloud.merge([a, b]))
            5
        """
        return cls(**_merge.merge_arrays(clouds))

    def to_grid(
        self,
        *,
        cell_size: float,
        aggregate: str = "mean",
        mode: str = "bin",
        radius: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        nodata: float = np.nan,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], CRS | None]:
        """Grid the cloud onto a regular raster surface, without GDAL.

        The aggregation names follow PDAL's ``writers.gdal`` vocabulary, and the window defaults to ``"bin"`` (only
        points inside the pixel contribute); see :mod:`pointclouds.ops.grid` for the full rationale.

        Args:
            cell_size: the pixel edge length in CRS units; must be positive.
            aggregate: ``min``, ``max``, ``mean``, ``count``, ``stdev``, ``idw``, or a percentile ``p<n>``.
            mode: ``"bin"`` (points inside the pixel) or ``"radius"`` (points within ``radius`` of the centre).
            radius: the search radius for ``mode="radius"``; defaults to ``cell_size * sqrt(2)`` when ``None``.
            bounds: ``(xmin, ymin, xmax, ymax)`` extent to grid; defaults to the cloud's own bounds.
            nodata: the value written to empty cells (``count`` excepted, which writes ``0``).

        Returns:
            ``(array, transform, crs)``: an ``(nrows, ncols)`` float64 array with row 0 northernmost, a GDAL-style
            affine 6-tuple ``(xmin, cell_size, 0, ymax, 0, -cell_size)``, and the cloud's CRS.

        Raises:
            ValueError: on a non-positive ``cell_size``; a non-positive-area extent (a degenerate ``bounds``, or
                the default extent of a flat cloud whose points share a single x or a single y); an unknown
                ``mode``; or an unrecognised ``aggregate``.

        Examples:
            >>> import numpy as np
            >>> ground = PointCloud(x=np.array([0.5, 1.5]), y=np.array([0.5, 0.5]), z=np.array([10.0, 20.0]))
            >>> array, _, _ = ground.to_grid(cell_size=1.0, aggregate="min", bounds=(0, 0, 2, 1))
            >>> array.tolist()
            [[10.0, 20.0]]
        """
        return _grid.to_grid(
            self, cell_size=cell_size, aggregate=aggregate, mode=mode, radius=radius, bounds=bounds, nodata=nodata
        )

    def _take(self, selector: np.ndarray) -> PointCloud:
        """Return a new cloud holding the selected points, every coordinate and attribute subset in step.

        Args:
            selector: a boolean mask of length ``len(self)``, or an integer index array into the points.

        Returns:
            A new, eager cloud. An attribute the cloud does not carry (an empty array) stays empty rather than
            being indexed, so the empty-means-absent convention is preserved.
        """
        self.load()
        attributes = {name: values[selector] if values.size else values for name, values in self._attributes.items()}
        return PointCloud(x=self._x[selector], y=self._y[selector], z=self._z[selector], crs=self._crs, **attributes)

    def to_crs(
        self,
        crs: CRS | str | int,
        *,
        vertical: bool = False,
        on_invalid: Literal["raise", "mask", "drop"] = "raise",
    ) -> PointCloud:
        """Reproject the cloud into another CRS with pyproj, returning a new cloud.

        The transform is built with ``Transformer.from_crs(..., always_xy=True)``, so coordinates are always
        read and written in ``(x, y)`` / easting-northing / longitude-latitude order regardless of the axis
        order the EPSG registry declares for either CRS (e.g. EPSG:4326's registry order is lat/lon, yet this
        returns lon in ``x`` and lat in ``y``).

        **Vertical datum.** By default (``vertical=False``) only x and y are transformed; z is carried over
        unchanged, still in the source vertical datum. This is the right choice when the target is a 2-D
        horizontal CRS - there is no vertical transformation to apply - but it means a 3-D reprojection is never
        done implicitly. To shift z as well, pass ``vertical=True``; this requires a compound or 3-D CRS on both
        sides (one that carries a vertical axis) and raises ``ValueError`` otherwise. The most accurate vertical
        operation is then used, and if its grid (e.g. a geoid model) is not installed the transform is refused
        with ``ValueError`` rather than falling back to a less-accurate operation or to one that passes z through
        untouched. The trade-off is strict - a legitimate vertical reprojection is refused on an install lacking
        that grid - but it guarantees z is never silently left in the source datum when a vertical transform was
        asked for, and it does so the same way on every pyproj/PROJ build.

        **Out-of-domain points** are handled by ``on_invalid``. By default (``"raise"``) a coordinate outside the
        transform's area of use raises ``ProjError``, and because that is all-or-nothing a single non-computable
        point aborts the whole reprojection. A cloud that may carry sentinel or noise coordinates (e.g. stray
        ``0, 0``) can instead pass ``on_invalid="mask"`` to set each unreprojectable point to ``NaN`` in x/y/z
        (keeping the point count and attribute alignment), or ``on_invalid="drop"`` to remove those points and
        their attribute rows (shrinking the cloud). Both lenient modes keep every other point.

        Present attribute arrays (intensity, classification, return number) are carried over by value - the
        returned cloud owns independent copies, so editing it never reaches back into the source - and stay
        aligned with the reprojected coordinates; an absent attribute is carried over as the same empty array. The
        whole cloud is materialised: a lazy source is loaded first, and the cost is O(N) in points, with no
        chunked/streaming path.

        Args:
            crs: the target CRS, as anything :class:`pyproj.CRS` accepts (an int or string EPSG code, WKT, a
                ``CRS``).
            vertical: when ``True``, transform z as well as x/y; requires a vertical/compound CRS on both sides.
            on_invalid: what to do with points outside the transform's domain - ``"raise"`` (the default; one bad
                point raises ``ProjError``), ``"mask"`` (set them to ``NaN``, preserving count and alignment), or
                ``"drop"`` (remove them and their attribute rows).

        Returns:
            A new, loaded :class:`PointCloud` in ``crs``, with x/y (and z when ``vertical``) transformed. With
            ``on_invalid="drop"`` it holds only the points that reprojected; otherwise it has the same count as
            ``self``.

        Raises:
            ValueError: when ``on_invalid`` is not one of ``"raise"``/``"mask"``/``"drop"``; when :attr:`crs` is
                ``None`` (no source CRS to transform from); when ``vertical`` is ``True`` but the source or target
                CRS carries no vertical axis; or when ``vertical`` is ``True`` and the required vertical-datum
                operation (e.g. a geoid grid) is not installed.
            pyproj.exceptions.ProjError: when ``on_invalid="raise"`` and a point falls outside the transform's
                domain (a single such point aborts the whole reprojection).

        Examples:
            - Reproject a Dutch RD New cloud into UTM zone 31N; the result reports the target CRS:
                ```python
                >>> import numpy as np
                >>> cloud = PointCloud(x=np.array([155000.0]), y=np.array([463000.0]), z=np.zeros(1),
                ...                    crs="EPSG:28992")
                >>> utm = cloud.to_crs("EPSG:32631")
                >>> utm.crs.to_epsg()
                32631

                ```
            - ``always_xy=True`` means longitude lands in ``x`` and latitude in ``y`` even for EPSG:4326,
              whose registry order is lat/lon:
                ```python
                >>> import numpy as np
                >>> cloud = PointCloud(x=np.array([155000.0]), y=np.array([463000.0]), z=np.zeros(1),
                ...                    crs="EPSG:28992")
                >>> wgs = cloud.to_crs("EPSG:4326")
                >>> round(float(wgs.x[0]), 2), round(float(wgs.y[0]), 2)
                (5.39, 52.16)

                ```
            - A cloud with no CRS has nothing to transform from, so it refuses:
                ```python
                >>> import numpy as np
                >>> cloud = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))
                >>> cloud.to_crs("EPSG:4326")
                Traceback (most recent call last):
                    ...
                ValueError: cannot reproject a cloud whose crs is None; set a source CRS first

                ```
        """
        if on_invalid not in ("raise", "mask", "drop"):
            raise ValueError(f"on_invalid must be 'raise', 'mask' or 'drop'; got {on_invalid!r}")
        if self._crs is None:
            raise ValueError("cannot reproject a cloud whose crs is None; set a source CRS first")
        target = CRS.from_user_input(crs)
        if vertical and not (_has_vertical_axis(self._crs) and _has_vertical_axis(target)):
            raise ValueError(
                "vertical=True needs a compound or 3-D CRS (one with a vertical axis) on both sides; "
                f"got source {self._crs.to_string()!r} and target {target.to_string()!r}. Refusing rather "
                "than leaving z in the source datum."
            )
        errcheck = on_invalid == "raise"
        if vertical:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the "best unavailable" warning is turned into a raise below
                group = TransformerGroup(self._crs, target, always_xy=True)
            if not group.best_available:
                raise ValueError(
                    "vertical=True needs the most accurate vertical operation between "
                    f"{self._crs.to_string()!r} and {target.to_string()!r}, but its grid (e.g. a geoid model) "
                    "is not installed; refusing rather than leaving z in the source datum."
                )
            x, y, z = group.transformers[0].transform(self.x, self.y, self.z, errcheck=errcheck)
        else:
            transformer = Transformer.from_crs(self._crs, target, always_xy=True)
            x, y = transformer.transform(self.x, self.y, errcheck=errcheck)
            z = self.z.copy()
        intensity = self.intensity.copy()
        classification = self.classification.copy()
        return_number = self.return_number.copy()
        if on_invalid != "raise":
            x, y, z = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if vertical:
                valid &= np.isfinite(z)
            if on_invalid == "mask":
                x[~valid] = y[~valid] = z[~valid] = np.nan
            else:  # "drop" - remove the unreprojectable points and their attributes, keeping alignment
                x, y, z = x[valid], y[valid], z[valid]
                intensity = intensity[valid] if intensity.size else intensity
                classification = classification[valid] if classification.size else classification
                return_number = return_number[valid] if return_number.size else return_number
        return PointCloud(
            x=x,
            y=y,
            z=z,
            intensity=intensity,
            classification=classification,
            return_number=return_number,
            crs=target,
        )

    def __len__(self) -> int:
        """Return how many points are in memory, loading the records first if needed.

        Use :attr:`count` instead when the header's number is enough - that one is free.
        """
        return int(self.load()._x.shape[0])

    def __repr__(self) -> str:
        """Describe the cloud without loading anything."""
        crs = self._crs.to_string() if self._crs is not None else None
        state = "loaded" if self._loaded else "lazy"
        return f"PointCloud(count={self.count}, crs={crs!r}, {state})"


_EMPTY_F8 = np.empty(0, dtype=np.float64)


def _has_vertical_axis(crs: CRS) -> bool:
    """Whether ``crs`` carries height, so a vertical datum shift is meaningful.

    True for a compound CRS (horizontal + vertical), a 3-D CRS (three axes), or a bare vertical CRS; false for a
    plain 2-D horizontal CRS, where there is no z to transform.

    Args:
        crs: the CRS to inspect.

    Returns:
        ``True`` when the CRS has a vertical component.
    """
    return crs.is_compound or crs.is_vertical or len(crs.axis_info) >= 3


def _as_attribute(values: np.ndarray | None, dtype: type) -> np.ndarray:
    """Return ``values`` as an array of ``dtype``, mapping ``None`` to a zero-length array.

    Args:
        values: the array to normalise, or ``None``.
        dtype: the dtype to cast to.

    Returns:
        An array of ``dtype``; zero-length when ``values`` is ``None``.
    """
    if values is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _ept_depth(level: int | range | None) -> int | None:
    """Translate the unified ``level`` argument to the EPT ``depth`` (a maximum octree depth).

    COPC takes an exact level or a range; EPT takes a maximum depth. An ``int`` level maps straight to that
    maximum depth. A ``range`` has no EPT equivalent, so it is refused rather than guessed.

    Args:
        level: the ``level`` argument from :meth:`PointCloud.read_window`.

    Returns:
        The EPT ``depth`` (an ``int``), or ``None`` when ``level`` is ``None``.

    Raises:
        ValueError: when ``level`` is a ``range`` - EPT reads down to a single maximum depth, so pass an int
            level or a resolution instead.
    """
    if isinstance(level, range):
        raise ValueError(
            "an EPT source takes a maximum depth, not a level range; pass level=<int> (a max depth) or "
            "resolution=<spacing>"
        )
    return level
