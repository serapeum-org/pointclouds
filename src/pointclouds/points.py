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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from pyproj import CRS

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
    :meth:`read_file`, which is header-first and loads records only when they are asked for.

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
            ValueError: if ``x``, ``y`` and ``z`` do not all share one length.
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

    @classmethod
    def read_file(cls, path: str | Path, *, lazy: bool = True) -> PointCloud:
        """Open a LAS/LAZ file.

        Args:
            path: the file to read.
            lazy: when ``True`` (the default) only the header is read; the record arrays decompress on first
                access. When ``False`` the records are read immediately.

        Returns:
            A cloud carrying the file's header metadata, and - unless ``lazy`` - its points.

        Raises:
            FileNotFoundError: when the path does not exist.
            ValueError: when the suffix is neither ``.las`` nor ``.laz``.

        Examples:
            >>> cloud = PointCloud.read_file("examples/data/sample.laz")  # doctest: +SKIP
            >>> cloud.count, cloud.crs is not None  # header only - no records touched  # doctest: +SKIP
            (400, True)
        """
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
        return cloud if lazy else cloud.load()

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
        """
        self.load()
        lookup = {"x": self._x, "y": self._y, "z": self._z, **self._attributes}
        try:
            cols = [np.asarray(lookup[name.lower()], dtype=float) for name in dimensions]
        except KeyError as exc:
            raise KeyError(f"unknown dimension {exc.args[0]!r}; this cloud holds {sorted(lookup)}") from None
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
        attributes = {
            name: values[selector] if values.size else values for name, values in self._attributes.items()
        }
        return PointCloud(
            x=self._x[selector], y=self._y[selector], z=self._z[selector], crs=self._crs, **attributes
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
