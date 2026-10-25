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

from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from pointclouds.io import las as _las

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

    def to_crs(self, crs: CRS | str, *, vertical: bool = False) -> PointCloud:
        """Reproject the cloud into another CRS with pyproj, returning a new cloud.

        The transform is built with ``Transformer.from_crs(..., always_xy=True)``, so coordinates are always
        read and written in ``(x, y)`` / easting-northing / longitude-latitude order regardless of the axis
        order the EPSG registry declares for either CRS (e.g. EPSG:4326's registry order is lat/lon, yet this
        returns lon in ``x`` and lat in ``y``).

        **Vertical datum.** By default (``vertical=False``) only x and y are transformed; z is carried over
        unchanged, still in the source vertical datum. This is the right choice when the target is a 2-D
        horizontal CRS - there is no vertical transformation to apply - but it means a 3-D reprojection is never
        done implicitly. To shift z as well, pass ``vertical=True``; this requires a compound or 3-D CRS on both
        sides (one that carries a vertical axis) and raises otherwise, so z is never silently left in the source
        datum when a vertical transform was asked for but is impossible.

        Attribute arrays (intensity, classification, return number) are carried over unchanged and stay aligned
        with the reprojected coordinates.

        Args:
            crs: the target CRS, as anything :class:`pyproj.CRS` accepts (an EPSG string, WKT, a ``CRS``).
            vertical: when ``True``, transform z as well as x/y; requires a vertical/compound CRS on both sides.

        Returns:
            A new, loaded :class:`PointCloud` in ``crs``, with x/y (and z when ``vertical``) transformed.

        Raises:
            ValueError: when :attr:`crs` is ``None`` (no source CRS to transform from), or when ``vertical`` is
                ``True`` but the source or target CRS carries no vertical axis.

        Examples:
            >>> import numpy as np
            >>> cloud = PointCloud(x=np.array([155000.0]), y=np.array([463000.0]), z=np.zeros(1), crs="EPSG:28992")
            >>> utm = cloud.to_crs("EPSG:32631")
            >>> utm.crs.to_epsg()
            32631
        """
        if self._crs is None:
            raise ValueError("cannot reproject a cloud whose crs is None; set a source CRS first")
        target = CRS.from_user_input(crs)
        if vertical and not (_has_vertical_axis(self._crs) and _has_vertical_axis(target)):
            raise ValueError(
                "vertical=True needs a compound or 3-D CRS (one with a vertical axis) on both sides; "
                f"got source {self._crs.to_string()!r} and target {target.to_string()!r}. Refusing rather "
                "than leaving z in the source datum."
            )
        transformer = Transformer.from_crs(self._crs, target, always_xy=True)
        if vertical:
            x, y, z = transformer.transform(self.x, self.y, self.z)
        else:
            x, y = transformer.transform(self.x, self.y)
            z = self.z
        return PointCloud(
            x=x,
            y=y,
            z=z,
            intensity=self.intensity,
            classification=self.classification,
            return_number=self.return_number,
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
