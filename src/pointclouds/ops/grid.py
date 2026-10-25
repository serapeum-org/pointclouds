"""Grid a cloud to a raster surface (``to_grid``). (Phase 2 - T-11).

NumPy binning plus a SciPy KD-tree for the radius window, no GDAL. The aggregation names follow PDAL's
``writers.gdal`` vocabulary exactly - ``min``, ``max``, ``mean``, ``count``, ``stdev``, ``idw`` and ``p<n>`` -
so a reader arriving from PDAL meets the same words.

Decision - the default window is ``mode="bin"`` (the §6/§9 open question, settled here).
    PDAL defaults to a radius window with a ``binmode`` opt-in; this package does the opposite, because a DSM/DTM
    caller almost always means "the points that fall in this pixel", and digital-rivers' bare-earth path only
    ever uses bin mode. The choice is made explicit in the signature rather than implied: pass ``mode="radius"``
    for PDAL's behaviour, where every point within ``radius`` of a cell centre contributes.

The result is ``(array, transform, crs)``: a plain NumPy array, a GDAL-style affine 6-tuple
``(x_origin, cell_size, 0, y_origin, 0, -cell_size)``, and the cloud's :class:`pyproj.CRS` (or ``None``) - never a
GDAL object. Turning that into a pyramids ``Dataset`` is the separate ``[gis]`` step in
:mod:`pointclouds.convert` (T-12).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from pyproj import CRS

    from pointclouds.points import PointCloud

__all__ = ["to_grid"]

#: Matches a percentile aggregate such as ``p95`` or ``p99.5`` and captures the percentile.
_PERCENTILE = re.compile(r"^p(\d{1,3}(?:\.\d+)?)$")

#: Aggregates that need no per-point distance, so the plain reducers serve them.
_SIMPLE = frozenset({"min", "max", "mean", "count", "stdev"})

#: Floor on the point-to-centre distance used by ``idw``, so a point sitting on a cell centre does not blow up.
_IDW_EPS = 1e-12


def to_grid(
    cloud: PointCloud,
    *,
    cell_size: float,
    aggregate: str = "mean",
    mode: str = "bin",
    radius: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    nodata: float = np.nan,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], CRS | None]:
    """Bin or interpolate the cloud onto a regular grid.

    Args:
        cloud: the cloud to grid; its records are loaded if they are not in memory yet.
        cell_size: the pixel edge length in CRS units; must be positive.
        aggregate: how the z-values in (or near) a cell are reduced - one of ``min``, ``max``, ``mean``,
            ``count``, ``stdev``, ``idw``, or a percentile ``p<n>`` such as ``p95`` (PDAL ``writers.gdal`` names).
        mode: ``"bin"`` (the default) counts only points inside the pixel; ``"radius"`` counts every point within
            ``radius`` of the pixel centre.
        radius: the search radius for ``mode="radius"``; defaults to ``cell_size * sqrt(2)`` (PDAL's default)
            when ``None``. Ignored in bin mode.
        bounds: ``(xmin, ymin, xmax, ymax)`` extent to grid; defaults to the cloud's own bounds.
        nodata: the value written to empty cells. The default ``NaN`` forces a float grid. ``count`` is the one
            exception - an empty cell counts ``0``, not ``nodata``.

    Returns:
        ``(array, transform, crs)``: an ``(nrows, ncols)`` float64 array (row 0 is the northernmost), a GDAL-style
        affine 6-tuple ``(xmin, cell_size, 0, ymax, 0, -cell_size)``, and the cloud's CRS.

    Raises:
        ValueError: when ``cell_size`` is not positive, when ``bounds`` has a non-positive extent, when ``mode``
            is neither ``"bin"`` nor ``"radius"``, or when ``aggregate`` is not a recognised name.

    Examples:
        >>> import numpy as np
        >>> from pointclouds import PointCloud
        >>> cloud = PointCloud(x=np.array([0.5, 1.5]), y=np.array([0.5, 0.5]), z=np.array([10.0, 20.0]))
        >>> array, transform, _ = cloud.to_grid(cell_size=1.0, aggregate="min", bounds=(0, 0, 2, 1))
        >>> array.tolist()
        [[10.0, 20.0]]
        >>> transform
        (0.0, 1.0, 0.0, 1.0, 0.0, -1.0)
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive; got {cell_size}")
    if mode not in ("bin", "radius"):
        raise ValueError(f"unknown mode {mode!r}; expected 'bin' or 'radius'")
    reducer = _reducer(aggregate)  # validate the aggregate before any heavy work

    xmin, ymin, xmax, ymax = cloud.bounds if bounds is None else bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"to_grid needs a positive-area bounds; got {(xmin, ymin, xmax, ymax)}")
    ncols = math.ceil((xmax - xmin) / cell_size)
    nrows = math.ceil((ymax - ymin) / cell_size)

    x, y, z = cloud.x, cloud.y, cloud.z
    if mode == "bin":
        cell, values, distance = _bin_contributions(x, y, z, xmin, ymax, cell_size, ncols, nrows)
    else:
        cell, values, distance = _radius_contributions(
            x, y, z, xmin, ymax, cell_size, ncols, nrows, radius if radius is not None else cell_size * math.sqrt(2)
        )

    flat = reducer(cell, values, distance, ncols * nrows, nodata)
    transform = (float(xmin), float(cell_size), 0.0, float(ymax), 0.0, -float(cell_size))
    return flat.reshape(nrows, ncols), transform, cloud.crs


def _bin_contributions(x, y, z, xmin, ymax, cell_size, ncols, nrows):
    """Assign each point to the one cell it falls in, dropping points outside the grid.

    Returns the flat cell index, the z-value and the distance to that cell's centre, one entry per kept point.
    """
    col = np.floor((x - xmin) / cell_size).astype(np.int64)
    row = np.floor((ymax - y) / cell_size).astype(np.int64)
    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    col, row = col[inside], row[inside]

    centre_x = xmin + (col + 0.5) * cell_size
    centre_y = ymax - (row + 0.5) * cell_size
    distance = np.hypot(x[inside] - centre_x, y[inside] - centre_y)
    return row * ncols + col, z[inside], distance


def _radius_contributions(x, y, z, xmin, ymax, cell_size, ncols, nrows, radius):
    """Let every point within ``radius`` of a cell centre contribute to that cell.

    A point can land in several cells. Returns the flat cell index, the z-value and the point-to-centre distance,
    one entry per (cell, point) pair.
    """
    ncells = ncols * nrows
    rows, cols = np.divmod(np.arange(ncells), ncols)
    centre_x = xmin + (cols + 0.5) * cell_size
    centre_y = ymax - (rows + 0.5) * cell_size

    tree = cKDTree(np.column_stack([x, y]))
    neighbours = tree.query_ball_point(np.column_stack([centre_x, centre_y]), r=radius)

    counts = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=ncells)
    cell = np.repeat(np.arange(ncells), counts)
    points = np.fromiter((p for n in neighbours for p in n), dtype=np.int64, count=int(counts.sum()))
    distance = np.hypot(x[points] - centre_x[cell], y[points] - centre_y[cell])
    return cell, z[points], distance


def _reducer(aggregate: str):
    """Return the reducer for an aggregate name, or raise when the name is not recognised."""
    if aggregate in _SIMPLE:
        return lambda cell, values, distance, ncells, nodata: _reduce_simple(
            aggregate, cell, values, ncells, nodata
        )
    if aggregate == "idw":
        return _reduce_idw
    match = _PERCENTILE.match(aggregate)
    if match:
        percentile = float(match.group(1))
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile out of range in {aggregate!r}; expected p0..p100")
        return lambda cell, values, distance, ncells, nodata: _reduce_percentile(
            percentile, cell, values, ncells, nodata
        )
    raise ValueError(
        f"unknown aggregate {aggregate!r}; expected one of "
        "min, max, mean, count, stdev, idw, or a percentile like p95"
    )


def _reduce_simple(aggregate: str, cell: np.ndarray, values: np.ndarray, ncells: int, nodata: float) -> np.ndarray:
    """Reduce with one of the count-based aggregates (``count``/``mean``/``min``/``max``/``stdev``)."""
    count = np.bincount(cell, minlength=ncells)
    if aggregate == "count":
        return count.astype(np.float64)  # an empty cell honestly holds zero points, not nodata

    grid = np.full(ncells, float(nodata), dtype=np.float64)
    occupied = count > 0
    if aggregate in ("mean", "stdev"):
        total = np.bincount(cell, weights=values, minlength=ncells)
        mean = total[occupied] / count[occupied]
        if aggregate == "mean":
            grid[occupied] = mean
        else:
            sq = np.bincount(cell, weights=values * values, minlength=ncells)
            variance = np.clip(sq[occupied] / count[occupied] - mean * mean, 0.0, None)
            grid[occupied] = np.sqrt(variance)
    elif aggregate == "min":
        reduced = np.full(ncells, np.inf)
        np.minimum.at(reduced, cell, values)
        grid[occupied] = reduced[occupied]
    else:  # max
        reduced = np.full(ncells, -np.inf)
        np.maximum.at(reduced, cell, values)
        grid[occupied] = reduced[occupied]
    return grid


def _reduce_idw(cell: np.ndarray, values: np.ndarray, distance: np.ndarray, ncells: int, nodata: float) -> np.ndarray:
    """Reduce by inverse-distance weighting, each contribution weighted by ``1 / distance`` to the cell centre."""
    grid = np.full(ncells, float(nodata), dtype=np.float64)
    weight = 1.0 / np.maximum(distance, _IDW_EPS)
    weight_sum = np.bincount(cell, weights=weight, minlength=ncells)
    weighted = np.bincount(cell, weights=weight * values, minlength=ncells)
    occupied = weight_sum > 0
    grid[occupied] = weighted[occupied] / weight_sum[occupied]
    return grid


def _reduce_percentile(
    percentile: float, cell: np.ndarray, values: np.ndarray, ncells: int, nodata: float
) -> np.ndarray:
    """Reduce to a per-cell percentile, looping over occupied cells only."""
    grid = np.full(ncells, float(nodata), dtype=np.float64)
    if cell.size == 0:
        return grid
    order = np.argsort(cell, kind="stable")
    sorted_cells, sorted_values = cell[order], values[order]
    occupied, starts = np.unique(sorted_cells, return_index=True)
    bounds = np.append(starts, sorted_cells.size)
    for k, index in enumerate(occupied):
        grid[index] = np.percentile(sorted_values[bounds[k] : bounds[k + 1]], percentile)
    return grid
