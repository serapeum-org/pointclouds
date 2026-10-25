"""Clip a cloud to an extent or a polygon. (Phase 2 - T-9).

ADR - clip stays in the pure-numpy core (the §9 open question, settled here).
    The core's selling point is a pure-PyPI wheel that installs anywhere numpy runs, so adding shapely to
    ``[project.dependencies]`` was rejected. Both clip forms are therefore pure numpy:

    - ``bbox`` is a vectorised coordinate comparison (:func:`pointclouds.ops.filter.bbox_mask`).
    - ``polygon`` is a vectorised even-odd ray-casting point-in-polygon test over a single ring.

    So the core dependency set is unchanged. Out of scope for the core, and left to the ``[gis]`` vector path
    (pyramids ``FeatureCollection``): polygons with holes, multipart polygons, and shapely/OGR geometry objects.

:func:`clip_mask` returns a boolean mask; :meth:`PointCloud._take` does the subsetting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pointclouds.ops.filter import bbox_mask

if TYPE_CHECKING:
    from pointclouds.points import PointCloud

__all__ = ["clip_mask"]


def clip_mask(
    cloud: PointCloud,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    polygon: np.ndarray | None = None,
) -> np.ndarray:
    """Return the mask of points kept by a bbox or polygon clip.

    Args:
        cloud: the cloud to clip; its records are loaded if they are not in memory yet.
        bbox: ``(xmin, ymin, xmax, ymax)`` extent, edges included.
        polygon: an ``(M, 2)`` array of ``(x, y)`` vertices forming a single ring (closed or open).

    Returns:
        ``(N,)`` boolean mask over the cloud's points.

    Raises:
        ValueError: when neither or both of ``bbox`` and ``polygon`` are given, or when ``polygon`` is not an
            ``(M, 2)`` array of at least three vertices.

    Examples:
        >>> import numpy as np
        >>> from pointclouds import PointCloud
        >>> cloud = PointCloud(x=np.array([0.0, 2.0]), y=np.array([0.0, 2.0]), z=np.zeros(2))
        >>> clip_mask(cloud, bbox=(1.0, 1.0, 3.0, 3.0)).tolist()
        [False, True]
        >>> ring = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])
        >>> clip_mask(cloud, polygon=ring).tolist()
        [False, True]
    """
    if (bbox is None) == (polygon is None):
        raise ValueError("clip needs exactly one of bbox or polygon")
    if bbox is not None:
        return bbox_mask(cloud.x, cloud.y, bbox)
    return _points_in_polygon(cloud.x, cloud.y, np.asarray(polygon, dtype=np.float64))


def _points_in_polygon(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Return the mask of points inside a single polygon ring by even-odd ray casting.

    The test is vectorised over all points and loops only over the ring's edges, so its cost is
    ``O(points * edges)``. A point on the boundary is reported inconsistently, as the even-odd rule always does;
    that is acceptable for clipping a dense cloud.

    Args:
        x: ``(N,)`` x-coordinates.
        y: ``(N,)`` y-coordinates.
        ring: ``(M, 2)`` vertices; the ring is treated as closed, so the first vertex need not be repeated.

    Returns:
        ``(N,)`` boolean mask, ``True`` for points inside the ring.

    Raises:
        ValueError: when ``ring`` is not an ``(M, 2)`` array of at least three vertices.
    """
    if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
        raise ValueError(f"polygon must be an (M, 2) array of at least 3 vertices; got shape {ring.shape}")

    px, py = ring[:, 0], ring[:, 1]
    inside = np.zeros(x.shape[0], dtype=bool)
    # Walk each edge (j -> i); a horizontal edge never flips the ray parity, and the (yi > y) != (yj > y)
    # guard makes it harmless, so the divide/invalid warnings it would raise are suppressed rather than hit.
    with np.errstate(divide="ignore", invalid="ignore"):
        j = len(px) - 1
        for i in range(len(px)):
            crosses = (py[i] > y) != (py[j] > y)
            x_at_edge = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
            inside ^= crosses & (x < x_at_edge)
            j = i
    return inside
