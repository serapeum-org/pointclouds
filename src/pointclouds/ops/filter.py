"""Build the boolean masks that subset a cloud by attribute or extent. (Phase 2 - T-9).

This is the one mask-building helper the four public operations share. :func:`build_mask` turns the
``filter`` criteria into a boolean mask over the points; :func:`bbox_mask` is the extent test that both
:meth:`~pointclouds.points.PointCloud.filter` and :mod:`pointclouds.ops.clip` reuse. Nothing here constructs a
cloud or indexes its arrays - that is :meth:`PointCloud._take`'s job, so the alignment invariant lives in one
place. Everything imports with the core dependency set alone (numpy); no GDAL, no pyramids, no shapely.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pointclouds.points import PointCloud

__all__ = ["bbox_mask", "build_mask"]

#: The attribute dimensions :func:`build_mask` can filter on, each a membership test against the given value(s).
_ATTRIBUTE_CRITERIA = ("classification", "return_number", "intensity")


def bbox_mask(x: np.ndarray, y: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Return the mask of points inside an axis-aligned bounding box, edges included.

    Args:
        x: ``(N,)`` x-coordinates.
        y: ``(N,)`` y-coordinates.
        bbox: ``(xmin, ymin, xmax, ymax)``; a point on an edge counts as inside.

    Returns:
        ``(N,)`` boolean mask, ``True`` where the point falls within ``bbox``.

    Raises:
        ValueError: when ``bbox`` is not four numbers with ``xmin <= xmax`` and ``ymin <= ymax``.

    Examples:
        >>> import numpy as np
        >>> x = np.array([0.0, 5.0, 9.0])
        >>> y = np.array([0.0, 0.0, 0.0])
        >>> bbox_mask(x, y, (4.0, -1.0, 10.0, 1.0)).tolist()
        [False, True, True]
    """
    xmin, ymin, xmax, ymax = bbox
    if xmin > xmax or ymin > ymax:
        raise ValueError(f"bbox must be (xmin, ymin, xmax, ymax) with min <= max; got {bbox}")
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def build_mask(
    cloud: PointCloud,
    *,
    classification: int | Iterable[int] | None = None,
    return_number: int | Iterable[int] | None = None,
    intensity: int | Iterable[int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Combine the given criteria into one boolean mask, ANDing every criterion together.

    Each attribute criterion is a membership test: a scalar keeps points equal to it, an iterable keeps points
    whose value is in the set. ``bbox`` keeps points inside the extent. A criterion left ``None`` is ignored; with
    no criteria at all the mask keeps everything.

    Args:
        cloud: the cloud to test; its record arrays are loaded if they are not in memory yet.
        classification: an ASPRS class code, or an iterable of them, to keep.
        return_number: a return number, or an iterable of them, to keep.
        intensity: an intensity value, or an iterable of them, to keep.
        bbox: ``(xmin, ymin, xmax, ymax)`` extent to keep, edges included.

    Returns:
        ``(N,)`` boolean mask over the cloud's points.

    Raises:
        ValueError: when a criterion names an attribute the cloud does not carry (an empty array), or when
            ``bbox`` is malformed.

    Examples:
        >>> import numpy as np
        >>> from pointclouds import PointCloud
        >>> cloud = PointCloud(
        ...     x=np.arange(4.0), y=np.zeros(4), z=np.zeros(4),
        ...     classification=np.array([2, 5, 2, 6], dtype=np.uint8),
        ... )
        >>> build_mask(cloud, classification=2).tolist()
        [True, False, True, False]
        >>> build_mask(cloud, classification=[2, 6], bbox=(1.0, -1.0, 3.0, 1.0)).tolist()
        [False, False, True, True]
    """
    n = cloud.x.shape[0]  # also forces a lazy cloud to load, so every array below is populated
    mask = np.ones(n, dtype=bool)

    for name, criterion in zip(
        _ATTRIBUTE_CRITERIA, (classification, return_number, intensity), strict=True
    ):
        if criterion is None:
            continue
        values = getattr(cloud, name)
        if values.size == 0:
            raise ValueError(f"cannot filter on {name!r}: this cloud carries no {name} data")
        wanted = np.atleast_1d(np.asarray(criterion))
        mask &= np.isin(values, wanted)

    if bbox is not None:
        mask &= bbox_mask(cloud.x, cloud.y, bbox)

    return mask
