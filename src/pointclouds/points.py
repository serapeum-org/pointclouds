"""The :class:`PointCloud` container.

A point cloud is held as parallel 1-D arrays - index ``i`` selects the i-th point across every field - so a filter
or decimation stays aligned across coordinates and attributes. This module is the nucleus of the package and must
import with the core dependency set alone (numpy); it must never import GDAL or pyramids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PointCloud"]


@dataclass
class PointCloud:
    """In-memory point cloud: parallel per-dimension arrays plus a CRS.

    Attributes:
        x: ``(N,)`` float64 x-coordinates.
        y: ``(N,)`` float64 y-coordinates.
        z: ``(N,)`` float64 z-coordinates.
        intensity: ``(N,)`` uint16 return intensity, or empty.
        classification: ``(N,)`` uint8 ASPRS class codes, or empty.
        return_number: ``(N,)`` uint8 return-number-within-pulse, or empty.
        crs: a :class:`pyproj.CRS` (or ``None`` when the header carried no SRS).

    Note:
        ``read_file`` and the operations (filter/decimate/grid) land in Phase 1 (plan tasks T-8/T-9/T-11). This
        skeleton defines the field layout and the alignment invariant only.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint16))
    classification: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    return_number: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    crs: object | None = None

    def __post_init__(self) -> None:
        """Reject a cloud whose coordinate arrays disagree in length.

        Raises:
            ValueError: if ``x``, ``y`` and ``z`` do not all share one length.
        """
        n = len(self.x)
        if not (len(self.y) == n and len(self.z) == n):
            raise ValueError(f"x/y/z must share a length; got {len(self.x)}, {len(self.y)}, {len(self.z)}")

    def __len__(self) -> int:
        """Return the number of points in the cloud."""
        return int(self.x.shape[0])

    def to_array(self, dimensions: tuple[str, ...] = ("X", "Y", "Z")) -> np.ndarray:
        """Stack the requested dimensions column-wise into an ``(N, k)`` array.

        Args:
            dimensions: dimension names, case-insensitive (``"X"``/``"Y"``/``"Z"`` map to the coordinate arrays).

        Returns:
            ``(N, k)`` float64 array, one column per requested dimension.
        """
        lookup = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "intensity": self.intensity,
            "classification": self.classification,
            "return_number": self.return_number,
        }
        cols = [np.asarray(lookup[name.lower()], dtype=float) for name in dimensions]
        return np.column_stack(cols)
