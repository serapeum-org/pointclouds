"""Choose which points survive a decimation: random thinning and voxel thinning. (Phase 2 - T-9).

Both entry points return an integer index array into the cloud's points; :meth:`PointCloud._take` does the
actual subsetting, so neither function constructs a cloud. Random thinning is reproducible under a seed; voxel
thinning keeps one point per cube and is deterministic. Pure numpy - no GDAL, no pyramids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pointclouds.points import PointCloud

__all__ = ["select_indices"]


def select_indices(
    cloud: PointCloud,
    *,
    target: int | None = None,
    fraction: float | None = None,
    method: str = "random",
    voxel_size: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return the indices of the points to keep under the requested decimation.

    Args:
        cloud: the cloud to thin; its records are loaded if they are not in memory yet.
        target: for ``method="random"``, the number of points to keep. Never more than the cloud holds.
        fraction: for ``method="random"``, the share of points to keep, in ``(0, 1]``. A positive fraction over a
            non-empty cloud keeps at least one point (it never rounds down to zero).
        method: ``"random"`` for a uniform sample without replacement, ``"voxel"`` for one point per cube.
        voxel_size: for ``method="voxel"``, the cube edge length in CRS units.
        seed: seeds the random sample so a run is reproducible; ignored by ``"voxel"``.

    Returns:
        A sorted ``(k,)`` int array of indices into the cloud's points, so ``_take`` preserves point order.

    Raises:
        ValueError: when the arguments for the chosen method are missing, ambiguous, or out of range, or when
            ``method`` is neither ``"random"`` nor ``"voxel"``.

    Examples:
        >>> import numpy as np
        >>> from pointclouds import PointCloud
        >>> cloud = PointCloud(x=np.arange(10.0), y=np.zeros(10), z=np.zeros(10))
        >>> len(select_indices(cloud, target=3, seed=0))
        3
        >>> line = PointCloud(x=np.array([0.0, 0.5, 1.0, 1.5]), y=np.zeros(4), z=np.zeros(4))
        >>> select_indices(line, method="voxel", voxel_size=1.0).tolist()
        [0, 2]
    """
    n = cloud.x.shape[0]  # forces a lazy cloud to load

    if method == "random":
        return _random_indices(n, target=target, fraction=fraction, seed=seed)
    if method == "voxel":
        return _voxel_indices(cloud, voxel_size=voxel_size)
    raise ValueError(f"unknown decimation method {method!r}; expected 'random' or 'voxel'")


def _random_indices(n: int, *, target: int | None, fraction: float | None, seed: int | None) -> np.ndarray:
    """Return a sorted uniform sample of ``n`` indices, sized by ``target`` or ``fraction``."""
    if (target is None) == (fraction is None):
        raise ValueError("random decimation needs exactly one of target or fraction")
    if target is not None:
        if target < 0:
            raise ValueError(f"target must be non-negative; got {target}")
        keep = min(int(target), n)
    else:
        assert fraction is not None  # noqa: S101 - guaranteed by the exactly-one check above
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        # A positive fraction over a non-empty cloud keeps at least one point: rounding a small share down to
        # zero would turn "keep some" into "keep none", which no caller asking for a fraction means.
        keep = min(n, max(1, int(round(fraction * n)))) if n else 0

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=keep, replace=False)
    chosen.sort()
    return chosen


def _voxel_indices(cloud: PointCloud, *, voxel_size: float | None) -> np.ndarray:
    """Return the index of the first point in each occupied ``voxel_size`` cube, in point order."""
    if voxel_size is None or voxel_size <= 0:
        raise ValueError(f"voxel decimation needs a positive voxel_size; got {voxel_size}")

    x, y, z = cloud.x, cloud.y, cloud.z
    if x.size == 0:
        return np.empty(0, dtype=np.intp)

    keys = np.stack(
        [
            np.floor((x - x.min()) / voxel_size).astype(np.int64),
            np.floor((y - y.min()) / voxel_size).astype(np.int64),
            np.floor((z - z.min()) / voxel_size).astype(np.int64),
        ],
        axis=1,
    )
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return first
