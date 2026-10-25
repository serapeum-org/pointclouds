"""Concatenate several clouds into one. (Phase 2 - T-9).

:func:`merge_arrays` assembles the combined coordinate and attribute arrays and settles the CRS, then
:meth:`PointCloud.merge` builds the cloud from them. Keeping the assembly here, separate from construction,
avoids importing :class:`PointCloud` at module load. Pure numpy - no GDAL, no pyramids.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyproj import CRS

    from pointclouds.points import PointCloud

__all__ = ["merge_arrays"]

#: The attribute dimensions that are carried through a merge, alongside x/y/z.
_ATTRIBUTES = ("intensity", "classification", "return_number")


def merge_arrays(clouds: Sequence[PointCloud]) -> dict[str, Any]:
    """Concatenate the clouds' arrays and resolve a single CRS.

    Args:
        clouds: the clouds to merge, in order; each is loaded if its records are not in memory yet.

    Returns:
        A mapping of ``PointCloud`` constructor keyword to value: ``x``/``y``/``z``, the ``crs``, and each
        attribute that every cloud carries. An attribute no cloud carries is omitted.

    Raises:
        ValueError: when ``clouds`` is empty, when the clouds disagree on CRS, or when an attribute is present
            on some clouds but absent on others (so the concatenation would misalign).

    Examples:
        >>> import numpy as np
        >>> from pointclouds import PointCloud
        >>> a = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))
        >>> b = PointCloud(x=np.ones(3), y=np.zeros(3), z=np.zeros(3))
        >>> arrays = merge_arrays([a, b])
        >>> arrays["x"].tolist()
        [0.0, 0.0, 1.0, 1.0, 1.0]
        >>> sorted(arrays)
        ['crs', 'x', 'y', 'z']
    """
    clouds = list(clouds)
    if not clouds:
        raise ValueError("merge needs at least one cloud")

    merged: dict[str, Any] = {
        "x": np.concatenate([c.x for c in clouds]),
        "y": np.concatenate([c.y for c in clouds]),
        "z": np.concatenate([c.z for c in clouds]),
        "crs": _resolve_crs(clouds),
    }

    for name in _ATTRIBUTES:
        arrays = [getattr(c, name) for c in clouds]
        present = [values.size > 0 for values in arrays]
        if all(present):
            merged[name] = np.concatenate(arrays)
        elif any(present):
            raise ValueError(
                f"cannot merge: {name!r} is present on some clouds but not others; "
                "give every cloud the dimension, or none"
            )
    return merged


def _resolve_crs(clouds: Sequence[PointCloud]) -> CRS | None:
    """Return the one CRS the clouds share, or raise when they disagree.

    ``None`` (no known SRS) is treated as a distinct value, so merging a cloud that has a CRS with one that does
    not is a conflict rather than a silent guess.
    """
    distinct: list[CRS | None] = []
    for crs in (c.crs for c in clouds):
        if not any(_crs_equal(crs, seen) for seen in distinct):
            distinct.append(crs)
    if len(distinct) > 1:
        shown = ", ".join("None" if c is None else c.to_string() for c in distinct)
        raise ValueError(f"cannot merge clouds with conflicting CRSs: {{{shown}}}")
    return distinct[0]


def _crs_equal(a: CRS | None, b: CRS | None) -> bool:
    """Whether two CRSs are the same, counting ``None`` as equal only to ``None``."""
    if a is None or b is None:
        return a is None and b is None
    return a == b
