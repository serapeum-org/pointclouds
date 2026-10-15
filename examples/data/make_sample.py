"""Regenerate the committed LAS/LAZ sample fixtures.

The two files next to this script - ``sample.las`` and ``sample.laz`` - are small, synthetic and deterministic:
400 points on a 20 x 20 grid in EPSG:28992 (Amersfoort / RD New), carrying intensity, ASPRS classification and
return number, with a sloping surface in z and a handful of points raised to stand in for vegetation.

They are committed so the test suite and the docs examples need no download and no vendor data. Run this script
only to change what they contain::

    pixi run -e dev sample-data

Keep it deterministic: the tests assert on exact counts and bounds.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

#: Grid size per axis; the fixture holds SIDE * SIDE points.
SIDE = 20
#: Spacing in metres between neighbouring points.
SPACING = 2.0
#: Lower-left corner in EPSG:28992, roughly central Netherlands.
ORIGIN = (155000.0, 463000.0)


def build_arrays() -> dict[str, np.ndarray]:
    """Build the fixture's point arrays.

    Returns:
        A mapping of dimension name to array: ``x``, ``y``, ``z``, ``intensity``, ``classification`` and
        ``return_number``, each of length ``SIDE * SIDE``.
    """
    rng = np.random.default_rng(20261015)
    cols, rows = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="xy")
    x = ORIGIN[0] + cols.ravel() * SPACING
    y = ORIGIN[1] + rows.ravel() * SPACING

    # A gentle plane in z, so gridding with min/max/mean gives visibly different surfaces.
    z = 10.0 + 0.05 * (x - ORIGIN[0]) + 0.02 * (y - ORIGIN[1])
    classification = np.full(x.size, 2, dtype=np.uint8)  # ASPRS 2 = ground

    # One in eight points becomes "vegetation" a few metres up, in ASPRS class 5 (high vegetation).
    canopy = rng.choice(x.size, size=x.size // 8, replace=False)
    z[canopy] += rng.uniform(3.0, 12.0, size=canopy.size)
    classification[canopy] = 5

    return {
        "x": x,
        "y": y,
        "z": z,
        "intensity": rng.integers(1, 2000, size=x.size).astype(np.uint16),
        "classification": classification,
        "return_number": np.where(classification == 5, 1, 2).astype(np.uint8),
    }


def write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write ``arrays`` to ``path`` as LAS 1.4, point format 3, in EPSG:28992.

    Args:
        path: destination file; the suffix decides LAS or LAZ.
        arrays: the mapping returned by :func:`build_arrays`.
    """
    header = laspy.LasHeader(version="1.4", point_format=3)
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([ORIGIN[0], ORIGIN[1], 0.0])
    header.add_crs(CRS.from_epsg(28992))

    las = laspy.LasData(header)
    las.x = arrays["x"]
    las.y = arrays["y"]
    las.z = arrays["z"]
    las.intensity = arrays["intensity"]
    las.classification = arrays["classification"]
    las.return_number = arrays["return_number"]
    las.write(str(path))


def main() -> None:
    """Write ``sample.las`` and ``sample.laz`` next to this script."""
    here = Path(__file__).parent
    arrays = build_arrays()
    for name in ("sample.las", "sample.laz"):
        target = here / name
        write(target, arrays)
        print(f"wrote {target} ({target.stat().st_size / 1024:.1f} KiB, {arrays['x'].size} points)")


if __name__ == "__main__":
    main()
