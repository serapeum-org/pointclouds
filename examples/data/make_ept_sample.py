"""Regenerate the committed EPT (Entwine Point Tile) fixtures.

Entwine is the tool that normally writes EPT, and it carries a GDAL/PDAL stack this package exists without. So,
as with the COPC fixture, this script assembles small EPT datasets by hand - directory, manifest, hierarchy and
per-node data files - so the windowed-read tests run against a *real* tiled octree rather than a mock.

Run it only to change the fixtures::

    pixi run -e dev ept-sample-data

It writes two datasets next to this script, differing only in how each node is encoded:

- ``sample-ept/``         - ``dataType: "laszip"``: every node is a standalone LAZ (read with laspy).
- ``sample-ept-binary/``  - ``dataType: "binary"``: every node is the schema's dimensions packed
  little-endian (read with numpy), with X/Y/Z stored as scaled int32.

Both hold the same 1600 points on a 40 x 40 grid in EPSG:28992 as the COPC fixture, over a two-level octree: a
root node subsampling the whole extent, then eight depth-1 octants holding the rest. That shape makes both query
paths meaningful - a bounding box skips octants, and ``depth=0`` / a coarse ``resolution`` reads the root alone.

An EPT dataset is four parts, which this script writes:

1. ``ept.json`` - the manifest: cubic ``bounds``, tight ``boundsConforming``, ``dataType``, ``hierarchyType``,
   ``points``, ``schema`` (the dimensions), ``span`` (the per-edge resolution), ``srs`` and ``version``.
2. ``ept-data/<d>-<x>-<y>-<z>.<ext>`` - one file per octree node.
3. ``ept-hierarchy/0-0-0-0.json`` - the index: ``"d-x-y-z"`` -> point count (``-1`` would defer to a nested
   page; this fixture is shallow enough to need only the root page).
4. The directory itself, named for the dataset.

Keep it deterministic: the tests assert on exact counts and bounds.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import laspy
import numpy as np
from make_sample import EPSG, ORIGIN, build_arrays
from pyproj import CRS

#: Points per axis - denser than the LAS/LAZ samples so the octree has something to split.
SIDE = 40
#: Every n-th point goes in the root node; the rest drop to depth 1.
ROOT_STRIDE = 16
#: Points-per-edge resolution recorded in the manifest's ``span``.
SPAN = 128
#: Scale for the binary dataset's int32 X/Y/Z (centimetre precision, as LAS uses).
BIN_SCALE = 0.01


def octree() -> tuple[np.ndarray, float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the point arrays and assign every point to an octree node.

    Returns:
        ``(cube_min, cube_side, nodes, arrays)`` - the cubic octree origin, its side length, a mapping of
        ``"d-x-y-z"`` key to the point indices in that node, and the full point arrays.
    """
    arrays = build_arrays(side=SIDE)
    xyz = np.column_stack([arrays["x"], arrays["y"], arrays["z"]])
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    side = float(np.max(maxs - mins))
    cube_min = mins  # the cube's low corner; its side is the longest data axis

    total = arrays["x"].size
    indices = np.arange(total)
    root = indices[::ROOT_STRIDE]
    rest = np.setdiff1d(indices, root)
    # depth-1 octant per remaining point: which half of the cube it falls in, per axis.
    octant = np.clip(((xyz[rest] - cube_min) / (side / 2.0)).astype(int), 0, 1)

    nodes: dict[str, np.ndarray] = {"0-0-0-0": root}
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                sel = rest[(octant[:, 0] == x) & (octant[:, 1] == y) & (octant[:, 2] == z)]
                if sel.size:
                    nodes[f"1-{x}-{y}-{z}"] = sel
    return cube_min, side, nodes, arrays


def write_laszip_node(path: Path, arrays: dict[str, np.ndarray], sel: np.ndarray) -> None:
    """Write one node's points as a standalone LAZ (``dataType: "laszip"``).

    Args:
        path: the ``ept-data/<key>.laz`` destination.
        arrays: the full point arrays.
        sel: the indices of the points in this node.
    """
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = np.array([BIN_SCALE, BIN_SCALE, BIN_SCALE])
    header.offsets = np.array([ORIGIN[0], ORIGIN[1], 0.0])
    header.add_crs(CRS.from_epsg(EPSG))
    las = laspy.LasData(header)
    las.x = arrays["x"][sel]
    las.y = arrays["y"][sel]
    las.z = arrays["z"][sel]
    las.intensity = arrays["intensity"][sel]
    las.classification = arrays["classification"][sel]
    las.return_number = arrays["return_number"][sel]
    las.write(str(path))


def binary_schema() -> list[dict[str, object]]:
    """The EPT schema for the binary dataset: X/Y/Z as scaled int32, the attributes as their native ints.

    Returns:
        The ``schema`` list for ``ept.json``.
    """
    return [
        {"name": "X", "type": "signed", "size": 4, "scale": BIN_SCALE, "offset": ORIGIN[0]},
        {"name": "Y", "type": "signed", "size": 4, "scale": BIN_SCALE, "offset": ORIGIN[1]},
        {"name": "Z", "type": "signed", "size": 4, "scale": BIN_SCALE, "offset": 0.0},
        {"name": "Intensity", "type": "unsigned", "size": 2},
        {"name": "ReturnNumber", "type": "unsigned", "size": 1},
        {"name": "Classification", "type": "unsigned", "size": 1},
    ]


def write_binary_node(path: Path, arrays: dict[str, np.ndarray], sel: np.ndarray) -> None:
    """Write one node's points packed little-endian per :func:`binary_schema` (``dataType: "binary"``).

    Args:
        path: the ``ept-data/<key>.bin`` destination.
        arrays: the full point arrays.
        sel: the indices of the points in this node.
    """
    xi = np.round((arrays["x"][sel] - ORIGIN[0]) / BIN_SCALE).astype("<i4")
    yi = np.round((arrays["y"][sel] - ORIGIN[1]) / BIN_SCALE).astype("<i4")
    zi = np.round((arrays["z"][sel] - 0.0) / BIN_SCALE).astype("<i4")
    dtype = np.dtype(
        [
            ("X", "<i4"),
            ("Y", "<i4"),
            ("Z", "<i4"),
            ("Intensity", "<u2"),
            ("ReturnNumber", "u1"),
            ("Classification", "u1"),
        ]
    )
    records = np.empty(sel.size, dtype=dtype)
    records["X"], records["Y"], records["Z"] = xi, yi, zi
    records["Intensity"] = arrays["intensity"][sel]
    records["ReturnNumber"] = arrays["return_number"][sel]
    records["Classification"] = arrays["classification"][sel]
    path.write_bytes(records.tobytes())


def manifest(data_type: str, cube_min: np.ndarray, side: float, arrays: dict[str, np.ndarray], count: int) -> dict:
    """Build the ``ept.json`` manifest for a dataset.

    Args:
        data_type: ``"laszip"`` or ``"binary"``.
        cube_min: the octree cube's low corner.
        side: the octree cube's side length.
        arrays: the full point arrays (for the conforming bounds).
        count: the total point count.

    Returns:
        The manifest as a dict ready to serialise.
    """
    cube_max = cube_min + side
    xyz_min = [float(arrays["x"].min()), float(arrays["y"].min()), float(arrays["z"].min())]
    xyz_max = [float(arrays["x"].max()), float(arrays["y"].max()), float(arrays["z"].max())]
    schema = binary_schema() if data_type == "binary" else [{"name": d["name"]} for d in binary_schema()]
    return {
        "bounds": [float(cube_min[0]), float(cube_min[1]), float(cube_min[2]),
                   float(cube_max[0]), float(cube_max[1]), float(cube_max[2])],
        "boundsConforming": [xyz_min[0], xyz_min[1], xyz_min[2], xyz_max[0], xyz_max[1], xyz_max[2]],
        "dataType": data_type,
        "hierarchyType": "json",
        "points": count,
        "schema": schema,
        "span": SPAN,
        "srs": {"authority": "EPSG", "horizontal": str(EPSG)},
        "version": "1.0.0",
    }


def write_dataset(root: Path, data_type: str) -> int:
    """Assemble one EPT dataset under ``root``.

    Args:
        root: the dataset directory to create.
        data_type: ``"laszip"`` or ``"binary"``.

    Returns:
        The number of points written.
    """
    cube_min, side, nodes, arrays = octree()
    (root / "ept-data").mkdir(parents=True, exist_ok=True)
    (root / "ept-hierarchy").mkdir(parents=True, exist_ok=True)

    extension = "laz" if data_type == "laszip" else "bin"
    hierarchy = {}
    for key, sel in nodes.items():
        target = root / "ept-data" / f"{key}.{extension}"
        if data_type == "laszip":
            write_laszip_node(target, arrays, sel)
        else:
            write_binary_node(target, arrays, sel)
        hierarchy[key] = int(sel.size)

    count = int(arrays["x"].size)
    # Trailing newline so pre-commit's end-of-file-fixer leaves the committed fixtures untouched.
    (root / "ept.json").write_text(json.dumps(manifest(data_type, cube_min, side, arrays, count), indent=2) + "\n")
    (root / "ept-hierarchy" / "0-0-0-0.json").write_text(json.dumps(hierarchy, indent=2) + "\n")
    return count


def main() -> None:
    """Write both EPT fixtures next to this script."""
    here = Path(__file__).parent
    for name, data_type in (("sample-ept", "laszip"), ("sample-ept-binary", "binary")):
        count = write_dataset(here / name, data_type)
        print(f"wrote {here / name} ({data_type}, {count} points)")


if __name__ == "__main__":
    main()
