"""Regenerate the committed COPC fixture, ``sample.copc.laz``.

``laspy`` reads COPC but does not write it, and PDAL - the usual way to make one - would drag a GDAL stack into
a repository whose whole point is not having one. So this script assembles the file itself. It is small, exact
and self-contained, and it exists so the windowed-read tests run against a *real* octree rather than a mock:
a mocked ``CopcReader`` would prove only that the calls are wired up, not that a bounding box actually prunes
chunks.

Run it only to change what the fixture contains::

    pixi run -e dev copc-sample-data

What a COPC file is, in the four pieces this script writes:

1. A LAS 1.4 header whose **first** VLR is the 160-byte ``copc`` info VLR (octree centre, half-size, root
   spacing, and where the hierarchy lives). A reader that finds it anywhere but first rejects the file.
2. Point data where every octree node is its own LAZ chunk, so a node can be fetched and decompressed on its
   own. ``lazrs`` gives one chunk per ``finish_current_chunk()`` call.
3. A hierarchy EVLR (``copc``/1000): one 32-byte entry per node - voxel key, absolute byte offset, byte size,
   point count.
4. The LASzip VLR describing the chunk layout, which is what marks the point data as compressed.

The octree is deliberately shallow - a root holding every ``ROOT_STRIDE``-th point, then one level of eight
octants holding the rest - because two levels are all it takes to make both query paths meaningful: a bounding
box skips whole octants, and ``resolution=`` stops at the root.

Keep it deterministic: the tests assert on exact counts and bounds, so the creation date is pinned rather than
taken from the clock.
"""

from __future__ import annotations

import io
import struct
from datetime import date
from pathlib import Path

import laspy

# lazrs is used directly here to compress the per-node chunks; it is not a declared dependency of the package
# but arrives with laspy[lazrs]. LasZipVlr and VLRList are laspy internals with no stability guarantee - this
# script is the one place that reaches for them, and the generator test is what catches a laspy move that breaks
# them. This module is run as a script (`python examples/data/make_copc_sample.py`), which puts its own
# directory on sys.path so the sibling `make_sample` import below resolves; running it as `python -m` would not.
import lazrs
import numpy as np
from laspy.vlrs.known import LasZipVlr
from make_sample import EPSG, ORIGIN, build_arrays
from pyproj import CRS

#: Points per axis for the COPC fixture - denser than the LAS/LAZ samples so the octree has something to split.
SIDE = 40
#: Every n-th point goes in the root node; the rest are pushed down to level 1.
ROOT_STRIDE = 16
#: COPC requires point format 6, 7 or 8 (the three LAS 1.4 formats COPC permits; LAS 1.4 itself defines 6-10).
#: 6 is the smallest that carries what we need.
POINT_FORMAT = 6
#: Pinned so re-running the script produces a byte-identical file.
CREATION_DATE = date(2026, 10, 25)
#: Size of an EVLR header, which sits between ``start_of_first_evlr`` and the hierarchy entries themselves.
EVLR_HEADER_SIZE = 60
#: A ``copc`` info VLR is exactly this long: 7 doubles, 2 uint64s (72 bytes), then 88 reserved bytes.
COPC_INFO_SIZE = 160


def build_las(arrays: dict[str, np.ndarray]) -> laspy.LasData:
    """Put ``arrays`` into an in-memory LAS 1.4 / point-format-6 cloud.

    Args:
        arrays: the mapping returned by :func:`make_sample.build_arrays`.

    Returns:
        A :class:`laspy.LasData` whose packed record array this script compresses chunk by chunk.
    """
    header = laspy.LasHeader(version="1.4", point_format=POINT_FORMAT)
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([ORIGIN[0], ORIGIN[1], 0.0])
    header.creation_date = CREATION_DATE
    header.add_crs(CRS.from_epsg(EPSG))

    las = laspy.LasData(header)
    las.x = arrays["x"]
    las.y = arrays["y"]
    las.z = arrays["z"]
    las.intensity = arrays["intensity"]
    las.classification = arrays["classification"]
    las.return_number = arrays["return_number"]
    las.update_header()  # point count, per-return counts, and the data's real mins/maxs
    return las


def root_cube(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Return the octree root's centre and half-size: the smallest cube containing every point.

    Args:
        arrays: the point arrays.

    Returns:
        ``(centre, halfsize)``. Every node in an octree is a cube, so the root has to be one too - the longest
        data axis sets the side, and the shorter axes sit inside it with room to spare.
    """
    mins = np.array([arrays["x"].min(), arrays["y"].min(), arrays["z"].min()])
    maxs = np.array([arrays["x"].max(), arrays["y"].max(), arrays["z"].max()])
    halfsize = float(np.max(maxs - mins)) / 2.0
    return (mins + maxs) / 2.0, halfsize


def assign_nodes(
    arrays: dict[str, np.ndarray], centre: np.ndarray, halfsize: float
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Split the points over a two-level octree.

    The root takes a regular subsample - every :data:`ROOT_STRIDE`-th point - which is what makes a coarse read
    cheap: ask for the root alone and you get an evenly thinned cloud over the whole extent. Everything else
    drops into the level-1 octant that contains it.

    Args:
        arrays: the point arrays.
        centre: the root cube's centre, from :func:`root_cube`.
        halfsize: the root cube's half-size, from :func:`root_cube`.

    Returns:
        One ``(voxel_key, point_indices)`` pair per non-empty node, root first. A voxel key is
        ``(level, x, y, z)``, as the COPC specification writes it.
    """
    total = arrays["x"].size
    indices = np.arange(total)
    root = indices[::ROOT_STRIDE]
    rest = np.setdiff1d(indices, root)

    xyz = np.column_stack([arrays["x"], arrays["y"], arrays["z"]])
    # A level-1 voxel has side `halfsize`, so the octant index per axis is the half of the root cube a point
    # falls in. Clipped because a point exactly on the far face would index 2.
    octant = np.clip(((xyz[rest] - (centre - halfsize)) / halfsize).astype(int), 0, 1)

    nodes: list[tuple[tuple[int, int, int, int], np.ndarray]] = [((0, 0, 0, 0), root)]
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                selected = rest[(octant[:, 0] == x) & (octant[:, 1] == y) & (octant[:, 2] == z)]
                if selected.size:
                    nodes.append(((1, x, y, z), selected))
    return nodes


def compress_nodes(
    record: np.ndarray, nodes: list[tuple[tuple[int, int, int, int], np.ndarray]], laz_vlr: "lazrs.LazVlr"
) -> tuple[bytes, list[tuple[int, int]]]:
    """Compress each node's points into its own LAZ chunk, in one contiguous block.

    Args:
        record: the packed point record array, one row per point.
        nodes: the node list from :func:`assign_nodes`.
        laz_vlr: the chunk-layout descriptor. The *same* object must go into the header's LASzip VLR, or the
            header would describe a different layout than the one the points were compressed with - built once
            in :func:`write` and passed to both.

    Returns:
        ``(point_data, spans)`` - the bytes that go at the file's point-data offset, and one
        ``(start, size)`` pair per node, as offsets *within* that block.
    """
    destination = io.BytesIO()
    compressor = lazrs.LasZipCompressor(destination, laz_vlr)
    compressor.reserve_offset_to_chunk_table()

    spans: list[tuple[int, int]] = []
    for _key, selected in nodes:
        start = destination.tell()
        compressor.compress_many(record[selected].tobytes())
        compressor.finish_current_chunk()  # one chunk per node is what makes a node independently readable
        spans.append((start, destination.tell() - start))
    compressor.done()
    return destination.getvalue(), spans


def hierarchy_bytes(
    nodes: list[tuple[tuple[int, int, int, int], np.ndarray]],
    spans: list[tuple[int, int]],
    offset_to_point_data: int,
) -> bytes:
    """Pack the octree hierarchy as COPC entries.

    Args:
        nodes: the node list from :func:`assign_nodes`.
        spans: the per-node ``(start, size)`` pairs from :func:`compress_nodes`.
        offset_to_point_data: the file offset the point-data block starts at, which turns each span's relative
            start into the absolute offset an entry has to carry.

    Returns:
        The concatenated 32-byte entries: voxel key (4 int32), chunk offset (uint64), chunk size and point
        count (2 int32).
    """
    entries = b""
    for (key, selected), (start, size) in zip(nodes, spans, strict=True):
        entries += struct.pack("<iiii", *key)
        entries += struct.pack("<Qii", offset_to_point_data + start, size, len(selected))
    return entries


def copc_info_bytes(centre: np.ndarray, halfsize: float, spacing: float, root_offset: int, root_size: int) -> bytes:
    """Pack the ``copc`` info VLR that has to be a COPC file's first VLR.

    Args:
        centre: the root cube's centre.
        halfsize: the root cube's half-size.
        spacing: the root node's point spacing. ``resolution=`` on a read is measured against this, and the
            octree halves it per level.
        root_offset: file offset of the root hierarchy page's record data.
        root_size: length of that page in bytes.

    Returns:
        Exactly :data:`COPC_INFO_SIZE` bytes.
    """
    payload = (
        struct.pack("<3d", *centre)
        + struct.pack("<d", halfsize)
        + struct.pack("<d", spacing)
        + struct.pack("<Q", root_offset)
        + struct.pack("<Q", root_size)
        + struct.pack("<2d", 0.0, 0.0)  # gps time min/max: unused, the fixture carries no GPS time
    )
    return payload.ljust(COPC_INFO_SIZE, b"\0")


def write(path: Path) -> int:
    """Assemble the fixture and write it to ``path``.

    Args:
        path: destination file, conventionally named ``*.copc.laz``.

    Returns:
        How many points were written.
    """
    arrays = build_arrays(side=SIDE)
    las = build_las(arrays)
    centre, halfsize = root_cube(arrays)
    nodes = assign_nodes(arrays, centre, halfsize)
    # Build the LAZ chunk-layout descriptor once (True = variable-size chunks, as COPC requires) and use the
    # same object for the point compression and the header's LASzip VLR, so the two cannot disagree.
    laz_vlr = lazrs.LazVlr.new_for_compression(POINT_FORMAT, 0, True)
    point_data, spans = compress_nodes(las.points.array, nodes, laz_vlr)

    # The root holds `len(root)` points spread over the full extent, so this is their mean spacing - the honest
    # number for the info VLR, and the one `resolution=` is compared against.
    root_count = len(nodes[0][1])
    spacing = (halfsize * 2) / root_count**0.5

    header = las.header
    header.number_of_evlrs = 1

    # The COPC info VLR goes first, ahead of the WKT VLR `add_crs` left in the list.
    info_vlr = laspy.VLR("copc", 1, "COPC info", b"\0" * COPC_INFO_SIZE)
    header.vlrs.insert(0, info_vlr)
    header.vlrs.append(LasZipVlr(laz_vlr.record_data()))

    # Two passes: the first learns `offset_to_point_data`, which the hierarchy's absolute offsets need, and the
    # hierarchy's own position in turn goes back into the info VLR. Both VLRs keep their size, so the second
    # write lands at the same offset - `ensure_same_size` makes that an assertion rather than a hope.
    header.write_to(io.BytesIO())
    offset_to_point_data = header.offset_to_point_data
    entries = hierarchy_bytes(nodes, spans, offset_to_point_data)

    start_of_first_evlr = offset_to_point_data + len(point_data)
    header.start_of_first_evlr = start_of_first_evlr
    info_vlr.record_data = copc_info_bytes(
        centre, halfsize, spacing, start_of_first_evlr + EVLR_HEADER_SIZE, len(entries)
    )

    out = io.BytesIO()
    header.write_to(out, ensure_same_size=True)
    out.write(point_data)
    evlrs = laspy.vlrs.vlrlist.VLRList([laspy.VLR("copc", 1000, "COPC hierarchy", entries)])
    evlrs.write_to(out, as_extended=True)

    path.write_bytes(out.getvalue())
    return int(header.point_count)


def main() -> None:
    """Write ``sample.copc.laz`` next to this script."""
    target = Path(__file__).parent / "sample.copc.laz"
    count = write(target)
    print(f"wrote {target} ({target.stat().st_size / 1024:.1f} KiB, {count} points)")


if __name__ == "__main__":
    main()
