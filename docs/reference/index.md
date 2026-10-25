# API Reference

::: pointclouds

## PointCloud

::: pointclouds.points

## LAS / LAZ engine

The laspy-backed reader behind `PointCloud.read_file`. Most callers never need it directly; it is documented
because the header/record split is part of the package's contract.

::: pointclouds.io.las

## COPC engine (windowed reads)

The spatially-indexed reader behind `PointCloud.read_window`. This is the path a large cloud should be read
through: a bounding box or a target resolution prunes the file's octree, so only the chunks that matter are
fetched and decompressed.

::: pointclouds.io.copc

## EPT engine (windowed reads)

The spatially-indexed reader for Entwine Point Tile datasets, behind `PointCloud.read_window` /
`PointCloud.read_file` when the source is an EPT directory or `ept.json`. Where COPC packs the octree into one
LAZ file, EPT spreads it across a directory of per-node files; the same bounding-box and resolution window prunes
which nodes are fetched. Local and remote (via the `[cloud]` extra); `laszip` and `binary` node encodings.

::: pointclouds.io.ept

## The pyramids boundary

::: pointclouds.convert
