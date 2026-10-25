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

## The pyramids boundary

::: pointclouds.convert
