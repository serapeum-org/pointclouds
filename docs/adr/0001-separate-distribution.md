# ADR 0001 - point clouds live in their own distribution

- Status: Accepted
- Date: 2026-10-15

## Context

Point-cloud support (LAS/LAZ/COPC) was evaluated for `pyramids` (the GIS engine) as an in-repo module behind a
`[lidar]` extra. The surface is expected to grow well past a format reader - classification, segmentation,
meshing, level-of-detail, and a PDAL-class pipeline.

## Decision

Ship point clouds as an independent distribution (`pointclouds`) that owns the data model and the engine, with
`pyramids-gis` as an **optional** `[gis]` extra reached only by the two converter methods. The core depends only
on numpy/laspy/pyproj/scipy and is GDAL-free.

## Consequences

- The package installs anywhere numpy runs; it does not inherit pyramids' platform matrix for its core.
- `pyramids` gains two generic primitives independently (see serapeum-org/pyramids#1135).
- `digitalrivers.lidar` migrates here (see serapeum-org/digital-rivers#84, #85).

Full rationale: `planning/placement-decision.md`.
