# pointclouds

Read, filter, decimate and grid **point clouds** (LAS / LAZ, and COPC/EPT) in Python.

A *point cloud* is a set of 3-D measured points - typically from lidar or photogrammetry - each carrying `x`, `y`,
`z` plus per-point attributes such as intensity, return number and ASPRS classification. `pointclouds` reads the
standard exchange formats, exposes the header without touching the records, and provides the operations a cloud
needs before analysis or rendering: classification filtering, decimation, reprojection, and gridding to a raster
surface (DSM / DTM / density).

## Install

```bash
pip install pointclouds          # core: reads LAS/LAZ, grids to arrays, no GDAL required
pip install pointclouds[gis]     # + pyramids-gis: to_dataset() / to_feature_collection()
pip install pointclouds[cloud]   # + fsspec/s3fs: remote LAZ / COPC
pip install pointclouds[all]     # everything
```

The core depends only on `numpy`, `laspy`, `pyproj` and `scipy` - no GDAL - so it installs anywhere numpy runs.
GIS interoperability (turning a cloud into a `pyramids` `Dataset` or `FeatureCollection`) is an optional extra.

## Quick look

```python
import pointclouds as pc

cloud = pc.PointCloud.read_file("survey.laz")
cloud.crs, cloud.bounds, cloud.count          # from the header, no record read
xyz = cloud.to_array(("X", "Y", "Z"))         # (N, 3) float64

# COPC: read a window instead of the whole cloud - only the chunks that overlap are decompressed
box = pc.PointCloud.read_window("survey.copc.laz", bbox=(155000, 463000, 156000, 464000))
coarse = pc.PointCloud.read_window("survey.copc.laz", resolution=5.0)   # ~one point per 5 m

# EPT (Entwine): same windowed read, by directory or URL (remote needs the [cloud] extra)
box = pc.PointCloud.read_window("s3://bucket/survey-ept", bbox=(155000, 463000, 156000, 464000))

ground = cloud.filter(classification=2)
thin = ground.decimate(target=2_000_000, method="random")

arr, transform, crs = ground.to_grid(cell_size=1.0, aggregate="min")   # pure NumPy
dtm = ground.to_dataset(cell_size=1.0, aggregate="min")                # needs [gis]
```

## Where it sits

`pointclouds` is an independent package in the serapeum-org stack. It interoperates with
[`pyramids`](https://github.com/serapeum-org/pyramids) (the raster/vector GIS engine) through two converter methods
behind the `[gis]` extra, but does not depend on it for reading, gridding or classification. See
`planning/placement-decision.md` for the full architecture rationale.

## License

GPLv3.
