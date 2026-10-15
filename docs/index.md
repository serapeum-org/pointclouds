# pointclouds

Read, filter, decimate and grid point clouds (LAS / LAZ / COPC) in Python, with an optional bridge to the
`pyramids` GIS engine.

See [Installation](installation.md) to get started.

## Quick start

Reading is **header-first**: opening a file reads its header and nothing else, so metadata costs the same whether
the file holds four hundred points or four billion. The record arrays decompress the moment you touch one.

```python
import pointclouds as pc

cloud = pc.PointCloud.read_file("examples/data/sample.laz")

cloud.crs         # CRS.from_epsg(28992) - from the header
cloud.bounds      # (155000.0, 463000.0, 155038.0, 463038.0)
cloud.count       # 400, straight from the header: no record was read
cloud.dimensions  # ('X', 'Y', 'Z', 'intensity', 'return_number', ...)

xyz = cloud.to_array(("X", "Y", "Z"))   # (400, 3) - this is what loads the records
```

Two things worth knowing:

- `cloud.count` is the header's point count and is free. `len(cloud)` is how many points are *in memory*, so on a
  freshly opened cloud it triggers the read. They agree once the records are loaded.
- Pass `lazy=False` to `read_file` to read the records up front, and `cloud.loaded` tells you which state you are
  in without changing it.

Everything above runs on the bare install — numpy, laspy, pyproj and scipy, with no GDAL anywhere. The
`pyramids` converters are a separate, optional step behind the `[gis]` extra.

## Sample data

`examples/data/sample.las` and `sample.laz` are small synthetic fixtures (400 points on a 20 × 20 grid in
EPSG:28992, with ground and vegetation classes). They are committed, so nothing needs downloading;
`examples/data/make_sample.py` regenerates them.
