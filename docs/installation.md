# Installation

```bash
pip install pointclouds          # core - no GDAL required
pip install pointclouds[gis]     # + pyramids-gis converters
pip install pointclouds[cloud]   # + remote LAZ / COPC
pip install pointclouds[all]
```

The core installs anywhere numpy runs. The `[gis]` extra pulls `pyramids-gis` (and its vendored GDAL) and is only
needed for `to_dataset()` / `to_feature_collection()`.
