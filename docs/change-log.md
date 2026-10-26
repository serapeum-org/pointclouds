# Changelog

## 0.1.0 (2026-10-26)

- PointCloud is no longer a dataclass, so it no longer
has a generated __eq__ and two clouds compare by identity. The
generated __eq__ already raised "truth value of an array is ambiguous"
on any cloud with array fields, so no working comparison is lost.

### Feat

- **io**: add COPC and EPT spatially-indexed windowed reads (#13)
- **ops**: add filter, decimate, clip, merge and to_grid to PointCloud (#12)
- **points**: add to_crs for pyproj reprojection (#11)
- **io**: read LAS/LAZ header-first into PointCloud (#8)
