# Description

- Please include a summary of the change and which issue is fixed.
- Please also include relevant motivation and context.
- List any dependencies that are required for this change.

# Issues

- Closes # (issue)

## Type of change

Check relevant points.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

# How Has This Been Tested?

- Please describe the tests that you ran to verify your changes.
- Provide instructions so we can reproduce.
- Please also list any relevant details for your test configuration

- [ ] `uv run pytest -m "not gis and not cloud"` (the GDAL-free core)
- [ ] `uv run --all-extras pytest` (full suite, with the `gis` / `cloud` extras)
- [ ] `uv run --all-extras pytest --nbval --nbval-lax docs/examples` (notebooks)

# Checklist:

- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] New and existing unit tests pass locally with my changes.
- [ ] Public symbols I added or changed carry Google-style docstrings.
- [ ] The documentation is updated (reference page, guide, or notebook as applicable).
- [ ] `docs/change-log.md` records the change.
- [ ] The core stays GDAL-free: only `src/pointclouds/convert.py` imports `pyramids`, behind a
      guarded in-function import naming the `[gis]` extra, and
      `tests/test_core_has_no_gdal.py` is still green.
- [ ] The change stays on this package's side of the boundary (ADR-0001): `pointclouds` owns point
      clouds, `pyramids` owns raster and vector. Anything belonging upstream is reported there
      instead of reimplemented here.
