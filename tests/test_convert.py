"""The pyramids boundary, from the other side: what `require_pyramids` does when pyramids IS installed.

`tests/test_core_has_no_gdal.py` covers the absent case - the guarded import raising a hint that names the
`[gis]` extra. The present case can only run where the extra is installed, which is the `full` pixi environment,
so every test here carries the `gis` marker. `pixi run -e dev main` deselects them with
`-m "not gis and not cloud"`; `pixi run -e full test-all` runs them.

The module-level `importorskip` is the safety net for a bare `pytest` in an environment without the extra: the
marker decides *selection*, the importorskip keeps an unselected run from erroring on collection.
"""

from __future__ import annotations

import pytest

from pointclouds.convert import require_pyramids

pytest.importorskip("pyramids", reason="needs the [gis] extra (pyramids-gis)")

pytestmark = pytest.mark.gis


def test_require_pyramids_returns_the_dataset_and_georeference_classes():
    """With the extra installed, the guarded import hands back the two pyramids classes themselves."""
    dataset, georeference = require_pyramids()

    assert dataset.__name__ == "Dataset"
    assert georeference.__name__ == "GeoReference"


def test_the_returned_classes_come_from_pyramids():
    """The classes are pyramids' own, not a local stand-in - this is the boundary the [gis] extra buys."""
    dataset, georeference = require_pyramids()

    assert dataset.__module__.startswith("pyramids")
    assert georeference.__module__.startswith("pyramids")


def test_require_pyramids_is_stable_across_calls():
    """Repeated calls hand back the same class objects, so callers may hold on to them."""
    first_dataset, first_georeference = require_pyramids()
    second_dataset, second_georeference = require_pyramids()

    assert first_dataset is second_dataset
    assert first_georeference is second_georeference
