"""The laspy engine: header reads that touch no records, and record reads that honour the dimension list."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

from pointclouds.io import las

#: The fixtures are generated on a 20 x 20 grid at 2 m spacing - see examples/data/make_sample.py.
EXPECTED_COUNT = 400
EXPECTED_BOUNDS = (155000.0, 463000.0, 155038.0, 463038.0)


def test_read_header_reports_count_bounds_and_crs(sample_file):
    """The header carries the point count, the bounding box and the CRS as a pyproj object."""
    info = las.read_header(sample_file)

    assert info.count == EXPECTED_COUNT
    assert info.bounds == pytest.approx(EXPECTED_BOUNDS)
    assert isinstance(info.crs, CRS)
    assert info.crs.to_epsg() == 28992
    assert info.version == "1.4"
    assert info.point_format == 3


def test_read_header_lists_the_point_format_dimensions(sample_file):
    """Dimensions come from the point format, so the attribute fields are all declared."""
    dimensions = {name.lower() for name in las.read_header(sample_file).dimensions}

    assert {"x", "y", "z", "intensity", "classification", "return_number"} <= dimensions


def test_read_header_touches_no_point_records(sample_file, monkeypatch):
    """A header read must not decompress records - the whole point of the header-first design.

    Enforced by making the record path explode: if `read_header` ever reaches for `laspy.read`, this fails.
    """

    def explode(*args, **kwargs):
        raise AssertionError("read_header must not read point records")

    monkeypatch.setattr(las.laspy, "read", explode)

    assert las.read_header(sample_file).count == EXPECTED_COUNT


def test_read_points_returns_every_canonical_dimension(sample_file):
    """With no dimension list, every dimension the engine maps comes back, aligned and typed."""
    arrays = las.read_points(sample_file)

    assert set(arrays) == {"x", "y", "z", "intensity", "classification", "return_number"}
    assert all(values.shape == (EXPECTED_COUNT,) for values in arrays.values())
    assert arrays["x"].dtype == np.float64
    assert arrays["intensity"].dtype == np.uint16
    assert arrays["classification"].dtype == np.uint8
    assert set(np.unique(arrays["classification"])) == {2, 5}


def test_read_points_honours_a_dimension_subset(sample_laz):
    """Only the requested dimensions are returned, and the names are case-insensitive."""
    arrays = las.read_points(sample_laz, dimensions=("X", "Z"))

    assert set(arrays) == {"x", "z"}
    assert arrays["z"].shape == (EXPECTED_COUNT,)


def test_read_points_rejects_an_unknown_dimension(sample_laz):
    """An unknown dimension fails loudly, listing what the engine does read."""
    with pytest.raises(KeyError, match="unknown dimension"):
        las.read_points(sample_laz, dimensions=("X", "nosuchdim"))


class _FakePointFormat:
    """A point format that declares no intensity dimension."""

    id = 0
    dimension_names = ("X", "Y", "Z", "classification", "return_number")


class _FakeLasData:
    """The minimum laspy.read() surface read_points touches, missing one attribute dimension."""

    point_format = _FakePointFormat()
    x = np.zeros(2)
    y = np.zeros(2)
    z = np.zeros(2)
    classification = np.ones(2, dtype=np.uint8)
    return_number = np.ones(2, dtype=np.uint8)


def test_a_dimension_the_point_format_lacks_comes_back_empty(sample_laz, monkeypatch):
    """A dimension the file does not declare is an empty array, not a missing key or an AttributeError."""
    monkeypatch.setattr(las.laspy, "read", lambda *args, **kwargs: _FakeLasData())

    arrays = las.read_points(sample_laz)

    assert arrays["intensity"].size == 0
    assert arrays["intensity"].dtype == np.uint16
    assert arrays["x"].shape == (2,)
    assert arrays["classification"].shape == (2,)


def test_missing_file_names_the_path(tmp_path):
    """A missing file raises FileNotFoundError naming the path, not a laspy internal error."""
    missing = tmp_path / "nope.laz"

    with pytest.raises(FileNotFoundError, match="nope.laz"):
        las.read_header(missing)


def test_unsupported_suffix_is_refused(tmp_path):
    """A non-LAS/LAZ suffix is refused before laspy is given a chance to guess."""
    other = tmp_path / "cloud.ply"
    other.write_bytes(b"")

    with pytest.raises(ValueError, match="not a LAS/LAZ file"):
        las.read_header(other)
