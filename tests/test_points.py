"""The PointCloud container: construction, the lazy/eager read split, and dimension access."""

from __future__ import annotations

import numpy as np
import pytest

from pointclouds import PointCloud
from pointclouds.io import las

EXPECTED_COUNT = 400
EXPECTED_BOUNDS = (155000.0, 463000.0, 155038.0, 463038.0)


def _forbid_record_reads(monkeypatch):
    """Make any record read fail, so a test can prove the header path never triggers one."""

    def explode(*args, **kwargs):
        raise AssertionError("this path must not read point records")

    monkeypatch.setattr(las, "read_points", explode)


class TestConstruction:
    """Building a cloud from arrays already in hand."""

    def test_coordinate_arrays_must_share_a_length(self):
        """x/y/z of differing lengths break the alignment invariant and are refused."""
        with pytest.raises(ValueError, match="must share a length"):
            PointCloud(x=np.zeros(3), y=np.zeros(2), z=np.zeros(3))

    def test_absent_attributes_become_empty_arrays(self, tiny_cloud):
        """An attribute the caller did not supply is an empty array, never None."""
        assert tiny_cloud.intensity.size == 0
        assert tiny_cloud.classification.size == 4

    def test_crs_accepts_a_string(self):
        """A CRS given as a string is normalised to a pyproj CRS."""
        cloud = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), crs="EPSG:4326")

        assert cloud.crs.to_epsg() == 4326

    def test_an_array_cloud_is_already_loaded(self, tiny_cloud):
        """There is nothing to defer when the arrays came from the caller."""
        assert tiny_cloud.loaded is True
        assert tiny_cloud.source is None

    def test_count_equals_len_without_a_header(self, tiny_cloud):
        """With no header, count can only mean how many points are held."""
        assert tiny_cloud.count == len(tiny_cloud) == 4

    def test_bounds_are_computed_from_the_arrays(self, tiny_cloud):
        """Without a header, bounds come from the coordinates themselves."""
        assert tiny_cloud.bounds == (0.0, 0.0, 3.0, 1.0)

    def test_an_empty_cloud_has_no_bounds(self):
        """An empty cloud with no header cannot answer bounds, and says so."""
        empty = PointCloud(x=np.empty(0), y=np.empty(0), z=np.empty(0))

        with pytest.raises(ValueError, match="no bounds"):
            _ = empty.bounds

    def test_dimensions_list_the_non_empty_attributes(self, tiny_cloud):
        """An array cloud reports its coordinates plus whichever attributes carry data."""
        assert tiny_cloud.dimensions == ("X", "Y", "Z", "classification")


class TestHeaderFirstRead:
    """read_file is header-first: metadata now, records when asked."""

    def test_a_read_is_lazy_by_default(self, sample_laz):
        """Nothing is decompressed until a record array is touched."""
        cloud = PointCloud.read_file(sample_laz)

        assert cloud.loaded is False
        assert cloud.source == sample_laz

    def test_metadata_answers_without_reading_records(self, sample_file, monkeypatch):
        """count, bounds, crs and dimensions all come from the header - proven by forbidding record reads."""
        cloud = PointCloud.read_file(sample_file)
        _forbid_record_reads(monkeypatch)

        assert cloud.count == EXPECTED_COUNT
        assert cloud.bounds == pytest.approx(EXPECTED_BOUNDS)
        assert cloud.crs.to_epsg() == 28992
        assert "intensity" in {name.lower() for name in cloud.dimensions}
        assert cloud.loaded is False

    def test_repr_does_not_load(self, sample_laz, monkeypatch):
        """Printing a cloud is never expensive."""
        cloud = PointCloud.read_file(sample_laz)
        _forbid_record_reads(monkeypatch)

        assert "lazy" in repr(cloud)
        assert "count=400" in repr(cloud)

    def test_touching_a_coordinate_loads_the_records(self, sample_laz):
        """First access to x/y/z decompresses, once."""
        cloud = PointCloud.read_file(sample_laz)

        assert cloud.x.shape == (EXPECTED_COUNT,)
        assert cloud.loaded is True

    def test_touching_an_attribute_loads_the_records(self, sample_laz):
        """The attribute arrays are on the same deferred path as the coordinates."""
        cloud = PointCloud.read_file(sample_laz)

        assert set(np.unique(cloud.classification)) == {2, 5}
        assert cloud.loaded is True

    def test_every_record_array_is_reachable(self, sample_laz):
        """All six mapped dimensions come back aligned, so nothing is wired to the wrong array."""
        cloud = PointCloud.read_file(sample_laz, lazy=False)

        arrays = (cloud.x, cloud.y, cloud.z, cloud.intensity, cloud.classification, cloud.return_number)

        assert all(values.shape == (EXPECTED_COUNT,) for values in arrays)
        assert cloud.y.min() == pytest.approx(EXPECTED_BOUNDS[1])
        assert cloud.z.min() > 0
        assert cloud.intensity.max() > 0
        assert set(np.unique(cloud.return_number)) == {1, 2}

    def test_eager_read_loads_immediately(self, sample_laz, monkeypatch):
        """lazy=False reads the records up front, so later access needs no load."""
        cloud = PointCloud.read_file(sample_laz, lazy=False)
        _forbid_record_reads(monkeypatch)

        assert cloud.loaded is True
        assert cloud.x.shape == (EXPECTED_COUNT,)

    def test_load_is_idempotent(self, sample_laz):
        """Calling load twice reads once and returns the same cloud."""
        cloud = PointCloud.read_file(sample_laz).load()
        first = cloud.x

        assert cloud.load().x is first

    def test_len_forces_a_load_and_agrees_with_count(self, sample_laz):
        """count is the cheap question and len is the honest one; they agree once loaded."""
        cloud = PointCloud.read_file(sample_laz)

        assert cloud.count == EXPECTED_COUNT
        assert cloud.loaded is False
        assert len(cloud) == EXPECTED_COUNT
        assert cloud.loaded is True

    def test_both_las_and_laz_read_the_same_cloud(self, sample_las, sample_laz):
        """Compression is an encoding detail; the two fixtures hold identical points."""
        uncompressed = PointCloud.read_file(sample_las, lazy=False)
        compressed = PointCloud.read_file(sample_laz, lazy=False)

        assert np.array_equal(uncompressed.to_array(), compressed.to_array())

    def test_a_missing_file_fails_at_read_file(self, tmp_path):
        """The error surfaces when the file is opened, not later on first access."""
        with pytest.raises(FileNotFoundError):
            PointCloud.read_file(tmp_path / "absent.laz")


class TestToArray:
    """Column-stacking dimensions into an (N, k) array."""

    def test_default_is_xyz(self, tiny_cloud):
        """The default stack is the three coordinates."""
        assert tiny_cloud.to_array().shape == (4, 3)

    def test_dimension_names_are_case_insensitive(self, tiny_cloud):
        """The names ``X`` and ``x`` select the same dimension."""
        assert np.array_equal(tiny_cloud.to_array(("X", "Z")), tiny_cloud.to_array(("x", "z")))

    def test_attributes_can_be_stacked_too(self, tiny_cloud):
        """Attribute arrays are stackable alongside the coordinates."""
        stacked = tiny_cloud.to_array(("Z", "classification"))

        assert stacked.shape == (4, 2)
        assert stacked[:, 1].tolist() == [2.0, 5.0, 2.0, 6.0]

    def test_unknown_dimension_is_refused(self, tiny_cloud):
        """An unknown name fails loudly, listing what the cloud holds."""
        with pytest.raises(KeyError, match="unknown dimension"):
            tiny_cloud.to_array(("X", "reflectance"))

    def test_it_loads_a_lazy_cloud(self, sample_laz):
        """to_array is a record operation, so it loads what it needs."""
        cloud = PointCloud.read_file(sample_laz)

        assert cloud.to_array(("X", "Y", "Z")).shape == (EXPECTED_COUNT, 3)


def test_the_phase_0_vertical_slice(sample_laz):
    """The §11.6 slice verbatim: it defines "scaffold done" and must run on the core install.

    No pyramids, no GDAL - just numpy, laspy and pyproj.
    """
    cloud = PointCloud.read_file(sample_laz)

    assert cloud.crs.to_epsg() == 28992
    assert cloud.bounds == pytest.approx(EXPECTED_BOUNDS)
    assert cloud.count == EXPECTED_COUNT
    assert cloud.to_array(("X", "Y", "Z")).shape == (EXPECTED_COUNT, 3)


class TestReadWindowDimensions:
    """`PointCloud.read_window` should accept `dimensions=` like the engine, and stay coupled to the ctor (L4)."""

    def test_a_dimension_subset_builds_a_cloud(self, sample_copc):
        """Requesting a subset still yields a usable cloud - x/y/z are read even if not asked for."""
        cloud = PointCloud.read_window(sample_copc, bbox=(155000, 463000, 155020, 463020), dimensions=("z",))

        assert len(cloud) == 121
        assert cloud.z.shape == (121,)
        assert cloud.classification.size == 0

    def test_canonical_dimensions_match_the_constructor(self):
        """`cls(crs=crs, **arrays)` depends on CANONICAL's keys being exactly the ctor's point parameters.

        If either side moves, the splat breaks at a call site; this makes it fail loudly here instead.
        """
        import inspect

        from pointclouds.io import _records

        params = set(inspect.signature(PointCloud.__init__).parameters) - {"self", "crs"}
        assert set(_records.CANONICAL) == params, (
            f"CANONICAL {sorted(_records.CANONICAL)} drifted from ctor params {sorted(params)}"
        )
