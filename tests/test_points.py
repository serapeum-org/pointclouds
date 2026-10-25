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


class TestToCrs:
    """Reprojecting a cloud with pyproj (no GDAL, no pyramids)."""

    def test_sets_the_target_crs_and_keeps_the_count(self):
        """The returned cloud is in the target CRS and holds the same number of points."""
        cloud = PointCloud(
            x=np.array([155000.0, 155038.0]), y=np.array([463000.0, 463038.0]), z=np.zeros(2), crs="EPSG:28992"
        )

        utm = cloud.to_crs("EPSG:32631")

        assert utm.crs.to_epsg() == 32631
        assert len(utm) == 2
        assert cloud.crs.to_epsg() == 28992, "the source cloud is left untouched"

    def test_axis_order_is_xy_even_for_a_latlon_crs(self):
        """EPSG:4326's registry order is lat/lon, yet always_xy puts longitude in x and latitude in y."""
        # 155000, 463000 in RD New is the Amersfoort datum point: ~5.39 E, ~52.16 N.
        cloud = PointCloud(x=np.array([155000.0]), y=np.array([463000.0]), z=np.zeros(1), crs="EPSG:28992")

        wgs = cloud.to_crs("EPSG:4326")

        assert wgs.x[0] == pytest.approx(5.387, abs=0.01), "x must be longitude, not latitude"
        assert wgs.y[0] == pytest.approx(52.155, abs=0.01), "y must be latitude"

    def test_a_round_trip_recovers_the_source_coordinates(self):
        """28992 -> 4326 -> 28992 returns the original x/y within floating-point tolerance."""
        x0 = np.array([155000.0, 160000.0, 180000.0])
        y0 = np.array([463000.0, 400000.0, 450000.0])
        cloud = PointCloud(x=x0, y=y0, z=np.zeros(3), crs="EPSG:28992")

        back = cloud.to_crs("EPSG:4326").to_crs("EPSG:28992")

        assert back.x == pytest.approx(x0, abs=1e-3)
        assert back.y == pytest.approx(y0, abs=1e-3)

    def test_attributes_are_carried_over_aligned(self):
        """Attribute arrays survive the transform unchanged and stay aligned with the coordinates."""
        cloud = PointCloud(
            x=np.array([155000.0, 155038.0]),
            y=np.array([463000.0, 463038.0]),
            z=np.array([1.0, 2.0]),
            classification=np.array([2, 5], dtype=np.uint8),
            crs="EPSG:28992",
        )

        out = cloud.to_crs("EPSG:32631")

        assert out.classification.tolist() == [2, 5]
        assert out.classification.shape == out.x.shape

    def test_a_cloud_without_a_crs_refuses_to_reproject(self, tiny_cloud):
        """With crs=None there is no source to transform from, so to_crs raises rather than guessing."""
        with pytest.raises(ValueError, match="crs is None"):
            tiny_cloud.to_crs("EPSG:4326")

    def test_horizontal_only_leaves_z_in_the_source_datum(self):
        """The default transform touches x/y only; z is carried over unchanged."""
        z0 = np.array([10.0, 20.0, 30.0])
        cloud = PointCloud(
            x=np.array([155000.0, 160000.0, 180000.0]),
            y=np.array([463000.0, 400000.0, 450000.0]),
            z=z0,
            crs="EPSG:28992",
        )

        out = cloud.to_crs("EPSG:32631")

        assert np.array_equal(out.z, z0)

    def test_vertical_shift_is_refused_without_a_vertical_crs(self):
        """vertical=True on a 2-D -> 2-D transform raises rather than silently leaving z in the source datum."""
        cloud = PointCloud(x=np.array([155000.0]), y=np.array([463000.0]), z=np.zeros(1), crs="EPSG:28992")

        with pytest.raises(ValueError, match="vertical"):
            cloud.to_crs("EPSG:32631", vertical=True)

    def test_vertical_shift_transforms_z_with_a_3d_crs(self):
        """vertical=True on a 3-D CRS on both sides actually moves z (EPSG:4979 geographic -> EPSG:4978 ECEF)."""
        cloud = PointCloud(x=np.array([5.0]), y=np.array([52.0]), z=np.array([0.0]), crs="EPSG:4979")

        ecef = cloud.to_crs("EPSG:4978", vertical=True)

        assert ecef.crs.to_epsg() == 4978
        assert np.isfinite(ecef.z[0])
        assert abs(ecef.z[0]) > 1.0, "ellipsoidal height 0 maps to a large ECEF Z component"

    def test_it_loads_a_lazy_cloud(self, sample_laz):
        """to_crs is a record operation, so it loads a lazy cloud before transforming."""
        cloud = PointCloud.read_file(sample_laz)

        out = cloud.to_crs("EPSG:4326")

        assert len(out) == EXPECTED_COUNT
        assert out.crs.to_epsg() == 4326


def test_the_phase_0_vertical_slice(sample_laz):
    """The §11.6 slice verbatim: it defines "scaffold done" and must run on the core install.

    No pyramids, no GDAL - just numpy, laspy and pyproj.
    """
    cloud = PointCloud.read_file(sample_laz)

    assert cloud.crs.to_epsg() == 28992
    assert cloud.bounds == pytest.approx(EXPECTED_BOUNDS)
    assert cloud.count == EXPECTED_COUNT
    assert cloud.to_array(("X", "Y", "Z")).shape == (EXPECTED_COUNT, 3)
