"""The LAS/LAZ reader's completion (#18): full attribute table, streaming, and the write round-trip.

The write path is assembled by laspy from an in-memory cloud; the read path is this package's own. So a
round-trip here exercises both halves - a cloud written by ``to_file`` / ``las.write`` and read back by
``read_file`` / ``read_points`` must come back with its attribute table and CRS intact, which is #18's stated
acceptance criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointclouds import PointCloud
from pointclouds.io import _records, las

ORIGIN = (155000.0, 463000.0)


def _rich_cloud(n: int = 64) -> PointCloud:
    """A cloud carrying the standard attribute table #18 names: classification/intensity/return + gps_time/RGB."""
    rng = np.random.default_rng(0)
    return PointCloud(
        x=ORIGIN[0] + np.arange(n, dtype=float),
        y=ORIGIN[1] + np.zeros(n),
        z=np.linspace(0.0, 10.0, n),
        crs="EPSG:28992",
        intensity=rng.integers(0, 2000, n).astype(np.uint16),
        return_number=rng.integers(1, 3, n).astype(np.uint8),
        number_of_returns=rng.integers(1, 3, n).astype(np.uint8),
        classification=rng.integers(1, 10, n).astype(np.uint8),
        gps_time=np.linspace(1.0e5, 1.0e5 + 1.0, n),
        red=rng.integers(0, 65535, n).astype(np.uint16),
        green=rng.integers(0, 65535, n).astype(np.uint16),
        blue=rng.integers(0, 65535, n).astype(np.uint16),
    )


class TestAttributeTable:
    """The canonical attribute table is the full standard LAS set, exposed uniformly by every read path."""

    def test_canonical_covers_the_standard_dimensions(self):
        """#1263's named dimensions are all in the canonical table."""
        assert {
            "intensity",
            "return_number",
            "number_of_returns",
            "classification",
            "scan_angle",
            "point_source_id",
            "gps_time",
            "red",
            "green",
            "blue",
            "nir",
        } <= set(_records.CANONICAL)

    def test_a_pointcloud_exposes_every_standard_attribute(self):
        """Each standard dimension is reachable both as a named property and via the attributes mapping."""
        cloud = _rich_cloud(8)

        assert cloud.gps_time.shape == (8,)
        assert cloud.red.shape == cloud.green.shape == cloud.blue.shape == (8,)
        assert cloud.number_of_returns.shape == (8,)
        assert {"gps_time", "red", "green", "blue"} <= set(cloud.attributes)

    def test_an_absent_standard_attribute_is_empty(self):
        """A dimension the cloud does not carry reads back as a zero-length array of the canonical dtype."""
        cloud = PointCloud(x=np.zeros(3), y=np.zeros(3), z=np.zeros(3))

        assert cloud.nir.size == 0 and cloud.nir.dtype == np.uint16
        assert cloud.gps_time.size == 0 and cloud.gps_time.dtype == np.float64


class TestWriteRoundTrip:
    """A cloud written out and read back keeps its attribute table and CRS (the #18 acceptance criterion)."""

    @pytest.mark.parametrize("suffix", [".laz", ".las"])
    def test_full_attribute_round_trip(self, tmp_path, suffix):
        """Every populated standard attribute, and the CRS, survive a write + read."""
        cloud = _rich_cloud()
        out = cloud.to_file(tmp_path / f"rt{suffix}")

        back = PointCloud.read_file(out, lazy=False)

        assert back.count == cloud.count
        assert back.crs.to_epsg() == 28992
        # Coordinates are quantised to the header scale (0.01 m), so they round-trip to within half a step.
        np.testing.assert_allclose(back.x, cloud.x, atol=0.005)
        np.testing.assert_allclose(back.z, cloud.z, atol=0.005)
        for name in ("intensity", "return_number", "number_of_returns", "classification"):
            assert np.array_equal(getattr(back, name), getattr(cloud, name)), name
        np.testing.assert_allclose(back.gps_time, cloud.gps_time, rtol=0, atol=1e-6)
        for name in ("red", "green", "blue"):
            assert np.array_equal(getattr(back, name), getattr(cloud, name)), name

    def test_the_point_format_grows_to_fit_the_attributes(self, tmp_path):
        """A cloud with RGB is written to a format that carries it; NIR bumps it again."""
        PointCloud(
            x=np.zeros(2),
            y=np.zeros(2),
            z=np.zeros(2),
            red=np.ones(2, np.uint16),
            green=np.ones(2, np.uint16),
            blue=np.ones(2, np.uint16),
        ).to_file(tmp_path / "rgb.laz")
        assert las.read_header(tmp_path / "rgb.laz").point_format == 7

        PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2), nir=np.ones(2, np.uint16)).to_file(tmp_path / "nir.laz")
        assert las.read_header(tmp_path / "nir.laz").point_format == 8

    def test_a_plain_cloud_uses_point_format_six(self, tmp_path):
        """With no colour, the write uses the compact LAS 1.4 format that still carries gps_time."""
        PointCloud(  # gps_time but no colour
            x=np.zeros(4),
            y=np.zeros(4),
            z=np.zeros(4),
            classification=np.ones(4, np.uint8),
            gps_time=np.linspace(0.0, 1.0, 4),
        ).to_file(tmp_path / "plain.laz")

        assert las.read_header(tmp_path / "plain.laz").point_format == 6

    def test_an_extra_dimension_round_trips(self, tmp_path):
        """A non-standard dimension is written as ExtraBytes and read back by its own name and dtype."""
        cloud = PointCloud(
            x=np.zeros(3),
            y=np.zeros(3),
            z=np.zeros(3),
            amplitude=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        )
        out = cloud.to_file(tmp_path / "extra.laz")

        back = PointCloud.read_file(out, lazy=False)
        assert "amplitude" in back.attributes
        np.testing.assert_allclose(back.attributes["amplitude"], [1.5, 2.5, 3.5])
        assert back.attributes["amplitude"].dtype == np.float32

    def test_a_bad_suffix_is_refused(self, tmp_path):
        """Writing to a non-LAS/LAZ path is refused before laspy is involved."""
        with pytest.raises(ValueError, match="not a LAS/LAZ file"):
            las.write(tmp_path / "cloud.ply", np.zeros(1), np.zeros(1), np.zeros(1))

    def test_a_mismatched_attribute_length_is_refused(self, tmp_path):
        """An attribute array that is neither empty nor one-per-point is rejected."""
        with pytest.raises(ValueError, match="length"):
            las.write(
                tmp_path / "bad.laz",
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                attributes={"classification": np.zeros(2, np.uint8)},
            )


class TestStreaming:
    """`read_points_chunked` / `iter_chunks` read a plain tile a slice at a time."""

    def test_chunks_cover_every_point_once(self, tmp_path):
        """The chunks concatenate back to the whole cloud, in order, with nothing missed or doubled."""
        cloud = _rich_cloud(50)
        out = cloud.to_file(tmp_path / "stream.laz")

        chunks = list(las.read_points_chunked(out, points_per_iteration=20))

        assert [c["x"].size for c in chunks] == [20, 20, 10]
        np.testing.assert_allclose(np.concatenate([c["x"] for c in chunks]), cloud.x)

    def test_iter_chunks_yields_clouds_with_the_crs(self, tmp_path):
        """`PointCloud.iter_chunks` yields eager clouds carrying the file's CRS."""
        out = _rich_cloud(50).to_file(tmp_path / "stream.laz")

        clouds = list(PointCloud.iter_chunks(out, 20))

        assert [len(c) for c in clouds] == [20, 20, 10]
        assert all(c.crs.to_epsg() == 28992 for c in clouds)
        assert all(c.loaded for c in clouds)

    def test_a_non_positive_chunk_size_is_refused(self, tmp_path):
        """Zero or negative points-per-iteration has no meaning."""
        out = _rich_cloud(4).to_file(tmp_path / "s.laz")

        with pytest.raises(ValueError, match="points_per_iteration must be positive"):
            list(las.read_points_chunked(out, points_per_iteration=0))


class TestPreferWkt:
    """`read_header` resolves the CRS with prefer_wkt=True so an OGC-WKT VLR wins over the GeoTIFF keys."""

    def test_a_wkt_crs_round_trips(self, tmp_path):
        """A cloud written with a CRS (WKT VLR) reads its CRS back."""
        _rich_cloud(4).to_file(tmp_path / "crs.laz")

        assert las.read_header(tmp_path / "crs.laz").crs.to_epsg() == 28992

    def test_read_header_passes_prefer_wkt(self, tmp_path, monkeypatch):
        """The header read must call parse_crs(prefer_wkt=True), not the bare default-sensitive call."""
        out = _rich_cloud(4).to_file(tmp_path / "crs.laz")
        seen = {}
        original = las.laspy.header.LasHeader.parse_crs

        def spy(self, prefer_wkt=True):
            seen["prefer_wkt"] = prefer_wkt
            return original(self, prefer_wkt=prefer_wkt)

        monkeypatch.setattr(las.laspy.header.LasHeader, "parse_crs", spy)
        las.read_header(out)

        assert seen.get("prefer_wkt") is True, "read_header should ask for WKT preference explicitly"


class TestEdgeCases:
    """The remaining branches: an empty standard dim on write, a None attribute, the rarer properties."""

    def test_write_skips_a_present_but_empty_standard_dimension(self, tmp_path):
        """A standard attribute given as a zero-length array is treated as absent and simply not written."""
        las.write(
            tmp_path / "empty_attr.laz",
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            attributes={"classification": np.array([], dtype=np.uint8), "intensity": np.ones(3, np.uint16)},
        )

        back = PointCloud.read_file(tmp_path / "empty_attr.laz", lazy=False)
        assert np.array_equal(back.intensity, np.ones(3, np.uint16))
        assert set(np.unique(back.classification)) == {0}, "an unwritten classification defaults to zero"

    def test_a_none_attribute_is_dropped(self):
        """Passing an attribute as None means absent - it is not carried and reads back empty."""
        cloud = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2), intensity=None)

        assert "intensity" not in cloud.attributes
        assert cloud.intensity.size == 0

    def test_the_rarer_standard_properties_are_exposed(self):
        """scan_angle / point_source_id / user_data are reachable as properties, full-length when carried."""
        cloud = PointCloud(
            x=np.zeros(2),
            y=np.zeros(2),
            z=np.zeros(2),
            scan_angle=np.array([-5, 5], dtype=np.int16),
            point_source_id=np.array([1, 2], dtype=np.uint16),
            user_data=np.array([7, 8], dtype=np.uint8),
        )

        assert cloud.scan_angle.tolist() == [-5, 5]
        assert cloud.point_source_id.tolist() == [1, 2]
        assert cloud.user_data.tolist() == [7, 8]


class TestScanAngleUnits:
    """`scan_angle` maps only the LAS 1.4 field, never the differently-scaled legacy scan_angle_rank (H2)."""

    def test_a_legacy_scan_angle_rank_is_not_read_as_scan_angle(self, tmp_path):
        """A point-format-3 file (scan_angle_rank, whole degrees) must not surface as the canonical scan_angle.

        scan_angle_rank (pf0-5) is int8 whole degrees; scan_angle (pf6-10) is int16 in 0.006deg increments.
        Mapping one onto the other would silently change the physical meaning on a cross-format write, so the
        legacy field is left unmapped (absent) rather than conflated.
        """
        import laspy

        header = laspy.LasHeader(version="1.2", point_format=3)
        header.scales = [0.01, 0.01, 0.01]
        las_data = laspy.LasData(header)
        las_data.x = np.array([0.0, 1.0])
        las_data.y = np.array([0.0, 1.0])
        las_data.z = np.array([0.0, 0.0])
        las_data.scan_angle_rank = np.array([-15, 15], dtype=np.int8)
        las_data.write(str(tmp_path / "pf3.laz"))

        cloud = PointCloud.read_file(tmp_path / "pf3.laz", lazy=False)
        assert cloud.scan_angle.size == 0, "a legacy scan_angle_rank must not be read as the canonical scan_angle"

    def test_a_scan_angle_round_trips_on_a_modern_format(self, tmp_path):
        """A point-format-6 scan_angle (int16, the canonical field) survives a write/read unchanged."""
        cloud = PointCloud(
            x=np.zeros(3), y=np.zeros(3), z=np.zeros(3),
            scan_angle=np.array([-2500, 0, 2500], dtype=np.int16),
            gps_time=np.zeros(3),  # force point format 6
        )
        out = cloud.to_file(tmp_path / "pf6.laz")

        assert PointCloud.read_file(out, lazy=False).scan_angle.tolist() == [-2500, 0, 2500]


class TestVectorExtraDimension:
    """A multi-element (N, k) ExtraBytes dimension must read AND write (length check is per-point) (M2)."""

    def test_a_vector_extra_round_trips(self, tmp_path):
        """An (N, 3) extra dimension survives a write/read, keeping its shape and dtype.

        The length validation must count points (first axis = N), not total elements (N*3), or `to_file` wrongly
        rejects a cloud that was read cleanly.
        """
        cloud = PointCloud(
            x=np.zeros(2), y=np.zeros(2), z=np.zeros(2),
            rgb3=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        )
        out = cloud.to_file(tmp_path / "vec.laz")

        back = PointCloud.read_file(out, lazy=False)
        assert back.attributes["rgb3"].shape == (2, 3)
        assert back.attributes["rgb3"].tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_a_wrong_point_axis_length_is_still_refused(self, tmp_path):
        """A vector extra with the wrong number of points (not of total elements) is still rejected."""
        with pytest.raises(ValueError, match="length"):
            las.write(
                tmp_path / "bad.laz", np.zeros(2), np.zeros(2), np.zeros(2),
                attributes={"rgb3": np.zeros((3, 3), dtype=np.uint8)},
            )


class TestWrittenFormatMaterializesDimensions:
    """A written LAS 1.4 format records all its declared dims; absent ones read back zero-filled (M3)."""

    def test_absent_gps_time_reads_back_present_and_zero(self, tmp_path):
        """A cloud with no gps_time, written to point format 6 (which mandates it), reads back N zeros.

        This pins the documented round-trip behaviour: the file genuinely carries gps_time, so the reader shows
        it present rather than absent. The "empty means absent" convention is about a fresh read, not a
        write/read round-trip.
        """
        cloud = PointCloud(
            x=np.zeros(3), y=np.zeros(3), z=np.zeros(3),
            classification=np.array([2, 2, 5], dtype=np.uint8),  # no gps_time
        )
        assert cloud.gps_time.size == 0, "the source cloud has no gps_time"

        back = PointCloud.read_file(cloud.to_file(tmp_path / "nogps.laz"), lazy=False)

        assert back.gps_time.shape == (3,), "point format 6 mandates gps_time, so it is present after a round-trip"
        assert back.gps_time.tolist() == [0.0, 0.0, 0.0], "an unwritten gps_time is zero-filled, not meaningful"
        assert np.array_equal(back.classification, cloud.classification), "populated dims keep their values"
