"""The COPC engine: header reads that touch no records, and windowed reads that really do prune the octree.

These run against a committed COPC file with a known two-level octree (see `examples/data/make_copc_sample.py`):
a root node holding every 16th point, then eight level-1 octants holding the rest. That shape is what makes the
assertions below meaningful - a mocked reader could satisfy "returns fewer points" without ever skipping a
chunk, which is the only thing a windowed read is for.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from pyproj import CRS

from pointclouds import PointCloud
from pointclouds.io import copc, las

#: The fixture is a 40 x 40 grid at 2 m spacing - see examples/data/make_copc_sample.py.
EXPECTED_COUNT = 1600
EXPECTED_BOUNDS = (155000.0, 463000.0, 155078.0, 463078.0)
#: Root spacing recorded in the COPC info VLR: the extent divided by the root's points-per-axis.
EXPECTED_SPACING = 7.8
#: Points in the root node (level 0) and in the eight level-1 octants.
ROOT_COUNT = 100
LEVEL_1_COUNT = 1500
#: A 20 x 20 m box on the fixture's lower-left corner, and how many of the 1600 points fall inside it.
CORNER_BBOX = (155000.0, 463000.0, 155020.0, 463020.0)
CORNER_COUNT = 121


def _count_queried_nodes(monkeypatch) -> list[int]:
    """Record how many octree nodes each query ends up reading.

    This is the measurement the whole feature rests on: a window is only worth having if it visits fewer nodes,
    not merely if it returns fewer points.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        A list that gains one entry, the node count, per query performed.
    """
    import laspy.copc as laspy_copc

    visited: list[int] = []
    real_loader = laspy_copc.load_octree_for_query

    def spy(*args, **kwargs):
        nodes = real_loader(*args, **kwargs)
        visited.append(len(nodes))
        return nodes

    monkeypatch.setattr(laspy_copc, "load_octree_for_query", spy)
    return visited


class TestHeader:
    """`read_header` answers from the header and the COPC info VLR alone."""

    def test_it_reports_count_bounds_crs_and_spacing(self, sample_copc):
        """The header carries the point count, the bounding box, the CRS, and the octree's root spacing."""
        info = copc.read_header(sample_copc)

        assert info.count == EXPECTED_COUNT
        assert info.bounds == pytest.approx(EXPECTED_BOUNDS)
        assert isinstance(info.crs, CRS)
        assert info.crs.to_epsg() == 28992
        assert info.spacing == pytest.approx(EXPECTED_SPACING)
        assert info.version == "1.4"
        assert info.point_format == 6  # COPC is always point format 6, 7 or 8

    def test_it_is_a_las_header_info(self, sample_copc):
        """A COPC header is a LAS header with one field added, so anything typed on the base accepts it."""
        assert isinstance(copc.read_header(sample_copc), las.LasHeaderInfo)

    def test_it_lists_the_point_format_dimensions(self, sample_copc):
        """Dimensions come from the point format, so the attribute fields are all declared."""
        dimensions = {name.lower() for name in copc.read_header(sample_copc).dimensions}

        assert {"x", "y", "z", "intensity", "classification", "return_number"} <= dimensions

    def test_it_touches_no_point_records(self, sample_copc, monkeypatch):
        """A header read must not run a query - the header-first contract the LAS engine keeps too.

        Enforced by making the query path explode: if `read_header` ever reaches for it, this fails.
        """

        def explode(*args, **kwargs):
            raise AssertionError("read_header must not query the octree")

        monkeypatch.setattr(copc.laspy.CopcReader, "query", explode)

        assert copc.read_header(sample_copc).count == EXPECTED_COUNT


class TestIsCopc:
    """`is_copc` separates an indexed cloud from a plain one without reading records."""

    def test_a_copc_file_is_recognised(self, sample_copc):
        """The fixture carries a COPC info VLR as its first VLR, so it is a COPC."""
        assert copc.is_copc(sample_copc) is True

    def test_a_plain_laz_is_not(self, sample_laz):
        """A LAZ with no octree answers False rather than raising - that is the question being asked."""
        assert copc.is_copc(sample_laz) is False

    def test_a_las_suffix_is_an_error_not_a_no(self, sample_las):
        """An uncompressed `.las` can never be COPC, so the suffix is refused instead of answered."""
        with pytest.raises(ValueError, match="COPC is always LAZ-compressed"):
            copc.is_copc(sample_las)

    def test_a_file_object_can_be_asked_too(self, sample_copc, sample_laz):
        """A handle has no suffix to judge by, so the answer has to come from the bytes."""
        assert copc.is_copc(io.BytesIO(sample_copc.read_bytes())) is True
        assert copc.is_copc(io.BytesIO(sample_laz.read_bytes())) is False


class TestBoundingBoxWindow:
    """`bbox=` reads a sub-box by pruning the octree to the chunks that overlap it."""

    def test_no_window_reads_the_whole_cloud(self, sample_copc):
        """With no argument the query spans the file: the honest default, and the expensive one."""
        arrays = copc.read_window(sample_copc)

        assert arrays["x"].size == EXPECTED_COUNT

    def test_only_points_inside_the_box_come_back(self, sample_copc):
        """Points are clipped to the box, not merely to the chunks overlapping it."""
        arrays = copc.read_window(sample_copc, bbox=CORNER_BBOX)

        xmin, ymin, xmax, ymax = CORNER_BBOX
        assert arrays["x"].size == CORNER_COUNT
        assert arrays["x"].min() >= xmin
        assert arrays["x"].max() <= xmax
        assert arrays["y"].min() >= ymin
        assert arrays["y"].max() <= ymax

    def test_it_reads_fewer_nodes_than_the_whole_file(self, sample_copc, monkeypatch):
        """The point of the feature: a corner box must visit fewer octree nodes, not just return fewer points."""
        visited = _count_queried_nodes(monkeypatch)

        copc.read_window(sample_copc)
        copc.read_window(sample_copc, bbox=CORNER_BBOX)

        whole_file, windowed = visited
        assert windowed < whole_file, f"a corner box visited {windowed} nodes, same as the {whole_file} for all"

    def test_a_box_outside_the_data_reads_nothing(self, sample_copc):
        """An empty window is empty arrays, not an error and not a missing key."""
        arrays = copc.read_window(sample_copc, bbox=(0.0, 0.0, 1.0, 1.0))

        assert arrays["x"].size == 0
        assert arrays["classification"].size == 0

    def test_the_box_is_two_dimensional(self, sample_copc):
        """A box spanning x and y keeps the file's whole z range: elevation is not a read window."""
        arrays = copc.read_window(sample_copc, bbox=EXPECTED_BOUNDS)

        assert arrays["z"].size == EXPECTED_COUNT

    def test_every_canonical_dimension_comes_back(self, sample_copc):
        """A window returns the same dimension set, dtypes and alignment as a whole-file read."""
        arrays = copc.read_window(sample_copc, bbox=CORNER_BBOX)

        assert set(arrays) == {"x", "y", "z", "intensity", "classification", "return_number"}
        assert all(values.shape == (CORNER_COUNT,) for values in arrays.values())
        assert arrays["x"].dtype == np.float64
        assert arrays["intensity"].dtype == np.uint16
        assert arrays["classification"].dtype == np.uint8

    def test_a_dimension_subset_is_honoured(self, sample_copc):
        """Only the requested dimensions are returned, and the names are case-insensitive."""
        arrays = copc.read_window(sample_copc, bbox=CORNER_BBOX, dimensions=("X", "Z"))

        assert set(arrays) == {"x", "z"}
        assert arrays["z"].shape == (CORNER_COUNT,)

    def test_a_degenerate_box_is_accepted(self, sample_copc):
        """A zero-width box is a point query, not an inverted box - it must not be refused.

        The fixture is a 2 m grid with a point exactly on the corner, so collapsing the box onto that corner
        selects it rather than returning nothing.
        """
        xmin, ymin, _, _ = CORNER_BBOX

        arrays = copc.read_window(sample_copc, bbox=(xmin, ymin, xmin, ymin))

        assert arrays["x"].size == 1
        assert arrays["x"][0] == pytest.approx(xmin)


class TestResolutionAndLevel:
    """`resolution=` and `level=` are the level-of-detail half of a window."""

    def test_the_root_spacing_reads_the_root_alone(self, sample_copc):
        """Asking for the spacing the root already has stops the descent there."""
        arrays = copc.read_window(sample_copc, resolution=EXPECTED_SPACING)

        assert arrays["x"].size == ROOT_COUNT

    def test_a_fine_resolution_reads_everything(self, sample_copc):
        """A resolution finer than the deepest level cannot prune, so the whole cloud comes back."""
        arrays = copc.read_window(sample_copc, resolution=0.5)

        assert arrays["x"].size == EXPECTED_COUNT

    def test_a_coarse_read_is_thinned_not_cropped(self, sample_copc):
        """The LOD claim: a coarse read covers the whole area sparsely, rather than a dense corner of it.

        The root is a regular subsample, so its own extent stays within one sample step of the file's. A cropped
        read would cover a fraction of each axis instead - a quarter, for one octant.
        """
        arrays = copc.read_window(sample_copc, resolution=EXPECTED_SPACING)

        xmin, ymin, xmax, ymax = EXPECTED_BOUNDS
        covered_x = (arrays["x"].max() - arrays["x"].min()) / (xmax - xmin)
        covered_y = (arrays["y"].max() - arrays["y"].min()) / (ymax - ymin)

        assert covered_x > 0.8, f"the coarse read spans only {covered_x:.0%} of the x extent"
        assert covered_y > 0.8, f"the coarse read spans only {covered_y:.0%} of the y extent"
        assert arrays["x"].min() == pytest.approx(xmin)
        assert arrays["y"].min() == pytest.approx(ymin)

    def test_an_exact_level_selects_only_that_level(self, sample_copc):
        """`level=` names octree levels directly, and the levels partition the cloud."""
        root = copc.read_window(sample_copc, level=0)
        deeper = copc.read_window(sample_copc, level=1)

        assert root["x"].size == ROOT_COUNT
        assert deeper["x"].size == LEVEL_1_COUNT
        assert root["x"].size + deeper["x"].size == EXPECTED_COUNT

    def test_a_level_range_is_accepted(self, sample_copc):
        """A range spanning both levels is the whole cloud."""
        arrays = copc.read_window(sample_copc, level=range(0, 2))

        assert arrays["x"].size == EXPECTED_COUNT

    def test_a_coarse_read_visits_fewer_nodes(self, sample_copc, monkeypatch):
        """Level pruning has to skip nodes, for the same reason a bounding box does."""
        visited = _count_queried_nodes(monkeypatch)

        copc.read_window(sample_copc)
        copc.read_window(sample_copc, resolution=EXPECTED_SPACING)

        whole_file, coarse = visited
        assert coarse < whole_file

    def test_a_resolution_coarser_than_the_root_still_reads_the_root(self, sample_copc):
        """There is no level above the root, so an absurdly coarse ask returns the root, never nothing.

        A caller zoomed far out should get a cheap cloud, not an empty one.
        """
        arrays = copc.read_window(sample_copc, resolution=10_000.0)

        assert arrays["x"].size == ROOT_COUNT

    def test_a_box_and_a_resolution_compose(self, sample_copc):
        """The two halves of a window are independent, so they can be asked for together."""
        arrays = copc.read_window(sample_copc, bbox=CORNER_BBOX, resolution=EXPECTED_SPACING)

        xmin, ymin, xmax, ymax = CORNER_BBOX
        assert 0 < arrays["x"].size < CORNER_COUNT
        assert arrays["x"].max() <= xmax
        assert arrays["y"].max() <= ymax


class TestRejections:
    """What the engine refuses, and whether the message sends the caller somewhere useful."""

    def test_a_plain_laz_cannot_be_windowed(self, sample_laz):
        """A file with no octree is refused by name, pointing at the whole-file read that does work."""
        with pytest.raises(ValueError, match="carries no COPC octree"):
            copc.read_window(sample_laz, bbox=CORNER_BBOX)

    def test_the_refusal_names_the_alternative(self, sample_laz):
        """The error has to say what to do instead, or it just restates the failure."""
        with pytest.raises(ValueError, match="read_file"):
            copc.read_window(sample_laz)

    def test_a_missing_file_names_the_path(self, tmp_path):
        """A missing file raises FileNotFoundError naming the path, not a laspy internal error."""
        with pytest.raises(FileNotFoundError, match="nope.laz"):
            copc.read_window(tmp_path / "nope.laz")

    def test_an_unsupported_suffix_is_refused(self, tmp_path):
        """A non-LAZ suffix is refused before laspy is given a chance to guess."""
        other = tmp_path / "cloud.ply"
        other.write_bytes(b"")

        with pytest.raises(ValueError, match="not a COPC file"):
            copc.read_window(other)

    def test_a_bbox_needs_four_values(self, sample_copc):
        """Three values is a mistake worth catching before the query runs."""
        with pytest.raises(ValueError, match="bbox must be"):
            copc.read_window(sample_copc, bbox=(1.0, 2.0, 3.0))

    def test_an_inverted_bbox_is_refused(self, sample_copc):
        """Swapped corners would silently select nothing, so they are an error instead."""
        with pytest.raises(ValueError, match="bbox is inverted"):
            copc.read_window(sample_copc, bbox=(155020.0, 463000.0, 155000.0, 463020.0))

    @pytest.mark.parametrize("resolution", [0.0, -1.0])
    def test_resolution_must_be_positive(self, sample_copc, resolution):
        """A zero or negative spacing has no meaning as a target resolution."""
        with pytest.raises(ValueError, match="resolution must be positive"):
            copc.read_window(sample_copc, resolution=resolution)

    def test_a_negative_level_is_refused(self, sample_copc):
        """Octree levels start at 0 at the root."""
        with pytest.raises(ValueError, match="non-negative octree level"):
            copc.read_window(sample_copc, level=-1)

    def test_resolution_and_level_are_mutually_exclusive(self, sample_copc):
        """They are two spellings of the same limit; accepting both would hide which one won."""
        with pytest.raises(ValueError, match="not both"):
            copc.read_window(sample_copc, resolution=1.0, level=1)

    def test_an_unknown_dimension_is_refused(self, sample_copc):
        """An unknown dimension fails loudly, listing what the engine does read."""
        with pytest.raises(KeyError, match="unknown dimension"):
            copc.read_window(sample_copc, dimensions=("X", "nosuchdim"))


class TestFileObjectSource:
    """A window can be read from an open handle, which is how a remote cloud is reached."""

    def test_a_file_object_reads_the_same_window(self, sample_copc):
        """Bytes in memory behave exactly like the path they came from."""
        handle = io.BytesIO(sample_copc.read_bytes())

        arrays = copc.read_window(handle, bbox=CORNER_BBOX)

        assert arrays["x"].size == CORNER_COUNT

    def test_the_caller_keeps_the_handle(self, sample_copc):
        """laspy would close a handle it was handed; a handle the caller opened is the caller's to close."""
        handle = io.BytesIO(sample_copc.read_bytes())

        copc.read_window(handle, bbox=CORNER_BBOX)

        assert not handle.closed

    def test_one_handle_serves_several_reads(self, sample_copc):
        """Each read rewinds, so a header read followed by a window read does not start mid-file."""
        handle = io.BytesIO(sample_copc.read_bytes())

        assert copc.read_header(handle).count == EXPECTED_COUNT
        assert copc.read_window(handle, bbox=CORNER_BBOX)["x"].size == CORNER_COUNT
        assert copc.read_header(handle).count == EXPECTED_COUNT


class TestTheFixtureItself:
    """The generator and the committed fixture have to stay in step.

    Every count asserted in this module comes from the committed file, so if `make_copc_sample.py` stops
    producing a readable COPC - a laspy change to header writing, say - that should surface here rather than
    the next time someone regenerates the fixture and finds the suite broken.
    """

    def test_the_generator_still_produces_a_readable_copc(self, tmp_path, data_dir, monkeypatch):
        """Regenerating the fixture yields a file this engine reads, with the same shape as the committed one.

        Byte equality is deliberately not asserted: laspy controls the header layout, so a patch release could
        shift bytes without breaking anything that matters here.
        """
        monkeypatch.syspath_prepend(str(data_dir))  # the generator imports its sibling make_sample
        make_copc_sample = pytest.importorskip("make_copc_sample")

        regenerated = tmp_path / "sample.copc.laz"
        written = make_copc_sample.write(regenerated)

        assert written == EXPECTED_COUNT
        info = copc.read_header(regenerated)
        assert info.count == EXPECTED_COUNT
        assert info.bounds == pytest.approx(EXPECTED_BOUNDS)
        assert info.spacing == pytest.approx(EXPECTED_SPACING)
        assert copc.read_window(regenerated, level=0)["x"].size == ROOT_COUNT
        assert copc.read_window(regenerated, bbox=CORNER_BBOX)["x"].size == CORNER_COUNT


class TestPointCloudWindow:
    """The container surface: `PointCloud.read_window`, and `read_file` delegating to it."""

    def test_a_window_is_a_cloud_of_only_those_points(self, sample_copc):
        """The returned cloud holds the window, with the file's CRS."""
        cloud = PointCloud.read_window(sample_copc, bbox=CORNER_BBOX)

        assert cloud.count == CORNER_COUNT
        assert len(cloud) == CORNER_COUNT
        assert cloud.crs.to_epsg() == 28992

    def test_count_and_bounds_describe_the_window_not_the_file(self, sample_copc):
        """A windowed cloud carries no header, so it cannot report the file's 1600 points as its own."""
        cloud = PointCloud.read_window(sample_copc, bbox=CORNER_BBOX)

        assert cloud.count != EXPECTED_COUNT
        assert cloud.bounds == pytest.approx(CORNER_BBOX)

    def test_a_window_is_eager(self, sample_copc):
        """There is nothing left to load: the window was the read."""
        cloud = PointCloud.read_window(sample_copc, bbox=CORNER_BBOX)

        assert cloud.loaded is True
        assert cloud.source is None

    def test_every_record_array_is_reachable(self, sample_copc):
        """The attribute arrays arrive aligned with the coordinates, as on any other cloud."""
        cloud = PointCloud.read_window(sample_copc, bbox=CORNER_BBOX)

        assert cloud.intensity.shape == (CORNER_COUNT,)
        assert cloud.classification.shape == (CORNER_COUNT,)
        assert cloud.return_number.shape == (CORNER_COUNT,)
        assert cloud.to_array(("X", "Y", "Z")).shape == (CORNER_COUNT, 3)

    def test_an_empty_window_is_an_empty_cloud(self, sample_copc):
        """Zero points is a valid answer; only asking such a cloud for bounds is not."""
        cloud = PointCloud.read_window(sample_copc, bbox=(0.0, 0.0, 1.0, 1.0))

        assert cloud.count == 0
        with pytest.raises(ValueError, match="no bounds"):
            _ = cloud.bounds

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bbox": CORNER_BBOX},
            {"resolution": EXPECTED_SPACING},
            {"level": 0},
        ],
    )
    def test_read_file_delegates_any_window_argument(self, sample_copc, kwargs):
        """`read_file(bbox=)` / `(resolution=)` / `(level=)` must land on the same path as `read_window`."""
        via_read_file = PointCloud.read_file(sample_copc, **kwargs)
        direct = PointCloud.read_window(sample_copc, **kwargs)

        assert via_read_file.count == direct.count
        assert via_read_file.loaded is True

    def test_read_file_without_a_window_is_still_header_first(self, sample_copc, monkeypatch):
        """A COPC file read with no window keeps the lazy LAS path: it is a valid LAZ, after all."""

        def explode(*args, **kwargs):
            raise AssertionError("a header-first read must not decompress records")

        monkeypatch.setattr(copc.laspy, "read", explode)

        cloud = PointCloud.read_file(sample_copc)

        assert cloud.loaded is False
        assert cloud.count == EXPECTED_COUNT


class TestBoundingBoxRobustness:
    """Inputs laspy's integer clipping would turn into silently wrong data (H1, M1)."""

    def test_an_oversized_box_clamps_to_the_file_and_returns_everything(self, sample_copc):
        """A box far larger than the data reads the whole cloud, not an empty one.

        laspy clips in scaled int32 space, which overflows for coordinates past ~2.1e7 and hands back zero
        points with only a numpy warning. A generous sentinel box is an ordinary request and must mean "all of
        it".
        """
        arrays = copc.read_window(sample_copc, bbox=(-1e9, -1e9, 1e9, 1e9))

        assert arrays["x"].size == EXPECTED_COUNT

    @pytest.mark.parametrize(
        "bbox",
        [
            (float("nan"), 0.0, 1.0, 1.0),
            (0.0, 0.0, float("inf"), 1.0),
            (float("-inf"), float("-inf"), float("inf"), float("inf")),
        ],
    )
    def test_a_non_finite_box_is_refused(self, sample_copc, bbox):
        """`nan`/`inf` corners are rejected up front, not cast to int32 garbage.

        `nan` in particular slips the inversion check (`nan > nan` is False), so without an explicit finiteness
        guard it would reach laspy and return an empty cloud in violation of the documented ValueError.
        """
        with pytest.raises(ValueError, match="finite"):
            copc.read_window(sample_copc, bbox=bbox)

    def test_points_are_clipped_exactly_to_the_box(self, sample_copc):
        """No point outside the requested box is returned, to the float edge - not laspy's scale-rounded edge.

        laspy rounds the box edges to the file's coordinate scale, so a box edge between grid lines would
        otherwise pull in a whole row up to half a scale unit outside it.
        """
        bbox = (154994.053, 462996.640, 155021.535, 463049.997)

        arrays = copc.read_window(sample_copc, bbox=bbox)

        xmin, ymin, xmax, ymax = bbox
        assert arrays["x"].size > 0
        assert arrays["x"].min() >= xmin and arrays["x"].max() <= xmax
        assert arrays["y"].min() >= ymin and arrays["y"].max() <= ymax


class TestErrorTranslation:
    """`_reader` must only relabel a genuine "not a COPC" error, never a backend or corruption failure (H2)."""

    def test_a_missing_lazrs_backend_is_not_reported_as_no_octree(self, sample_copc, monkeypatch):
        """Without lazrs, the file is still COPC - the backend is missing, so say that, don't misdirect.

        laspy raises `LazError` ("COPC support requires the 'lazrs' backend"), a `LaspyException` subclass. The
        old blanket translation turned it into "carries no COPC octree ... read it whole", a fallback that also
        fails because no lazrs means no LAZ at all.
        """
        monkeypatch.setattr(copc.laspy.copc, "lazrs", None)

        with pytest.raises(copc.laspy.errors.LazError, match="lazrs"):
            copc.read_window(sample_copc, bbox=CORNER_BBOX)

    def test_is_copc_does_not_swallow_a_backend_error(self, sample_copc, monkeypatch):
        """`is_copc` is a predicate; a missing backend must surface, not return a confidently wrong False."""
        monkeypatch.setattr(copc.laspy.copc, "lazrs", None)

        with pytest.raises(copc.laspy.errors.LazError):
            copc.is_copc(sample_copc)

    def test_a_truncated_file_is_not_reported_as_no_octree(self, sample_copc):
        """A structurally broken file re-raises laspy's read error unchanged, not a misdirecting ValueError.

        The old blanket translation turned "File is to small to be a valid LAS" into "carries no COPC octree ...
        read it whole", which would also fail. The structural error must surface as itself.
        """
        truncated = io.BytesIO(sample_copc.read_bytes()[:200])

        with pytest.raises(copc.laspy.errors.LaspyException) as excinfo:
            copc.read_window(truncated, bbox=CORNER_BBOX)

        assert "no COPC octree" not in str(excinfo.value), f"misdirecting message: {excinfo.value}"

    def test_a_genuine_non_copc_still_says_so(self, sample_laz):
        """The one case the translation is for: a valid LAZ with no octree keeps its helpful message."""
        with pytest.raises(ValueError, match="carries no COPC octree"):
            copc.read_window(sample_laz, bbox=CORNER_BBOX)
