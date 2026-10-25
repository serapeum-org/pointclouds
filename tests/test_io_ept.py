"""The EPT engine: manifest reads that touch no node data, and windowed reads that really do prune the octree.

These run against two committed EPT datasets with a known two-level octree (see
`examples/data/make_ept_sample.py`): a root node subsampling the whole extent, then four populated depth-1
octants. The `any_ept` fixture runs every shared assertion against both the `laszip` and `binary` encodings, so
the two decode paths are proven to agree, not just to run. The datasets hold the same 1600 points as the COPC
fixture, so a corner box returns the same 121 points - the windowed contract is identical across formats.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pyproj import CRS

from pointclouds import PointCloud
from pointclouds.io import ept

EXPECTED_COUNT = 1600
EXPECTED_BOUNDS = (155000.0, 463000.0, 155078.0, 463078.0)
EXPECTED_SPAN = 128
ROOT_COUNT = 100
CORNER_BBOX = (155000.0, 463000.0, 155020.0, 463020.0)
CORNER_COUNT = 121


class TestHeader:
    """`read_header` answers from `ept.json` alone."""

    def test_it_reports_count_bounds_crs_span_and_type(self, any_ept):
        """The manifest carries the count, conforming bounds, CRS, span and data type."""
        info = ept.read_header(any_ept)

        assert info.count == EXPECTED_COUNT
        assert info.bounds == pytest.approx(EXPECTED_BOUNDS)
        assert isinstance(info.crs, CRS)
        assert info.crs.to_epsg() == 28992
        assert info.span == EXPECTED_SPAN
        assert info.version == "1.0.0"
        assert info.data_type in {"laszip", "binary"}

    def test_it_accepts_the_manifest_path_or_the_directory(self, sample_ept):
        """A source may name the directory or the ept.json inside it; both resolve to the same dataset."""
        by_dir = ept.read_header(sample_ept)
        by_file = ept.read_header(sample_ept / "ept.json")

        assert by_dir == by_file

    def test_it_touches_no_node_data(self, any_ept, monkeypatch):
        """A header read must not fetch a node - the header-first contract every engine keeps.

        Enforced by making the node read explode: if `read_header` ever reaches for node bytes, this fails.
        """

        def explode(self, relative):
            if relative.startswith("ept-data"):
                raise AssertionError("read_header must not read node data")
            return original(self, relative)

        original = ept._Store.read_bytes
        monkeypatch.setattr(ept._Store, "read_bytes", explode)

        assert ept.read_header(any_ept).count == EXPECTED_COUNT


class TestIsEpt:
    """`is_ept` recognises a dataset without reading node data."""

    def test_a_directory_with_a_manifest_is_ept(self, sample_ept):
        """A directory holding an ept.json is an EPT dataset."""
        assert ept.is_ept(sample_ept) is True

    def test_the_manifest_path_is_ept(self, sample_ept):
        """Naming the ept.json itself also counts."""
        assert ept.is_ept(sample_ept / "ept.json") is True

    def test_a_plain_directory_is_not(self, tmp_path):
        """A directory with no ept.json is not EPT, and the answer is False, not an error."""
        assert ept.is_ept(tmp_path) is False

    def test_a_laz_file_is_not_ept(self, sample_copc):
        """A COPC/LAZ file is not an EPT dataset - so the container can dispatch on this."""
        assert ept.is_ept(sample_copc) is False


class TestBoundingBoxWindow:
    """`bbox=` reads a sub-box by pruning the octree to the nodes that overlap it."""

    def test_no_window_reads_the_whole_dataset(self, any_ept):
        """With no argument the window spans the dataset."""
        assert ept.read_window(any_ept)["x"].size == EXPECTED_COUNT

    def test_only_points_inside_the_box_come_back(self, any_ept):
        """Points are clipped exactly to the box, and both encodings return the same count as COPC (121)."""
        arrays = ept.read_window(any_ept, bbox=CORNER_BBOX)

        xmin, ymin, xmax, ymax = CORNER_BBOX
        assert arrays["x"].size == CORNER_COUNT
        assert arrays["x"].min() >= xmin and arrays["x"].max() <= xmax
        assert arrays["y"].min() >= ymin and arrays["y"].max() <= ymax

    def test_it_fetches_fewer_nodes_than_the_whole_dataset(self, sample_ept, monkeypatch):
        """The point of the feature: a corner box must fetch fewer node files, not just return fewer points."""
        fetched: list[str] = []
        original = ept._Store.read_bytes

        def spy(self, relative):
            if relative.startswith("ept-data"):
                fetched.append(relative)
            return original(self, relative)

        monkeypatch.setattr(ept._Store, "read_bytes", spy)

        ept.read_window(sample_ept)
        whole = len(fetched)
        fetched.clear()
        ept.read_window(sample_ept, bbox=CORNER_BBOX)
        windowed = len(fetched)

        assert 0 < windowed < whole, f"a corner box fetched {windowed} nodes vs {whole} for the whole dataset"

    def test_a_box_outside_the_data_reads_nothing(self, any_ept):
        """An empty window is empty arrays, not an error or a missing key."""
        arrays = ept.read_window(any_ept, bbox=(0.0, 0.0, 1.0, 1.0))

        assert arrays["x"].size == 0
        assert arrays["classification"].size == 0

    def test_every_canonical_dimension_comes_back(self, any_ept):
        """A window returns the same dimension set, dtypes and alignment as every other reader."""
        arrays = ept.read_window(any_ept, bbox=CORNER_BBOX)

        assert set(arrays) == {"x", "y", "z", "intensity", "classification", "return_number"}
        assert all(values.shape == (CORNER_COUNT,) for values in arrays.values())
        assert arrays["x"].dtype == np.float64
        assert arrays["intensity"].dtype == np.uint16
        assert arrays["classification"].dtype == np.uint8

    def test_a_dimension_subset_is_honoured(self, any_ept):
        """Only the requested dimensions are returned, case-insensitively."""
        arrays = ept.read_window(any_ept, bbox=CORNER_BBOX, dimensions=("X", "Z"))

        assert set(arrays) == {"x", "z"}
        assert arrays["z"].shape == (CORNER_COUNT,)


class TestResolutionAndDepth:
    """`resolution=` and `depth=` are the level-of-detail half of a window."""

    def test_depth_zero_reads_the_root_alone(self, any_ept):
        """`depth=0` reads only the root node - a thinned view over the whole extent."""
        assert ept.read_window(any_ept, depth=0)["x"].size == ROOT_COUNT

    def test_a_coarse_resolution_reads_the_root_alone(self, any_ept):
        """A resolution coarser than the root spacing stops at depth 0."""
        assert ept.read_window(any_ept, resolution=1e6)["x"].size == ROOT_COUNT

    def test_a_fine_resolution_reads_everything(self, any_ept):
        """A resolution finer than the deepest node cannot prune, so the whole dataset comes back."""
        assert ept.read_window(any_ept, resolution=0.01)["x"].size == EXPECTED_COUNT

    def test_a_coarse_read_spans_the_full_extent(self, any_ept):
        """The root is a subsample over the whole area, not a corner of it - the LOD claim."""
        arrays = ept.read_window(any_ept, depth=0)

        xmin, ymin, xmax, ymax = EXPECTED_BOUNDS
        covered_x = (arrays["x"].max() - arrays["x"].min()) / (xmax - xmin)
        assert covered_x > 0.8, f"the coarse read spans only {covered_x:.0%} of the x extent"

    def test_a_coarse_read_fetches_one_node(self, sample_ept, monkeypatch):
        """depth=0 fetches exactly the root data file."""
        fetched: list[str] = []
        original = ept._Store.read_bytes

        def spy(self, relative):
            if relative.startswith("ept-data"):
                fetched.append(relative)
            return original(self, relative)

        monkeypatch.setattr(ept._Store, "read_bytes", spy)
        ept.read_window(sample_ept, depth=0)

        assert fetched == ["ept-data/0-0-0-0.laz"]


class TestEncodingsAgree:
    """The laszip and binary datasets must decode to identical arrays - that is the binary path's proof."""

    def test_full_reads_match_point_for_point(self, sample_ept, sample_ept_binary):
        """Same points, same attributes, whichever encoding - sorted to ignore node/order differences."""
        laz = ept.read_window(sample_ept)
        binary = ept.read_window(sample_ept_binary)

        order_laz = np.lexsort((laz["y"], laz["x"]))
        order_bin = np.lexsort((binary["y"], binary["x"]))
        for name in ("x", "y", "z"):
            assert laz[name][order_laz] == pytest.approx(binary[name][order_bin], abs=0.01), name
        for name in ("classification", "intensity", "return_number"):
            assert np.array_equal(laz[name][order_laz], binary[name][order_bin]), name


class TestRejections:
    """What the engine refuses, and whether the message helps."""

    def test_a_missing_manifest_names_the_path(self, tmp_path):
        """A directory with no ept.json raises FileNotFoundError naming the manifest."""
        with pytest.raises(FileNotFoundError, match="ept.json"):
            ept.read_window(tmp_path)

    def test_a_non_finite_bbox_is_refused(self, sample_ept):
        """`nan`/`inf` corners are rejected before any node is fetched."""
        with pytest.raises(ValueError, match="finite"):
            ept.read_window(sample_ept, bbox=(float("nan"), 0.0, 1.0, 1.0))

    def test_an_inverted_bbox_is_refused(self, sample_ept):
        """Swapped corners are an error, not a silent empty result."""
        with pytest.raises(ValueError, match="inverted"):
            ept.read_window(sample_ept, bbox=(10.0, 10.0, 0.0, 20.0))

    @pytest.mark.parametrize("resolution", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_bad_resolution_is_refused(self, sample_ept, resolution):
        """Resolution must be positive and finite."""
        with pytest.raises(ValueError, match="resolution must be"):
            ept.read_window(sample_ept, resolution=resolution)

    def test_a_negative_depth_is_refused(self, sample_ept):
        """Octree depth starts at 0 at the root."""
        with pytest.raises(ValueError, match="depth must be"):
            ept.read_window(sample_ept, depth=-1)

    def test_resolution_and_depth_are_mutually_exclusive(self, sample_ept):
        """They are two spellings of the same limit; accepting both would hide which won."""
        with pytest.raises(ValueError, match="not both"):
            ept.read_window(sample_ept, resolution=1.0, depth=1)

    def test_an_unknown_dimension_is_refused(self, sample_ept):
        """An unknown dimension fails loudly, listing what the engine reads."""
        with pytest.raises(KeyError, match="unknown dimension"):
            ept.read_window(sample_ept, dimensions=("nosuchdim",))

    def test_zstandard_is_refused_with_a_clear_message(self, tmp_path, sample_ept):
        """A zstandard dataset is recognised but not decoded; the error says what to do."""
        manifest = json.loads((sample_ept / "ept.json").read_text())
        manifest["dataType"] = "zstandard"
        (tmp_path / "ept-hierarchy").mkdir(parents=True)
        (tmp_path / "ept-data").mkdir()
        (tmp_path / "ept.json").write_text(json.dumps(manifest))
        (tmp_path / "ept-hierarchy" / "0-0-0-0.json").write_text(
            ((sample_ept / "ept-hierarchy" / "0-0-0-0.json").read_text())
        )
        (tmp_path / "ept-data" / "0-0-0-0.bin").write_bytes(b"")

        with pytest.raises(NotImplementedError, match="zstandard"):
            ept.read_window(tmp_path, depth=0)


class TestRemoteNeedsCloud:
    """A remote URL needs the [cloud] extra; without it the error names the extra (not an ImportError)."""

    def test_a_url_without_fsspec_names_the_extra(self, monkeypatch):
        """When fsspec is absent, a URL source raises a ModuleNotFoundError naming pointclouds[cloud]."""
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "fsspec":
                raise ModuleNotFoundError("No module named 'fsspec'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        with pytest.raises(ModuleNotFoundError, match=r"pointclouds\[cloud\]"):
            ept.read_header("s3://bucket/dataset/ept.json")


class TestPointCloudWindow:
    """The container surface: `PointCloud.read_window` / `read_file` dispatch to the EPT engine."""

    def test_a_window_is_a_cloud_of_only_those_points(self, any_ept):
        """The returned cloud holds the window, eager, with the dataset's CRS."""
        cloud = PointCloud.read_window(any_ept, bbox=CORNER_BBOX)

        assert cloud.count == CORNER_COUNT
        assert len(cloud) == CORNER_COUNT
        assert cloud.loaded is True
        assert cloud.crs.to_epsg() == 28992

    def test_count_and_bounds_describe_the_window(self, any_ept):
        """A windowed cloud carries no header, so it reports the window, not the dataset's 1600 points."""
        cloud = PointCloud.read_window(any_ept, bbox=CORNER_BBOX)

        assert cloud.count != EXPECTED_COUNT
        assert cloud.bounds == pytest.approx(CORNER_BBOX)

    def test_read_file_dispatches_to_ept(self, sample_ept):
        """`read_file(bbox=)` on an EPT directory reaches the EPT engine, same as read_window."""
        via_read_file = PointCloud.read_file(sample_ept, bbox=CORNER_BBOX)
        direct = PointCloud.read_window(sample_ept, bbox=CORNER_BBOX)

        assert via_read_file.count == direct.count == CORNER_COUNT

    def test_level_maps_to_depth(self, sample_ept):
        """An int `level` on an EPT source is read as a maximum depth."""
        assert PointCloud.read_window(sample_ept, level=0).count == ROOT_COUNT

    def test_a_range_level_is_refused_for_ept(self, sample_ept):
        """EPT has no exact-level read, so a range level is refused rather than guessed."""
        with pytest.raises(ValueError, match="maximum depth"):
            PointCloud.read_window(sample_ept, level=range(0, 2))

    def test_a_dimension_subset_builds_a_cloud(self, any_ept):
        """x/y/z are forced in, so a subset request still yields a usable cloud."""
        cloud = PointCloud.read_window(any_ept, bbox=CORNER_BBOX, dimensions=("classification",))

        assert len(cloud) == CORNER_COUNT
        assert cloud.classification.shape == (CORNER_COUNT,)


class TestBinaryDecoderUnits:
    """Direct unit tests for the binary decoder's edges, which the committed schema does not exercise."""

    def test_an_unsupported_schema_type_is_refused(self):
        """A {type, size} the EPT spec does not define is rejected, naming the dimension."""
        with pytest.raises(ValueError, match="unsupported EPT schema type"):
            ept._binary_dtype([{"name": "X", "type": "signed", "size": 3}])

    def test_a_canonical_dimension_absent_from_the_schema_comes_back_empty(self):
        """A binary node whose schema omits a requested dimension yields a zero-length array, not a KeyError."""
        schema = [{"name": "X", "type": "signed", "size": 4, "scale": 1.0, "offset": 0.0}]
        data = np.array([(5,)], dtype=np.dtype([("X", "<i4")])).tobytes()

        arrays = ept._decode_binary(data, schema, ("x", "intensity"))

        assert arrays["x"].tolist() == [5.0]
        assert arrays["intensity"].size == 0
        assert arrays["intensity"].dtype == np.uint16

    def test_a_schema_dimension_this_package_does_not_map_is_ignored(self):
        """An EPT dimension with no canonical home (GpsTime) is simply not returned, not an error."""
        schema = [
            {"name": "X", "type": "signed", "size": 4, "scale": 1.0, "offset": 0.0},
            {"name": "GpsTime", "type": "float", "size": 8},
        ]
        record = np.array([(3, 123.5)], dtype=np.dtype([("X", "<i4"), ("GpsTime", "<f8")]))

        arrays = ept._decode_binary(record.tobytes(), schema, ("x",))

        assert set(arrays) == {"x"}
        assert arrays["x"].tolist() == [3.0]


class TestUnknownDataType:
    """An unrecognised dataType is an error, not a wrong decode."""

    def test_decode_refuses_an_unknown_type(self):
        """`_decode` names the three types it knows when handed something else."""
        with pytest.raises(ValueError, match="unknown EPT dataType"):
            ept._decode(b"", "parquet", [], ("x",))


class _FakeFs:
    """A minimal fsspec-like filesystem backed by a local directory, for exercising the remote path."""

    def open(self, path, mode="rb"):
        return open(path, mode)

    def exists(self, path):
        from pathlib import Path as _P

        return _P(path).exists()


class TestRemotePathViaFakeFsspec:
    """The remote branch, exercised by injecting a fake `fsspec` that reads from the local fixture."""

    @pytest.fixture
    def fake_fsspec(self, sample_ept, monkeypatch):
        """Install a fake `fsspec` whose get_fs_token_paths maps any URL to the local sample-ept directory."""
        import sys
        import types

        module = types.ModuleType("fsspec")
        module.get_fs_token_paths = lambda url: (_FakeFs(), None, [str(sample_ept)])
        monkeypatch.setitem(sys.modules, "fsspec", module)
        return module

    def test_read_header_over_a_url(self, fake_fsspec):
        """A URL source parses the manifest through the fake filesystem."""
        info = ept.read_header("s3://bucket/sample-ept")

        assert info.count == EXPECTED_COUNT
        assert info.data_type == "laszip"

    def test_read_window_over_a_url(self, fake_fsspec):
        """A windowed read fetches and decodes nodes through the fake filesystem."""
        arrays = ept.read_window("s3://bucket/sample-ept/ept.json", bbox=CORNER_BBOX)

        assert arrays["x"].size == CORNER_COUNT

    def test_is_ept_over_a_url(self, fake_fsspec):
        """`is_ept` checks for the manifest through the fake filesystem."""
        assert ept.is_ept("s3://bucket/sample-ept") is True


def _write_binary_ept(root, hierarchy_pages, nodes, *, srs=None):
    """Write a minimal binary EPT under `root` from explicit hierarchy pages and node point lists.

    Args:
        root: the dataset directory (created).
        hierarchy_pages: mapping of page key -> {node key: count} to write under ept-hierarchy/.
        nodes: mapping of node key -> list of (x, y, z) tuples to pack into ept-data/<key>.bin.
        srs: the manifest srs block, or None.
    """
    (root / "ept-data").mkdir(parents=True)
    (root / "ept-hierarchy").mkdir()
    schema = [
        {"name": "X", "type": "signed", "size": 4, "scale": 1.0, "offset": 0.0},
        {"name": "Y", "type": "signed", "size": 4, "scale": 1.0, "offset": 0.0},
        {"name": "Z", "type": "signed", "size": 4, "scale": 1.0, "offset": 0.0},
    ]
    dtype = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4")])
    for key, points in nodes.items():
        rec = np.array([tuple(int(v) for v in p) for p in points], dtype=dtype)
        (root / "ept-data" / f"{key}.bin").write_bytes(rec.tobytes())
    for page_key, page in hierarchy_pages.items():
        (root / "ept-hierarchy" / f"{page_key}.json").write_text(json.dumps(page))
    manifest = {
        "bounds": [0, 0, 0, 8, 8, 8],
        "boundsConforming": [0, 0, 0, 8, 8, 8],
        "dataType": "binary",
        "hierarchyType": "json",
        "points": sum(len(p) for p in nodes.values()),
        "schema": schema,
        "span": 2,
    }
    if srs is not None:
        manifest["srs"] = srs
    (root / "ept.json").write_text(json.dumps(manifest))


class TestNestedHierarchy:
    """A real EPT pages its hierarchy; a `-1` entry defers a subtree to its own page, loaded on demand."""

    def test_a_deferred_subtree_page_is_loaded_and_its_points_returned(self, tmp_path):
        """The root page marks 1-0-0-0 as -1 (its info is in 1-0-0-0.json); the walk must load that page."""
        root = tmp_path / "nested"
        _write_binary_ept(
            root,
            hierarchy_pages={
                "0-0-0-0": {"0-0-0-0": 1, "1-0-0-0": -1},
                "1-0-0-0": {"1-0-0-0": 2},
            },
            nodes={"0-0-0-0": [(1, 1, 1)], "1-0-0-0": [(2, 2, 2), (3, 3, 3)]},
        )

        arrays = ept.read_window(root)

        assert arrays["x"].size == 3, "the deferred node's 2 points plus the root's 1 should all be read"
        assert sorted(arrays["x"].tolist()) == [1.0, 2.0, 3.0]


class TestSrsVariants:
    """The manifest's srs block may give WKT, an authority code, or nothing."""

    def test_a_wkt_srs_is_parsed(self, tmp_path):
        """A WKT srs is read into a pyproj CRS."""
        root = tmp_path / "wkt"
        _write_binary_ept(root, {"0-0-0-0": {"0-0-0-0": 1}}, {"0-0-0-0": [(1, 1, 1)]},
                          srs={"wkt": CRS.from_epsg(28992).to_wkt()})

        assert ept.read_header(root).crs.to_epsg() == 28992

    def test_a_missing_srs_is_none(self, tmp_path):
        """No srs block means a CRS of None, not an error."""
        root = tmp_path / "nosrs"
        _write_binary_ept(root, {"0-0-0-0": {"0-0-0-0": 1}}, {"0-0-0-0": [(1, 1, 1)]}, srs=None)

        assert ept.read_header(root).crs is None


class TestMoreRejections:
    """Remaining input guards."""

    def test_a_bbox_needs_four_values(self, sample_ept):
        """Three values is caught before any node is read."""
        with pytest.raises(ValueError, match="bbox must be"):
            ept.read_window(sample_ept, bbox=(1.0, 2.0, 3.0))

    def test_a_local_store_exists_check(self, sample_ept):
        """The store's own existence check works on the local filesystem."""
        store = ept._store(sample_ept)

        assert store.exists("ept.json") is True
        assert store.exists("ept-data/nope.bin") is False

    def test_is_ept_on_a_url_without_fsspec_is_false(self, monkeypatch):
        """Without the cloud extra, is_ept on a URL answers False rather than raising."""
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "fsspec":
                raise ModuleNotFoundError("No module named 'fsspec'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        assert ept.is_ept("s3://bucket/dataset") is False
