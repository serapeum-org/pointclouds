"""Core operations on a cloud: filter, decimate, clip, merge and to_grid.

Everything here runs on the GDAL-free core - numpy and scipy alone - and leans on the DoD of issues #5 and #7:
every operation returns a new cloud with all arrays subset in step, decimation is reproducible, merge rejects a
CRS mix, and to_grid reproduces a hand-checked surface under PDAL's aggregate names.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointclouds import PointCloud


@pytest.fixture
def attributed_cloud() -> PointCloud:
    """A four-point cloud carrying intensity, classification and return number, for alignment checks."""
    return PointCloud(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([0.0, 0.0, 0.0, 0.0]),
        z=np.array([0.0, 1.0, 2.0, 3.0]),
        intensity=np.array([100, 200, 300, 400], dtype=np.uint16),
        classification=np.array([2, 5, 2, 6], dtype=np.uint8),
        return_number=np.array([1, 1, 2, 2], dtype=np.uint8),
        crs="EPSG:28992",
    )


class TestFilter:
    """Subsetting a cloud by attribute membership and extent."""

    def test_scalar_classification_keeps_matching_points_in_step(self, attributed_cloud):
        """Every array is subset together, so attributes stay aligned with the kept coordinates."""
        ground = attributed_cloud.filter(classification=2)

        assert ground.x.tolist() == [0.0, 2.0]
        assert ground.intensity.tolist() == [100, 300]
        assert ground.return_number.tolist() == [1, 2]

    def test_an_iterable_criterion_is_a_membership_test(self, attributed_cloud):
        """An iterable keeps points whose value is in the set."""
        kept = attributed_cloud.filter(classification=[2, 6])

        assert kept.classification.tolist() == [2, 2, 6]

    def test_intensity_is_a_membership_criterion_too(self, attributed_cloud):
        """intensity filters by membership, exactly like the other attribute criteria."""
        kept = attributed_cloud.filter(intensity=200)

        assert kept.x.tolist() == [1.0]

    def test_a_malformed_bbox_is_refused(self, attributed_cloud):
        """A bbox whose min exceeds its max is refused rather than silently emptied."""
        with pytest.raises(ValueError, match="min <= max"):
            attributed_cloud.filter(bbox=(2.0, 0.0, 1.0, 1.0))

    def test_criteria_are_anded_together(self, attributed_cloud):
        """Each criterion narrows the result; here class 2 AND return number 2 leaves one point."""
        kept = attributed_cloud.filter(classification=2, return_number=2)

        assert len(kept) == 1
        assert kept.x.tolist() == [2.0]

    def test_bbox_keeps_points_inside_the_extent(self, attributed_cloud):
        """The bbox criterion is a spatial membership test, edges included."""
        kept = attributed_cloud.filter(bbox=(1.0, -1.0, 2.0, 1.0))

        assert kept.x.tolist() == [1.0, 2.0]

    def test_no_criteria_keeps_everything(self, attributed_cloud):
        """A filter with no criterion is a copy of the whole cloud."""
        assert len(attributed_cloud.filter()) == 4

    def test_an_empty_result_is_a_valid_empty_cloud(self, attributed_cloud):
        """A criterion nothing matches yields an empty cloud, not an error."""
        kept = attributed_cloud.filter(classification=99)

        assert len(kept) == 0
        assert kept.intensity.size == 0

    def test_filtering_on_an_absent_attribute_is_refused(self):
        """A cloud with no classification cannot be filtered on it, and says so."""
        cloud = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))

        with pytest.raises(ValueError, match="carries no classification"):
            cloud.filter(classification=2)

    def test_the_original_is_untouched(self, attributed_cloud):
        """Filtering never mutates in place."""
        attributed_cloud.filter(classification=2)

        assert len(attributed_cloud) == 4


class TestDecimate:
    """Thinning a cloud, randomly or by voxel."""

    def test_random_target_is_reproducible_under_a_seed(self):
        """The same seed selects the same points."""
        cloud = PointCloud(x=np.arange(10.0), y=np.zeros(10), z=np.arange(10.0))

        first = cloud.decimate(target=4, seed=7)
        second = cloud.decimate(target=4, seed=7)

        assert first.x.tolist() == second.x.tolist()
        assert len(first) == 4

    def test_random_never_returns_more_than_the_cloud_holds(self):
        """A target larger than the cloud is clamped, not padded."""
        cloud = PointCloud(x=np.arange(5.0), y=np.zeros(5), z=np.zeros(5))

        assert len(cloud.decimate(target=99, seed=0)) == 5

    def test_random_fraction_keeps_that_share(self):
        """A fraction is rounded to a point count."""
        cloud = PointCloud(x=np.arange(10.0), y=np.zeros(10), z=np.zeros(10))

        assert len(cloud.decimate(fraction=0.3, seed=0)) == 3

    def test_random_needs_exactly_one_of_target_or_fraction(self):
        """Giving both, or neither, is ambiguous and refused."""
        cloud = PointCloud(x=np.arange(4.0), y=np.zeros(4), z=np.zeros(4))

        with pytest.raises(ValueError, match="exactly one of target or fraction"):
            cloud.decimate(target=2, fraction=0.5)

    def test_a_negative_target_is_refused(self):
        """A target cannot be negative."""
        cloud = PointCloud(x=np.arange(4.0), y=np.zeros(4), z=np.zeros(4))

        with pytest.raises(ValueError, match="target must be non-negative"):
            cloud.decimate(target=-1)

    def test_a_fraction_out_of_range_is_refused(self):
        """A fraction must fall in (0, 1]."""
        cloud = PointCloud(x=np.arange(4.0), y=np.zeros(4), z=np.zeros(4))

        with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
            cloud.decimate(fraction=1.5)

    def test_voxel_thinning_an_empty_cloud_is_empty(self):
        """Voxel thinning a cloud with no points yields an empty cloud, not an error."""
        empty = PointCloud(x=np.empty(0), y=np.empty(0), z=np.empty(0))

        assert len(empty.decimate(method="voxel", voxel_size=1.0)) == 0

    def test_voxel_keeps_one_point_per_cube(self):
        """Points sharing a voxel collapse to the first of them, in point order."""
        cloud = PointCloud(x=np.array([0.0, 0.5, 1.0, 1.5]), y=np.zeros(4), z=np.zeros(4))

        thinned = cloud.decimate(method="voxel", voxel_size=1.0)

        assert thinned.x.tolist() == [0.0, 1.0]

    def test_voxel_needs_a_positive_size(self):
        """Voxel thinning without a usable cube size is refused."""
        cloud = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))

        with pytest.raises(ValueError, match="positive voxel_size"):
            cloud.decimate(method="voxel")

    def test_an_unknown_method_is_refused(self):
        """Only random and voxel exist."""
        cloud = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))

        with pytest.raises(ValueError, match="unknown decimation method"):
            cloud.decimate(method="poisson")


class TestClip:
    """Keeping the points inside an extent or a polygon."""

    def test_bbox_clip_keeps_inside_points(self):
        """The bbox form is the numpy-only fast path."""
        cloud = PointCloud(x=np.array([0.0, 5.0, 9.0]), y=np.zeros(3), z=np.zeros(3))

        assert cloud.clip(bbox=(4.0, -1.0, 10.0, 1.0)).x.tolist() == [5.0, 9.0]

    def test_polygon_clip_keeps_inside_points(self):
        """A single ring selects by even-odd ray casting, no shapely needed."""
        square = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])
        cloud = PointCloud(x=np.array([2.0, 0.0, 2.0]), y=np.array([2.0, 0.0, 0.0]), z=np.zeros(3))

        kept = cloud.clip(polygon=square)

        assert kept.x.tolist() == [2.0] and kept.y.tolist() == [2.0]

    def test_a_concave_polygon_is_handled(self):
        """Ray casting is exact for a concave ring, not just a convex one."""
        arrow = np.array([[0.0, 0.0], [4.0, 2.0], [0.0, 4.0], [1.0, 2.0]])
        cloud = PointCloud(x=np.array([0.5, 3.0]), y=np.array([2.0, 2.0]), z=np.zeros(2))

        # (0.5, 2.0) sits in the notch (outside); (3.0, 2.0) is inside the body.
        assert cloud.clip(polygon=arrow).x.tolist() == [3.0]

    def test_exactly_one_region_is_required(self):
        """Neither, or both, is refused."""
        cloud = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))

        with pytest.raises(ValueError, match="exactly one of bbox or polygon"):
            cloud.clip()

    def test_a_malformed_polygon_is_refused(self):
        """A ring needs at least three (x, y) vertices."""
        cloud = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))

        with pytest.raises(ValueError, match="at least 3 vertices"):
            cloud.clip(polygon=np.array([[0.0, 0.0], [1.0, 1.0]]))


class TestMerge:
    """Concatenating clouds into one."""

    def test_merge_concatenates_in_order(self):
        """Every point from every input survives, in input order."""
        a = PointCloud(x=np.zeros(2), y=np.zeros(2), z=np.zeros(2))
        b = PointCloud(x=np.ones(3), y=np.zeros(3), z=np.zeros(3))

        merged = PointCloud.merge([a, b])

        assert merged.x.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]

    def test_merge_keeps_attributes_aligned(self, attributed_cloud):
        """Attributes present on every cloud are concatenated alongside the coordinates."""
        merged = PointCloud.merge([attributed_cloud, attributed_cloud])

        assert merged.classification.tolist() == [2, 5, 2, 6, 2, 5, 2, 6]
        assert len(merged.intensity) == 8

    def test_merge_refuses_conflicting_crs(self):
        """A CRS mix is a conflict, not something to guess through."""
        a = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), crs="EPSG:28992")
        b = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), crs="EPSG:4326")

        with pytest.raises(ValueError, match="conflicting CRSs"):
            PointCloud.merge([a, b])

    def test_merge_refuses_a_partial_attribute(self):
        """An attribute on some clouds but not others would misalign, so it is refused."""
        with_class = PointCloud(
            x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), classification=np.array([2], dtype=np.uint8)
        )
        without = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))

        with pytest.raises(ValueError, match="present on some clouds but not others"):
            PointCloud.merge([with_class, without])

    def test_merge_of_one_cloud_is_a_copy(self, attributed_cloud):
        """Merging a single cloud is well defined - it is that cloud's points."""
        assert len(PointCloud.merge([attributed_cloud])) == 4

    def test_merge_of_nothing_is_refused(self):
        """There is no cloud to build from an empty sequence."""
        with pytest.raises(ValueError, match="at least one cloud"):
            PointCloud.merge([])

    def test_merge_treats_a_missing_crs_as_a_conflict(self):
        """A cloud with a CRS and one without do not agree, so the merge is refused, not guessed."""
        known = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), crs="EPSG:28992")
        unknown = PointCloud(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))

        with pytest.raises(ValueError, match="conflicting CRSs"):
            PointCloud.merge([known, unknown])


@pytest.fixture
def grid_cloud() -> PointCloud:
    """Four points over a 2x2 grid of unit cells, with the SE cell left empty.

    Layout over bounds (0, 0, 2, 2): the NW cell holds z=10 and z=20, the NE cell z=30, the SW cell z=40.
    """
    return PointCloud(
        x=np.array([0.5, 0.5, 1.5, 0.5]),
        y=np.array([1.5, 1.5, 1.5, 0.5]),
        z=np.array([10.0, 20.0, 30.0, 40.0]),
        crs="EPSG:28992",
    )


class TestToGrid:
    """Gridding a cloud to a raster surface under PDAL's aggregate vocabulary."""

    BOUNDS = (0.0, 0.0, 2.0, 2.0)

    def test_mean_matches_a_hand_checked_surface(self, grid_cloud):
        """The NW cell averages 10 and 20; the empty SE cell is nodata."""
        array, _, _ = grid_cloud.to_grid(cell_size=1.0, aggregate="mean", bounds=self.BOUNDS)

        assert array[0].tolist() == [15.0, 30.0]
        assert array[1, 0] == 40.0
        assert np.isnan(array[1, 1])

    def test_transform_and_crs_round_trip(self, grid_cloud):
        """The transform is a GDAL-style affine 6-tuple and the CRS is the cloud's own."""
        _, transform, crs = grid_cloud.to_grid(cell_size=1.0, bounds=self.BOUNDS)

        assert transform == (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
        assert crs.to_epsg() == 28992

    @pytest.mark.parametrize(
        ("aggregate", "nw_cell"),
        [("min", 10.0), ("max", 20.0), ("mean", 15.0), ("count", 2.0), ("stdev", 5.0), ("p50", 15.0)],
    )
    def test_each_aggregate_reduces_the_nw_cell_correctly(self, grid_cloud, aggregate, nw_cell):
        """Each PDAL name reduces the two NW-cell points to its known value."""
        array, _, _ = grid_cloud.to_grid(cell_size=1.0, aggregate=aggregate, bounds=self.BOUNDS)

        assert array[0, 0] == pytest.approx(nw_cell)

    def test_count_fills_empty_cells_with_zero(self, grid_cloud):
        """An empty cell honestly holds zero points, so count writes 0 there rather than nodata."""
        array, _, _ = grid_cloud.to_grid(cell_size=1.0, aggregate="count", bounds=self.BOUNDS)

        assert array[1, 1] == 0.0

    def test_nodata_is_configurable(self, grid_cloud):
        """The empty-cell fill is whatever the caller asks for."""
        array, _, _ = grid_cloud.to_grid(cell_size=1.0, aggregate="mean", bounds=self.BOUNDS, nodata=-9999.0)

        assert array[1, 1] == -9999.0

    def test_idw_weights_the_nearer_point_more(self):
        """With one point near a cell centre and one far, idw leans to the near value."""
        cloud = PointCloud(
            x=np.array([0.5, 0.9]), y=np.array([0.5, 0.5]), z=np.array([0.0, 100.0])
        )
        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="idw", bounds=(0.0, 0.0, 1.0, 1.0))

        # The point at (0.5, 0.5) sits on the centre, so its zero dominates the weighting.
        assert array[0, 0] < 1.0

    def test_default_window_is_bin(self):
        """A single point contributes to exactly one cell by default (bin mode)."""
        cloud = PointCloud(x=np.array([0.5]), y=np.array([0.5]), z=np.array([5.0]))

        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="count", bounds=(0.0, 0.0, 2.0, 2.0))

        assert array.sum() == 1.0

    def test_radius_window_spreads_a_point_to_nearby_cells(self):
        """In radius mode the same point reaches every cell whose centre is within sqrt(2)."""
        cloud = PointCloud(x=np.array([0.5]), y=np.array([0.5]), z=np.array([5.0]))

        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="count", mode="radius", bounds=(0.0, 0.0, 2.0, 2.0))

        assert array.sum() == 4.0

    def test_bounds_default_to_the_cloud_extent(self, grid_cloud):
        """Omitting bounds grids the cloud's own extent."""
        array, transform, _ = grid_cloud.to_grid(cell_size=1.0)

        assert array.shape[0] >= 1 and array.shape[1] >= 1
        assert transform[1] == 1.0

    def test_a_degenerate_bounds_is_refused(self, grid_cloud):
        """A zero-area extent cannot be gridded."""
        with pytest.raises(ValueError, match="positive-area bounds"):
            grid_cloud.to_grid(cell_size=1.0, bounds=(0.0, 0.0, 0.0, 2.0))

    def test_a_non_positive_cell_size_is_refused(self, grid_cloud):
        """The pixel edge must be positive."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            grid_cloud.to_grid(cell_size=0.0, bounds=self.BOUNDS)

    def test_an_unknown_aggregate_is_refused(self, grid_cloud):
        """A name outside the PDAL vocabulary is refused with the list of valid ones."""
        with pytest.raises(ValueError, match="unknown aggregate"):
            grid_cloud.to_grid(cell_size=1.0, aggregate="median", bounds=self.BOUNDS)

    def test_an_unknown_mode_is_refused(self, grid_cloud):
        """Only bin and radius exist."""
        with pytest.raises(ValueError, match="unknown mode"):
            grid_cloud.to_grid(cell_size=1.0, mode="kriging", bounds=self.BOUNDS)

    def test_a_percentile_out_of_range_is_refused(self, grid_cloud):
        """A p<n> whose n is above 100 is refused."""
        with pytest.raises(ValueError, match="percentile out of range"):
            grid_cloud.to_grid(cell_size=1.0, aggregate="p150", bounds=self.BOUNDS)

    def test_points_on_the_max_edge_land_in_the_last_cell(self):
        """A point exactly on xmax / ymin is kept in the last cell, not dropped (default-bounds data loss)."""
        cloud = PointCloud(x=np.array([0.0, 2.0]), y=np.array([2.0, 0.0]), z=np.array([10.0, 20.0]))

        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="count")

        assert array.sum() == 2.0
        assert array[0, 0] == 1.0 and array[1, 1] == 1.0

    def test_percentile_of_an_empty_grid_is_all_nodata(self):
        """A percentile grid with no points in the extent is entirely nodata, not an error."""
        cloud = PointCloud(x=np.array([9.0]), y=np.array([9.0]), z=np.array([1.0]))

        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="p50", bounds=(0.0, 0.0, 2.0, 2.0))

        assert np.isnan(array).all()

    def test_points_outside_the_bounds_are_dropped(self):
        """In bin mode a point beyond the grid extent contributes to no cell."""
        cloud = PointCloud(x=np.array([0.5, 9.0]), y=np.array([0.5, 9.0]), z=np.array([1.0, 2.0]))

        array, _, _ = cloud.to_grid(cell_size=1.0, aggregate="count", bounds=(0.0, 0.0, 2.0, 2.0))

        assert array.sum() == 1.0
