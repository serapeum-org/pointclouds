"""The record contract both readers share: which dimensions exist, and what an absent one comes back as.

`pointclouds.io._records` is the seam that keeps the whole-file and windowed engines from drifting, so it is
worth testing on its own and not only through the two callers. The fakes below stand in for laspy record
sources, which is the point: the contract is about shapes and dtypes, not about laspy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointclouds.io import _records

ALL_DIMENSIONS = {"x", "y", "z", "intensity", "classification", "return_number"}


class _PointFormat:
    """A point format declaring exactly the dimension names it is given."""

    def __init__(self, *names: str) -> None:
        """Store the declared names.

        Args:
            names: dimension names the format declares, in file order.
        """
        self.dimension_names = names


class _Source:
    """The minimum record-source surface `_records.extract` touches."""

    def __init__(self, point_format: _PointFormat, **arrays: np.ndarray) -> None:
        """Build a source from a point format and whichever arrays it carries.

        Args:
            point_format: the format declaring the dimension names.
            arrays: per-dimension arrays, set as attributes.
        """
        self.point_format = point_format
        for name, values in arrays.items():
            setattr(self, name, values)


@pytest.fixture
def full_source() -> _Source:
    """A two-point source declaring and carrying every canonical dimension."""
    return _Source(
        _PointFormat("X", "Y", "Z", "intensity", "classification", "return_number"),
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        z=np.array([5.0, 6.0]),
        intensity=np.array([10, 20]),
        classification=np.array([2, 5]),
        return_number=np.array([1, 2]),
    )


class TestResolve:
    """`resolve` normalises a caller's dimension list, or refuses it."""

    def test_none_means_every_canonical_dimension(self):
        """`None` is the "read everything this package maps" request.

        Test scenario:
            resolve(None) returns all canonical names, so neither engine has to spell the list itself.
        """
        assert set(_records.resolve(None)) == ALL_DIMENSIONS, "None must expand to every canonical dimension"

    def test_names_are_lower_cased(self):
        """Dimension names are case-insensitive at the API edge.

        Test scenario:
            ("X", "Intensity") resolves to ("x", "intensity"), because LAS spells dimensions inconsistently.
        """
        assert _records.resolve(("X", "Intensity")) == ("x", "intensity"), "names should be lower-cased"

    def test_the_given_order_is_kept(self):
        """The caller's order survives, so a caller can rely on it when zipping results.

        Test scenario:
            ("Z", "X") stays ("z", "x") rather than being sorted into canonical order.
        """
        assert _records.resolve(("Z", "X")) == ("z", "x"), "resolve must not reorder the request"

    def test_an_empty_request_is_empty_not_everything(self):
        """An explicit empty tuple is not the same request as `None`.

        Test scenario:
            resolve(()) returns (), so `dimensions=()` reads no arrays instead of silently reading all of them.
        """
        assert _records.resolve(()) == (), "an empty request must stay empty"

    @pytest.mark.parametrize(
        "dimensions, unknown",
        [
            (("nosuchdim",), "nosuchdim"),
            (("X", "gps_time"), "gps_time"),
            (("RED",), "red"),
        ],
    )
    def test_an_unknown_dimension_is_refused(self, dimensions, unknown):
        """A dimension this package does not map fails loudly rather than coming back empty.

        Args:
            dimensions: the requested dimension names.
            unknown: the name expected to be reported as unknown.

        Test scenario:
            Names laspy may well declare - gps_time, red - are still not ones PointCloud holds, so asking for
            them is a mistake worth naming, not an empty array.
        """
        with pytest.raises(KeyError, match="unknown dimension") as excinfo:
            _records.resolve(dimensions)

        assert unknown in str(excinfo.value), f"the error should name {unknown!r}, got: {excinfo.value}"

    def test_the_refusal_lists_what_is_readable(self):
        """The error has to say what *is* available, or the caller is left guessing.

        Test scenario:
            The message includes the canonical names.
        """
        with pytest.raises(KeyError) as excinfo:
            _records.resolve(("nope",))

        message = str(excinfo.value)
        assert all(name in message for name in ALL_DIMENSIONS), f"message should list readable dims: {message}"


class TestDeclaredNames:
    """`declared_names` reports a source's own dimension names, lower-cased."""

    def test_it_lower_cases_the_point_format_names(self, full_source):
        """LAS declares "X" where the package says "x".

        Test scenario:
            A format declaring upper-case coordinates reports them lower-cased, so membership tests work.
        """
        assert _records.declared_names(full_source) == ALL_DIMENSIONS, "declared names should be lower-cased"

    def test_a_sparse_format_declares_less(self):
        """A format that carries no attributes declares only its coordinates.

        Test scenario:
            Point format 0-style declaration returns just x/y/z.
        """
        source = _Source(_PointFormat("X", "Y", "Z"))

        assert _records.declared_names(source) == {"x", "y", "z"}, "only the declared names should be reported"


class TestExtract:
    """`extract` turns a record source into the mapping `PointCloud` is built from."""

    def test_every_requested_dimension_is_present_and_typed(self, full_source):
        """The returned mapping is keyed by exactly what was asked for, with the canonical dtypes.

        Test scenario:
            A full request returns six aligned arrays, coordinates as float64 and attributes as their own
            integer types.
        """
        arrays = _records.extract(full_source, _records.resolve(None))

        assert set(arrays) == ALL_DIMENSIONS, f"expected every canonical dimension, got {sorted(arrays)}"
        assert all(values.shape == (2,) for values in arrays.values()), "arrays must stay aligned"
        assert arrays["x"].dtype == np.float64, f"x should be float64, got {arrays['x'].dtype}"
        assert arrays["intensity"].dtype == np.uint16, f"intensity should be uint16, got {arrays['intensity'].dtype}"
        assert arrays["classification"].dtype == np.uint8, "classification should be uint8"
        assert arrays["return_number"].dtype == np.uint8, "return_number should be uint8"

    def test_a_subset_returns_only_that_subset(self, full_source):
        """Asking for two dimensions reads two, which is what makes a partial read cheaper.

        Test scenario:
            ("x", "z") returns exactly those keys.
        """
        arrays = _records.extract(full_source, ("x", "z"))

        assert set(arrays) == {"x", "z"}, f"expected only x and z, got {sorted(arrays)}"

    def test_an_undeclared_dimension_is_empty_not_missing(self):
        """The contract that lets callers index the result unconditionally.

        Test scenario:
            A format without intensity still yields an "intensity" key, as a zero-length uint16 array rather
            than a KeyError or an AttributeError.
        """
        source = _Source(
            _PointFormat("X", "Y", "Z", "classification"),
            x=np.zeros(2),
            y=np.zeros(2),
            z=np.zeros(2),
            classification=np.ones(2, dtype=np.uint8),
        )

        arrays = _records.extract(source, _records.resolve(None))

        assert arrays["intensity"].size == 0, "an undeclared dimension should be empty"
        assert arrays["intensity"].dtype == np.uint16, "an empty array still carries the canonical dtype"
        assert arrays["classification"].shape == (2,), "a declared dimension is read normally"

    def test_coordinates_are_read_even_when_undeclared(self):
        """laspy always exposes scaled x/y/z, whatever the format's name list says.

        Test scenario:
            A format declaring no names at all still yields real coordinates, because treating them as absent
            would make every such file read as empty.
        """
        source = _Source(_PointFormat(), x=np.array([1.0]), y=np.array([2.0]), z=np.array([3.0]))

        arrays = _records.extract(source, ("x", "y", "z"))

        assert arrays["x"].shape == (1,), "coordinates must never be treated as undeclared"
        assert arrays["z"][0] == 3.0, f"expected z == 3.0, got {arrays['z'][0]}"

    def test_an_empty_request_gives_an_empty_mapping(self, full_source):
        """Nothing requested means nothing read, not everything read.

        Test scenario:
            extract(source, ()) returns {}.
        """
        assert _records.extract(full_source, ()) == {}, "an empty request should read nothing"

    def test_a_dtype_change_copies_rather_than_aliasing(self, full_source):
        """Casting to the canonical dtype yields a new array, not a view onto the source.

        The dtype-change case is the easy one - a cast always copies. The harder matching-dtype case is covered
        by TestExtractAlwaysCopies, since `extract` copies on every path regardless.

        Test scenario:
            A float64 intensity source comes back as uint16, which forces a copy; writing to the result must
            not reach back into the source.
        """
        source = _Source(
            _PointFormat("X", "Y", "Z", "intensity"),
            x=np.zeros(2),
            y=np.zeros(2),
            z=np.zeros(2),
            intensity=np.array([10.0, 20.0]),
        )

        arrays = _records.extract(source, ("intensity",))
        arrays["intensity"][0] = 99

        assert source.intensity[0] == 10.0, "extract must not hand back a view onto the source"


class TestExtractAlwaysCopies:
    """`extract` must own its arrays on every path, so the two engines cannot drift (M5)."""

    def test_a_matching_dtype_still_returns_an_independent_array(self):
        """Even when no cast is needed, the result is a copy - not a view aliasing the source record.

        On the COPC path the source is one packed record buffer; returning a view would keep the whole buffer
        alive behind a one-column result and let a caller mutate the source through it. The LAS path already
        copies, so the shared contract has to copy too.
        """
        values = np.array([2, 5], dtype=np.uint8)
        source = _Source(_PointFormat("X", "Y", "Z", "classification"),
                         x=np.zeros(2), y=np.zeros(2), z=np.zeros(2), classification=values)

        arrays = _records.extract(source, ("classification",))
        arrays["classification"][0] = 99

        assert values[0] == 2, "extract returned a view; mutating the result reached the source"
        assert arrays["classification"].base is None, "the returned array should own its data"
