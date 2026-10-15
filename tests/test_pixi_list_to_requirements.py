"""Tests for `ci/pixi_list_to_requirements.py`, the input stage of the pip-audit CI gate.

The script decides what gets audited. A bug that silently drops entries would not fail any job - it would make
the advisory gate pass over fewer packages, which is exactly the failure the round-1 review found in the version
that used `pip freeze`. So the filtering rules and the empty-inventory guard are pinned here.
"""

from __future__ import annotations

import io
import json

import pytest
from pixi_list_to_requirements import main, to_requirements


@pytest.fixture
def pixi_entries() -> list[dict]:
    """A pixi inventory with one of every shape the script has to handle.

    Returns:
        list[dict]: records as `pixi list --json` emits them - PyPI packages, a conda package, the versionless
            editable self-install, and an entry whose name needs normalising.
    """
    return [
        {"name": "scipy", "version": "1.18.1", "kind": "pypi"},
        {"name": "numpy", "version": "2.5.4", "kind": "pypi"},
        {"name": "python", "version": "3.14.8", "kind": "conda"},
        {"name": "pointclouds", "kind": "pypi"},
        {"name": "pyramids_gis", "version": "0.68.0", "kind": "pypi"},
    ]


class TestToRequirements:
    """Tests for to_requirements."""

    def test_keeps_only_pypi_packages(self, pixi_entries):
        """Test conda records are dropped while PyPI records survive.

        Test scenario:
            The inventory holds `python` as a conda package. pip-audit resolves names against PyPI, where a
            conda-only build is not a project, so it must not reach the requirements file.
        """
        result = to_requirements(pixi_entries)

        assert not any(line.startswith("python==") for line in result), f"conda package leaked through: {result}"
        assert "numpy==2.5.4" in result, f"PyPI package missing: {result}"

    def test_drops_the_versionless_self_install(self, pixi_entries):
        """Test the editable project entry is excluded.

        Test scenario:
            `pixi list` reports this package as a PyPI entry with no version (it is installed from `./`).
            pip-audit cannot look up a versionless name, and there is no release to audit.
        """
        result = to_requirements(pixi_entries)

        assert not any("pointclouds" in line for line in result), f"self-install leaked through: {result}"

    @pytest.mark.parametrize(
        "name",
        ["pointclouds", "PointClouds", "POINTCLOUDS"],
        ids=["lower", "mixed-case", "upper"],
    )
    def test_self_install_excluded_whatever_the_case(self, name):
        """Test the exclusion is case-insensitive.

        Args:
            name: the project name as pixi might report it.

        Test scenario:
            A versioned entry for this package - which happens if it is ever installed non-editably - is still
            excluded, because the exclusion is matched on the normalised name. The distribution name carries no
            separator, so case is the only variation that can occur.
        """
        result = to_requirements([{"name": name, "version": "0.1.0", "kind": "pypi"}])

        assert result == [], f"{name!r} should have been excluded, got {result}"

    def test_a_differently_named_project_is_not_excluded(self):
        """Test `point-clouds` is audited rather than mistaken for this package.

        Test scenario:
            PEP 503 normalises `_` to `-`, so `point_clouds` and `point-clouds` are one name - and that name is
            not this distribution. Were the exclusion to strip separators instead, a genuine third-party
            dependency called point-clouds would silently drop out of the audit.
        """
        result = to_requirements([{"name": "point_clouds", "version": "9.9.9", "kind": "pypi"}])

        assert result == ["point_clouds==9.9.9"], f"a distinct project must still be audited, got {result}"

    def test_output_is_sorted(self, pixi_entries):
        """Test the lines come back in sorted order.

        Test scenario:
            pixi's ordering is not guaranteed stable across platforms; a sorted file keeps CI logs and any diff
            of the generated requirements readable.
        """
        result = to_requirements(pixi_entries)

        assert result == sorted(result), f"output is not sorted: {result}"

    def test_formats_each_line_as_a_pin(self, pixi_entries):
        """Test every line is exactly `name==version`.

        Test scenario:
            pip-audit rejects a requirements file holding URLs, paths or unpinned names, and `--no-deps` needs
            every entry pinned.
        """
        result = to_requirements(pixi_entries)

        assert result == ["numpy==2.5.4", "pyramids_gis==0.68.0", "scipy==1.18.1"], f"unexpected lines: {result}"

    def test_empty_inventory_yields_no_lines(self):
        """Test an empty input produces an empty list rather than raising.

        Test scenario:
            The emptiness check belongs to `main`, which turns it into a non-zero exit; the pure function just
            reports what it found.
        """
        assert to_requirements([]) == [], "empty inventory should yield no lines"

    def test_entry_without_kind_is_ignored(self):
        """Test a record missing the `kind` field is skipped.

        Test scenario:
            Defensive: a future pixi version could omit the field. Skipping is right - auditing a package whose
            origin is unknown would at best produce a spurious failure.
        """
        result = to_requirements([{"name": "mystery", "version": "1.0"}])

        assert result == [], f"record without a kind should be skipped, got {result}"

    def test_does_not_mutate_its_input(self, pixi_entries):
        """Test the inventory passed in is left untouched.

        Test scenario:
            Purity: the caller may reuse the decoded JSON (the CI step does not, but a future caller might).
        """
        before = json.dumps(pixi_entries, sort_keys=True)

        to_requirements(pixi_entries)

        assert json.dumps(pixi_entries, sort_keys=True) == before, "input was mutated"


class TestMain:
    """Tests for main."""

    def test_writes_pins_to_stdout_and_exits_zero(self, pixi_entries, monkeypatch, capsys):
        """Test a populated inventory is written to stdout with a zero exit code.

        Args:
            pixi_entries: inventory fixture.
            monkeypatch: used to feed stdin.
            capsys: captures stdout.

        Test scenario:
            This is the CI path: `pixi list --json | python ci/pixi_list_to_requirements.py > reqs.txt`.
        """
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(pixi_entries)))

        code = main()

        out = capsys.readouterr().out
        assert code == 0, f"expected exit 0, got {code}"
        assert out.splitlines() == ["numpy==2.5.4", "pyramids_gis==0.68.0", "scipy==1.18.1"], f"bad stdout: {out!r}"

    def test_refuses_an_empty_inventory(self, monkeypatch, capsys):
        """Test an inventory with no PyPI package exits 1 and says why on stderr.

        Args:
            monkeypatch: used to feed stdin.
            capsys: captures stdout and stderr.

        Test scenario:
            The guard that stops the audit passing vacuously. Without it, a mis-piped command or an environment
            whose packages are all conda would produce an empty file and a green pip-audit job.
        """
        monkeypatch.setattr("sys.stdin", io.StringIO("[]"))

        code = main()

        captured = capsys.readouterr()
        assert code == 1, f"expected exit 1, got {code}"
        assert "refusing to audit an empty set" in captured.err, f"missing explanation: {captured.err!r}"
        assert captured.out == "", f"nothing should reach stdout, got {captured.out!r}"

    def test_conda_only_inventory_is_treated_as_empty(self, monkeypatch, capsys):
        """Test an inventory of conda packages alone is refused.

        Args:
            monkeypatch: used to feed stdin.
            capsys: captures stderr.

        Test scenario:
            Nothing in it is auditable by pip-audit, so passing it on would be the vacuous-pass failure again.
        """
        inventory = json.dumps([{"name": "python", "version": "3.14", "kind": "conda"}])
        monkeypatch.setattr("sys.stdin", io.StringIO(inventory))

        code = main()

        assert code == 1, f"expected exit 1, got {code}"
        assert "refusing to audit an empty set" in capsys.readouterr().err, "guard did not fire"

    def test_malformed_json_propagates(self, monkeypatch):
        """Test invalid JSON on stdin raises rather than being swallowed.

        Args:
            monkeypatch: used to feed stdin.

        Test scenario:
            A broken pipe or an error message arriving on stdin must fail the step loudly; returning 0 on
            unparseable input would hide a broken audit.
        """
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

        with pytest.raises(json.JSONDecodeError):
            main()
