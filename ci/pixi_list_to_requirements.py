"""Turn `pixi list --json` output into a pinned requirements file for pip-audit.

Used by `.github/workflows/pip-audit.yml`. It exists as a script rather than an inline heredoc so the conversion
can be run and tested locally:

    pixi list -e docs --json | python ci/pixi_list_to_requirements.py > reqs.txt

Why not `pip freeze`: `pip` is only in the `dev` dependency group, so `python -m pip freeze` in the `docs`
environment prints nothing and the audit would pass over an empty file. `pixi list` needs nothing installed inside
the environment.

Only `kind == "pypi"` entries survive. pip-audit resolves names against PyPI, and the conda packages in a pixi
environment (python, openssl, libzlib, ...) are not PyPI projects. The editable self-install is dropped too: it
carries no version in the lock and has no release to audit.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable

#: Distributions to leave out: this package itself, under either spelling.
_EXCLUDED = {"pointclouds"}


def to_requirements(packages: Iterable[dict]) -> list[str]:
    """Convert pixi's package records into sorted `name==version` lines.

    Three records are dropped: anything whose `kind` is not `pypi` (a conda build is not a PyPI project, so
    pip-audit cannot resolve it), anything without a version, and this package itself. What survives is a fully
    pinned set, which is what lets the audit run with `--no-deps`.

    Args:
        packages: the decoded `pixi list --json` array. Each record is a mapping with at least `name`, and
            normally `version` and `kind`; missing keys are treated as "not auditable" rather than as an error.

    Returns:
        Sorted `name==version` strings, one per PyPI package that carries a version. Empty when nothing in the
        inventory qualifies.

    Examples:
        - A mixed inventory keeps only the versioned PyPI packages:
            ```python
            >>> lines = to_requirements([
            ...     {"name": "scipy", "version": "1.18.1", "kind": "pypi"},
            ...     {"name": "numpy", "version": "2.5.4", "kind": "pypi"},
            ...     {"name": "python", "version": "3.14.8", "kind": "conda"},
            ... ])
            >>> lines
            ['numpy==2.5.4', 'scipy==1.18.1']
            >>> len(lines)
            2

            ```
        - The editable self-install carries no version, so it never reaches the audit:
            ```python
            >>> to_requirements([{"name": "pointclouds", "kind": "pypi"}])
            []

            ```
        - A differently named project is kept, because `point-clouds` is not this distribution:
            ```python
            >>> to_requirements([{"name": "point_clouds", "version": "9.9.9", "kind": "pypi"}])
            ['point_clouds==9.9.9']

            ```

    See Also:
        main: the stdin-to-stdout wrapper the CI step actually calls.
    """
    return sorted(
        f"{package['name']}=={package['version']}"
        for package in packages
        if package.get("kind") == "pypi"
        and package.get("version")
        and package.get("name", "").lower().replace("_", "-") not in _EXCLUDED
    )


def main() -> int:
    """Read `pixi list --json` on stdin and write requirements lines to stdout.

    This is what `.github/workflows/pip-audit.yml` runs. The non-zero exit on an empty result is the point: an
    empty requirements file makes pip-audit report success, so a mis-piped command or an environment whose
    packages are all conda would otherwise look like a clean advisory scan.

    Returns:
        `0` after writing at least one pin to stdout, `1` when nothing in the inventory was auditable. The
        explanation for a `1` goes to stderr, so stdout stays safe to redirect into a requirements file.

    Raises:
        json.JSONDecodeError: when stdin does not hold valid JSON - a broken pipe or an error message arriving
            in place of the inventory fails the step rather than producing an empty audit.

    Examples:
        - A populated inventory is written to stdout, one pin per line, and exits 0:
            ```python
            >>> import io, sys
            >>> saved, sys.stdin = sys.stdin, io.StringIO('[{"name": "numpy", "version": "2.5.4", "kind": "pypi"}]')
            >>> main()
            numpy==2.5.4
            0
            >>> sys.stdin = saved

            ```
        - An inventory with nothing auditable exits 1 and writes nothing to stdout:
            ```python
            >>> import io, sys
            >>> saved, sys.stdin = sys.stdin, io.StringIO('[{"name": "python", "version": "3.14", "kind": "conda"}]')
            >>> main()
            1
            >>> sys.stdin = saved

            ```

    See Also:
        to_requirements: the filtering rules, which this only wraps in I/O and an exit code.
    """
    lines = to_requirements(json.load(sys.stdin))
    if not lines:
        print("no PyPI packages in this pixi environment - refusing to audit an empty set", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
