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
    """Convert pixi's package records into sorted ``name==version`` lines.

    Args:
        packages: the decoded `pixi list --json` array.

    Returns:
        Sorted `name==version` strings, one per PyPI package that carries a version.

    Examples:
        >>> to_requirements([
        ...     {"name": "numpy", "version": "2.5.4", "kind": "pypi"},
        ...     {"name": "python", "version": "3.14.8", "kind": "conda"},
        ...     {"name": "pointclouds", "kind": "pypi"},
        ... ])
        ['numpy==2.5.4']
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

    Returns:
        ``0`` on success, ``1`` when the environment yielded no PyPI package at all - which means the inventory
        was empty or the wrong thing was piped in, and auditing it would pass vacuously.
    """
    lines = to_requirements(json.load(sys.stdin))
    if not lines:
        print("no PyPI packages in this pixi environment - refusing to audit an empty set", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
