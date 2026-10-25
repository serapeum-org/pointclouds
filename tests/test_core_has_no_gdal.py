"""The core must import and work with no pyramids and no GDAL installed.

This is the automated guard for the package's defining constraint: only ``pointclouds.convert`` may touch
pyramids/GDAL, and only when the ``[gis]`` extra is present. If these imports start pulling GDAL, the core stops
installing on platforms pyramids has no wheel for - which is the whole reason the dependency is optional.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

_CORE_MODULES = [
    "pointclouds",
    "pointclouds.points",
    "pointclouds.io.las",
    "pointclouds.io.copc",
    "pointclouds.io.ept",
    "pointclouds.io._records",
    "pointclouds.ops.grid",
    "pointclouds.ops.filter",
    "pointclouds.ops.decimate",
    "pointclouds.classify",
    "pointclouds.convert",  # importable without the extra; only calling a converter needs it
]


@pytest.mark.parametrize("name", _CORE_MODULES)
def test_core_module_imports_without_gis(name, monkeypatch):
    """Each core module imports even when pyramids/osgeo are unavailable."""
    real_import = builtins.__import__

    def _blocked(mod, *args, **kwargs):
        top = mod.split(".")[0]
        if top in {"pyramids", "osgeo"}:
            raise ModuleNotFoundError(f"blocked for test: {mod}")
        return real_import(mod, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    importlib.reload(importlib.import_module(name))


def test_to_array_core_path(tiny_cloud):
    """`to_array` is a pure-core operation and returns an (N, k) block."""
    xyz = tiny_cloud.to_array(("X", "Y", "Z"))
    assert xyz.shape == (4, 3)


def test_require_pyramids_hint_is_raised_when_the_import_fails(monkeypatch):
    """The hint path raises `ModuleNotFoundError` naming the extra, whether or not pyramids is installed.

    The test below only exercises this when pyramids is genuinely absent, so it skips in the `full`
    environment - the one environment where a regression in the guarded import would matter least to spot
    late. Blocking the import directly makes the contract hold in every environment.
    """
    from pointclouds import convert

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("pyramids"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ModuleNotFoundError, match=r"pointclouds\[gis\]") as excinfo:
        convert.require_pyramids()

    assert excinfo.value.__cause__ is not None, "the original ImportError should be chained as __cause__"


def test_require_pyramids_hint_without_extra():
    """`require_pyramids` names the [gis] extra when pyramids is absent (core path)."""
    from pointclouds import convert

    try:
        import pyramids  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError, match=r"pointclouds\[gis\]"):
            convert.require_pyramids()
    else:
        pytest.skip("pyramids is installed; the hint path is not exercised")
