# Contributing

See the repo README and `planning/placement-decision.md`.

## Development environment

The repo is managed with [pixi](https://pixi.sh); `pixi.lock` is committed and CI installs from it. Everything
below assumes pixi 0.73 or newer — that is the version pinned in the workflows, and the lock is format v7, which
older pixi refuses to read.

**Before the first `pixi install`, decide where the environments go.** The manifest declares nine of them, and a
full set is several GB. By default pixi materialises them in `.pixi/envs/` inside the checkout, which is the
wrong place if your clone lives in a synced folder (OneDrive, Dropbox, Google Drive): the sync client locks files
mid-install and you get permission errors on the next install or clean. Point pixi somewhere outside the tree
instead:

```bash
pixi config set --local detached-environments "C:/python-environments/pixi/envs"   # Windows
pixi config set --local detached-environments "$HOME/.cache/pixi/envs"            # Linux / macOS
```

That writes `.pixi/config.toml`, which is **deliberately not tracked** — it holds an absolute path specific to
your machine. CI sets nothing and uses the default location, which is correct there.

## Environments and tasks

| environment | what it is for |
|---|---|
| `dev` | the day-to-day environment: the package plus the `dev` tooling, on Python 3.14 |
| `py311` … `py314` | the CI test matrix, one per supported interpreter |
| `full` | `dev` plus the `gis` and `cloud` extras — the only environment where `pytest -m gis` can run |
| `docs` | MkDocs, mkdocstrings and mike |
| `notebook` | `dev` plus JupyterLab, for interactive work |
| `default` | the package alone, with no tooling |

Run a task with `pixi run -e <environment> <task>`:

```bash
pixi run -e dev main         # the core suite: no pyramids, no GDAL
pixi run -e full test-all    # the whole suite, extras included
pixi run -e dev typecheck    # mypy
pixi run -e dev doctest      # src doctests
pixi run -e dev sample-data  # regenerate the committed LAS/LAZ fixtures
pixi run -e full notebooks   # nbval over docs/examples
```

`pixi task list -e <environment>` shows what is available where. The test and tooling tasks live on the `dev`
feature, so they exist in `dev`, `full` and the `pyXXX` environments — not in `docs`, `notebook` or `default`,
which have no pytest to run them with.

## Before you push

```bash
pre-commit run --all-files
pixi run -e dev main
pixi run -e dev typecheck
```

The pre-commit hooks are the same ones CI runs. Two are skipped in CI and should not be skipped locally:
`no-commit-to-branch` (it is a local guard, and CI legitimately runs on `main`) and `mypy` (CI runs it in the
`static` job, inside the `py314` environment, which is the same feature set as `dev`).

## House rules

- The core stays GDAL-free. Only `src/pointclouds/convert.py` may import `pyramids`, inside a function, behind a
  guarded import that names the `[gis]` extra. `tests/test_core_has_no_gdal.py` enforces it — keep it green.
- `pointclouds` owns point clouds; `pyramids` owns raster and vector (ADR-0001). Work that belongs upstream is
  reported there rather than reimplemented here.
- Conventional Commits, and a branch per change: `<type>/<short-description>`.
