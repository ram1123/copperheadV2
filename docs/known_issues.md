---
title: Known issues
---

# Known issues

## Pixi CUDA issue

When I try to run `pixi shell` command, it gives the following error:

```bash
[shar1172@purdue-af-182 copperheadV2_Feb2026_depot]$ pixi shell
Error:   × Cannot install environment 'default'
  ╰─▶ Virtual package '__cuda' does not match any of the available virtual packages on your machine: [__glibc=2.28=0, __unix=0=0, __archspec=1=zen2, __linux=4.18.0=0]
  help:  You can mock the virtual package by overriding the environment variable, e.g.: '`CONDA_OVERRIDE_CUDA=12.0`'
```

This only happens for environments that include the `cuda` feature (currently `default` and
`default-legacy` — the day-to-day analysis environments with ROOT/ML/symbolic-regression). Fix:
set `CONDA_OVERRIDE_CUDA` before running `pixi shell`, to a value >= the `cuda` entry on the
`linux-64-cuda` platform in `pixi.toml`'s `[workspace] platforms` list (currently `12.4`; check
that file if this stops working again after a version bump there).

```bash
export CONDA_OVERRIDE_CUDA=12.4
pixi shell
```

Environments that don't include the `cuda` feature (`ci`, `ci-legacy`, `combine`,
`combine-legacy`) resolve on GPU-less machines without any override — the CUDA requirement is
scoped per-feature via a named platform variant (`linux-64-cuda`) rather than applied
workspace-wide, so plain `pixi shell -e ci` (etc.) just works.
