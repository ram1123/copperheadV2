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

To fix this, set the environment variable `CONDA_OVERRIDE_CUDA=12.0` before running the `pixi shell` command.

```bash
export CONDA_OVERRIDE_CUDA=12.0
pixi shell
```
