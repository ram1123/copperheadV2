---
title: VBF Statistics Pipeline
---

## VBF stats pipeline

After `stage3` has produced the datacards, run the Combine chain with
[run_stats_pipeline_VBF.sh](../run_stats_pipeline_VBF.sh). It needs the `combine` pixi environment
(different ROOT pin than `default`) and does **not** rebuild stage-2/stage-3; run those first.

```bash
./enter_pixi.sh combine
bash run_stats_pipeline_VBF.sh -m 11 -y Run3 -l label_for_ntuple
```

`-y` takes a single year or a pseudo-year (`Run3`, `Run2`, `Run2Run3`) that combines the already-built
per-year cards. It also accepts a comma-separated list, but then every step runs once per year; pass
one pseudo-year to get a single combined result.

### Modes

| `-m` | Name | What it runs |
| --- | --- | --- |
| `4` | `combine_vbf` | Build the VBF datacard + Combine workspace |
| `5` | `combine_vbf_significance` | Signal and stat-only significance fits |
| `6` | `combine_vbf_summary` | (Re)collect the significance summary CSV only |
| `7` | `combine_vbf_impacts` | Nuisance impacts, Asimov `r=1` and `r=0` |
| `8` | `combine_vbf_lhscan` | 1D likelihood scan (Asimov, `r=1` injected) |
| `9` | `combine_vbf_all` | `4` + `5` + significance summary |
| `10` | `combine_vbf_limit` | Expected 95% CL limit (`AsymptoticLimits --run blind`) + limit summary |
| `11` | `vbf_limit` | `9` + `10`: card, workspace, significance, expected limit, both summaries |
| `12` | `combine_vbf_jjregions` | Combine the `jj_both_central` + `jj_non_central` cards into one, then significance + limit |
| `13` | `combine_vbf_jjregions_impacts` | Same combination as `12`, then impacts |
| `14` | `combine_vbf_jjregions_lhscan` | Same combination as `12`, then the likelihood scan |

`./run_stats_pipeline_VBF.sh -h` prints the full description of every mode.

!!! warning "Blinded analysis"
    No mode fits real data. Impacts run on Asimov datasets with `r=1` (signal injected) and `r=0`
    (background-only) instead of an observed scenario, and limits are expected limits only.

!!! note "jj η-region runs"
    `JJ_ETA_REGION` (default `all`) must match the stage-2/3 run being targeted; a mismatch shows up
    as `Missing VBF SR/SB datacards for <year>`. Modes `12`–`14` need modes `4`/`9`/`11` already run
    under both `JJ_ETA_REGION=jj_both_central` and `JJ_ETA_REGION=jj_non_central`.
    `JJ_COMBINE_REGIONS` overrides the region pair (e.g.
    `jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25`).

## χ²/NDF for a blinded fit

To compute χ²/NDF when fitting only the sideband regions of a blinded analysis, use
`chi2_ndf_manual()` in [docs/demo/blindedChi2NDF/chi2Ndf.py](../docs/demo/blindedChi2NDF/chi2Ndf.py), which also
demonstrates how to call it.
