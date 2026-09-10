---
title: Get Yields
---

# Get yield

The main script is [get_yields.py](../scripts/get_yields.py). It reads stage-1 parquet files and computes raw event counts and weighted yields after applying the standard category and region selections.

How to run:

```bash
python scripts/get_yields.py \
   --input /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output \
   -y 2017
```

The `--input` path should point to the `stage1_output` directory. The script then resolves the per-year input automatically and will prefer `compacted` over `f1_0` when the compacted directory exists.

## Sample list source

The list of samples to process is read from:

- [configs/samples/samples.yaml](../configs/samples/samples.yaml)

Inside [get_yields.py](../scripts/get_yields.py), this is done through:

```py
get_bkg_sig_dicts(
    yaml_path="configs/samples/samples.yaml",
    year=year,
)
```

## Selection source

The event selection is defined in:

- [modules/selection.py](../modules/selection.py)

The script applies the standard category and region cuts through:

```py
def applyRegionCatCuts(
    events,
    category: str,
    region_name: str,
```

This function starts at [modules/selection.py](../modules/selection.py#L37).

## How ggH and VBF are split

The ggH and VBF category split is also defined in [modules/selection.py](../modules/selection.py#L122-L133):

```py
if category == "vbf":
    prod_cat_cut = prod_cat_cut & vbf_cut
    prod_cat_cut = prod_cat_cut & (
        ~btag_cut
    )  # btag cut is for VH and ttH categories
elif category == "ggh":
    prod_cat_cut = prod_cat_cut & ~vbf_cut
    prod_cat_cut = prod_cat_cut & (
        ~btag_cut
    )
```

## Counting events in a custom mass window (no category / region cut)

`get_yields.py` only reports the Higgs regions (h-peak `115 < m_mm < 135`, h-sidebands
`110-115` & `135-150`) because `applyRegionCatCuts` hard-codes them. If you just want the
number of events in some mass window (e.g. the Z window `70 < m_mm < 110`) with **no**
ggH/VBF split and no region cut, use:

- [count_dimuon_mass_window.py](../scripts/count_dimuon_mass_window.py)

The compacted stage-1 parquet is **not** mass-window filtered — it holds the full
`dimuon_mass` spectrum after the baseline dimuon selection (two good opposite-charge muons,
trigger, PV, etc.). This script reads only the `dimuon_mass` column (fast, ~10 s for a full
year of data, no Dask) and counts.

```bash
python scripts/count_dimuon_mass_window.py \
   --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/stage1_output \
   -y 2025 --samples 'data_[C-G]' --mass-min 70 --mass-max 110 --breakdown
```

- `--input` — same as `get_yields.py`: point at `stage1_output`, it resolves the per-year
  path and prefers `compacted/` over `f1_0/`.
- `--samples` — sample-dir glob(s), default `data*`. Bracket classes work, so
  `'data_[C-G]'` drops `data_B`.
- `--mass-min` / `--mass-max` — window edges in GeV (default `70` / `110`).
- `--field` — mass branch (default `dimuon_mass`).
- `--breakdown` — also print a fixed mass-bin distribution table (useful sanity check: the
  `[115,135)` bin equals `get_yields.py`'s "nocat h-peak", and `[110,115)+[135,150)` equals
  "nocat h-sidebands").

Example output (2025 data, eras C–G):

```
rows on disk (no window)  : 83718768
count 70 < dimuon_mass < 110 : 75960754

mass distribution (GeV bins):
  [     0,     60) : 4201627
  [    60,     70) : 1421618
  [    70,     76) : 1359741
  [    76,    106) : 74092153
  [   106,    110) : 508860
  [   110,    115) : 412255
  [   115,    135) : 786345
  [   135,    150) : 271462
  [   150,    200) : 391710
  [   200,    inf) : 272997
```

## Comparing yields

There is also [compare_yield_csv.py](../scripts/compare_yield_csv.py), which can be used to compare two CSV outputs produced by [get_yields.py](../scripts/get_yields.py). This is useful for checking whether two productions give consistent yields or for spotting regressions.

This will print the summary like:

```bash
====================================================================================================
YEAR: 2022postEE
====================================================================================================
           sample category      region       year  raw_events_A  raw_events_B  raw_events_diff_pct     yield_A     yield_B  yield_diff_pct
            data*      vbf      h-peak 2022postEE          1326          1307            -1.432881 1326.000000 1307.000000       -1.432881
 dyTo2L_M-50_incl      vbf      h-peak 2022postEE         11471         10821            -5.666463 2252.661380 2046.038848       -9.172374
            ewk_*      vbf      h-peak 2022postEE          3835          3799            -0.938722   51.667347   51.237093       -0.832739
     ggh_powhegPS      vbf      h-peak 2022postEE         10953         10278            -6.162695    6.210484    5.763953       -7.189949
         ttjets_*      vbf      h-peak 2022postEE         15632         15207            -2.718782  250.612726  242.916114       -3.071118
vbf_powheg_dipole      vbf      h-peak 2022postEE        326802        321480            -1.628509    6.550290    6.436373       -1.739113
             w*_*      vbf      h-peak 2022postEE          6491          6111            -5.854260   44.101458   41.194833       -6.590770
             zz_*      vbf      h-peak 2022postEE          5232          4892            -6.498471    5.439032    5.075334       -6.686812
            data*      vbf h-sidebands 2022postEE          1139          1120            -1.668130 1139.000000 1120.000000       -1.668130
 dyTo2L_M-50_incl      vbf h-sidebands 2022postEE         10064          9522            -5.385533 1840.600305 1681.217659       -8.659275
            ewk_*      vbf h-sidebands 2022postEE          3278          3244            -1.037218   44.612385   44.102603       -1.142693
     ggh_powhegPS      vbf h-sidebands 2022postEE           182           169            -7.142857    0.107269    0.098133       -8.517135
         ttjets_*      vbf h-sidebands 2022postEE         13901         13532            -2.654485  222.928150  216.068470       -3.077081
vbf_powheg_dipole      vbf h-sidebands 2022postEE          4611          4536            -1.626545    0.094564    0.092808       -1.857152
             w*_*      vbf h-sidebands 2022postEE          5711          5398            -5.480651   38.706532   36.228572       -6.401917
             zz_*      vbf h-sidebands 2022postEE          4533          4248            -6.287227    4.616150    4.339529       -5.992463

====================================================================================================
```
