# PyROOT Plotting Conventions

Responsible: **analysis-specific implementation convention**, not a CMS recommendation.

## Stored sources

| # | Source | Location |
|---|--------|----------|
| C1 | Canonical example | `MVA_training/pileup_symbolic_regression/corr_region_real_fake.py` |
| C2 | Shared gradient palette | `modules/root_2dColorProfile.py` (`set_gradient_style()`) |
| C3 | `hf*` jet-variable sentinel rule | project `CLAUDE.md` ("Working conventions specific to this repo") |

Classification tags: **[Analysis-specific]**, **[Implementation]**.

---

## 1. The pattern (C1)

Any new PyROOT (not matplotlib) plotting script should follow this sequence:

```python
import ROOT
from modules.root_2dColorProfile import set_gradient_style

ROOT.gROOT.SetBatch(True)          # before any drawing — no popup windows on a batch node
ROOT.gStyle.SetOptStat(0)          # no stats box
set_gradient_style()               # shared 5-stop RGB gradient for 2D histograms/profiles

h.SetMinimum(<explicit value>)
h.SetMaximum(<explicit value>)     # explicit Z-axis range so multi-panel canvases are
                                    # comparable to each other, not auto-scaled per panel
```

`set_gradient_style()` (C2) also calls `ROOT.TGaxis.SetMaxDigits(3)` to avoid
scientific-notation clutter on axes — call it once, before drawing, not per-histogram.

For ratio panels specifically, `SetMinimum`/`SetMaximum` are typically `0.0`/`1.0`
or similarly tight, explicit bounds (C1) rather than left to ROOT's auto-range.

---

## 2. `hf*` jet-variable sentinel guard (C3)

`hfsigmaEtaEta`, `hfsigmaPhiPhi`, `hfcentralEtaStripSize`, `hfadjacentEtaStripsSize`,
`hfEmEF`, `hfHEF` are **only physically meaningful for HF (forward, `|η| ≥ 3`)
jets**. NanoAOD fills them with a **`-1` sentinel** for every non-HF jet.

**Always guard on region (`|η| ≥ 3`) or on the sentinel value itself before
histogramming these** — plotting them unguarded mixes real HF measurements with a
flat spike of `-1`s from every central/HE jet, which will visibly (and silently)
distort any resulting distribution.

---

## 3. Review checklist

1. `SetBatch(True)` called before any `TCanvas`/drawing call.
2. `set_gradient_style()` used for 2D histograms/profiles instead of a locally
   redefined palette.
3. `SetOptStat(0)` set (unless the stats box is deliberately wanted for a specific
   debug plot — say so if so).
4. Z-axis (or ratio-panel) ranges set explicitly when the plot will be compared
   panel-to-panel or run-to-run.
5. Any `hf*` variable is filtered to HF jets, or its `-1` sentinel is excluded,
   before it's filled into a histogram.

## Last verified

- Local source review: 2026-09-17
