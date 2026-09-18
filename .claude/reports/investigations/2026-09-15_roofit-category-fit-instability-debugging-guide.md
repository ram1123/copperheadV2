# RooFit per-category fit-instability debugging guide (BW⊗DCB + RooCMSShape)

- Date: 2026-09-15
- Type: Investigation / methodology reference
- Status: Resolved for `30-45_EE`; methodology reusable for any future troubled category
- Applicable era: any (this is about the fit model, not a specific year)
- Relevant files: `src/lib/ebeMassResCalibration/{basic_class_for_calibration.py,fit_config.yml}`,
  `.claude/scripts/debug_calibration_fit_isolated.py`

## When to reach for this doc

Trigger conditions, any one of which means "don't trust this category's number yet":
- `fit_result.status() != 0` in `fit_params.json`.
- Every parameter's `err` is `0.0` in `fit_params.json` (HESSE failed).
- A category's `calibration_factor` is a 2x+ outlier vs. its neighbors (same eta at
  other pT bins, other eta combos at the same pT bin) -- **check this even when status=0
  and chi2 looks fine**. A converged, good-chi2 fit can still have an unphysical `sigma`
  (see Finding 1 below) -- cross-category consistency is a real check that chi2 alone
  will not catch.
- A parameter sits exactly at one of its configured min/max bounds in the fit dump
  (`Constant Parameter`/`Floating Parameter` table `fitTo()` prints, or `fit_params.json`'s
  `val`/`min`/`max` matching).

## How to iterate quickly

Use `.claude/scripts/debug_calibration_fit_isolated.py` instead of the full pipeline:
```bash
cd /cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2
CONDA_OVERRIDE_CUDA=12.4 pixi run -e default python3 -u \
  /path/to/repo/.claude/scripts/debug_calibration_fit_isolated.py \
  --input-path <label's stage1 save dir> --year <year> --sample <MC|Data> \
  --category <cat> --out-dir /tmp/some_scratch_dir
```
It reuses the cached `mass_<cat>.npy` (written by any prior `--fixCat` run) so each
iteration is seconds, not a full skim read. Edit `fit_config.yml`, rerun, inspect the
printed `fitTo()` dump + the saved PNG. Once satisfied, apply for real via
`getCalibrationFactor.py --steps step1 --fixCat <cat> ...` (matches the doc's own
`--fixCat` workflow) to update `fit_results.csv`/`calibration_factors.csv` in place.

## Parameter cheat sheet -- what each one actually controls, and how it fails

| Parameter | Physical meaning | Failure mode observed | Fix that worked |
|---|---|---|---|
| `observable.range`/`fit_range`/`full_range`/`fft_cache_range` | The mass window fit + binned. Must all four change together (a real category-specific override needs to touch all four, not just `fit_range` -- `range` controls the `RooRealVar`'s own binning/dataset window, not just what `fitTo()` sees) | If real data doesn't fill the default `[80,100]` window (check `mass_<cat>.npy.max()`), RooCMSShape has to describe a near-empty stretch, destabilizing everything | Restrict to match the real data extent. Also trim any *low*-side region with a visibly different (badly-modeled) shape, not just an empty tail -- see Finding 2 |
| `alpha1`/`alpha2` (CB tail onset) vs. `n1`/`n2` (CB tail power) vs. `sigma` (CB core width) | Together they are famously degenerate. If `alpha1` shrinks toward its floor (default `0.01`), the "core Gaussian" region becomes negligible and the power-law tail does all the work -- `sigma` can then collapse to an arbitrarily small, physically meaningless value *while chi2 looks great* | Symptom: converged-looking fit, good chi2, but `sigma` is a 3-5x outlier vs. neighbors. This is the single most important gotcha -- **chi2 alone will not catch it** | See Finding 1: the fix is *not* to force alpha/n to arbitrary neighbor values (can force a bad chi2 back in) -- fix the observable range first; this degeneracy is often a symptom of the model straining to cover a badly-fit region, not an independent bug |
| `bwz_mZ` (BW pole mass), `bwz_Width` (BW natural width) | Physically these are fixed constants (PDG values), not per-category shape knobs | When left floating with default bounds, they go degenerate with `sigma`/`mean` (natural width and pole mass can each independently absorb "total width" or "total shift"). Widening one bound just relocates the pinning to the next boundary -- whack-a-mole, not a fix | Widen (don't fully fix) with a modest floor and check where chi2 *plateaus* (multiple nearby floor values giving similar chi2 = you've found the plateau, not a leftover degeneracy). Fully fixing both to PDG was tried and was *worse* here (see Finding 3) -- too rigid, forces `mean` to overcompensate |
| `exp_alpha`/`exp_beta`/`exp_gamma`/`exp_peak` (RooCMSShape background) | Background shape/normalization | If the global fixed defaults don't match a category's real background, the *signal* model may silently warp to compensate (this is how the alpha/n degeneracy above often gets triggered in the first place) | Floating `exp_beta` alone reproduced the same signal collapse (didn't help). Floating `exp_beta`+`exp_gamma` *together* made MIGRAD hang outright (never returned within 240s) -- avoid floating more than one CMSShape shape parameter at once without a strong prior |
| `n2` at its `min: 0.0` floor | -- | Common collapse point in a troubled category | Give it a small nonzero floor (e.g. `min: 1.0`) before trying anything else |
| `Minos: true` (stage2 fit option) | Profile-likelihood-based asymmetric errors, computed independently of the Hessian | Does **not** help if HESSE is failing and the calling code reads `sigma.getError()` -- that method returns the Hessian-based (parabolic) error regardless of whether Minos ran. RooFit exposes Minos's results via `getErrorLo()`/`getErrorHi()` on the `RooRealVar`, which `save_fit_params_to_json`/`generateBWxDCB_RooCMSShape_plot` do not currently read | To actually benefit from Minos you'd need a code change (read `getAsymErrorLo/Hi` when Minos was requested), not just a `fit_config.yml` tweak -- out of scope for a config-only fix |

## Diagnostic order that worked in practice (30-45_EE case study)

1. **Check the raw data extent** via the cached `mass_<cat>.npy` (`.min()`/`.max()`,
   `(mass > edge).mean()`) *before* touching any RooFit parameter. Confirms whether the
   default `[80,100]` window is even appropriate for this category.
2. **Restrict `range` (all four observable keys) to match the real data**, re-fit, and
   check three things together -- not just one:
   - `status()` / whether HESSE produced nonzero errors,
   - chi2/ndf and the pull plot's visual flatness,
   - the resulting `sigma` (or calibration factor) against **neighboring categories**.
3. If `status`/errors look fine but `sigma` is a cross-category outlier (Finding 1),
   don't reach for forcing `alpha1/alpha2/n1/n2` yet -- first look at the pull plot for a
   *structured* residual pattern (a real, unmodeled feature in the data, e.g. an elevated
   shoulder) rather than pure noise. If there's a structured pattern in a specific mass
   sub-range, trim the range further to exclude it (Finding 2) rather than constraining
   signal-tail parameters to paper over it.
4. Only after the range is right, address any parameters still pinned at bounds: widen
   floors incrementally and watch where chi2/ndf **stops improving** (a plateau) rather
   than picking the first value that "looks converged." Re-widening the SAME parameter
   past the plateau can start trading width with `sigma` again and make things worse.
5. Zero HESSE errors that persist even in an otherwise-good fit (flat pulls, chi2/ndf
   near 1) usually mean exactly one parameter is still sitting at a hard bound -- a
   genuine flat likelihood direction, not necessarily a bad fit. Confirm by checking
   which single parameter is at a bound in the final dump; if the other two checks
   (pulls, cross-category sigma) are clean, this can be an accepted, documented
   limitation rather than something to keep chasing.

## Trial log: `30-45_EE`, 2022preEE Data (reference case)

All chi2 values below are chi2/ndf as printed on the plot (`frame.chiSquare(...)`).

| Trial | Config change (cumulative unless noted) | status | chi2/ndf | sigma (GeV) | sigma_err | Verdict |
|---|---|---|---|---|---|---|
| 0. Original | `fit_config.yml` global + `"30-45_*"` wildcard only (no `30-45_EE` override) | 303 | -- (raw chi2=1986.9) | 1.452 | 0.0 | **Broken** -- non-convergent, zero errors. Image: `images/30-45_EE_trial0_original_broken.png` |
| A. Range only | `range/fit_range/full_range/fft_cache_range = [80,92]` | 0 (forced pos-def) | 4.18 | 0.537 | 0.151 | Converged, clean-looking chi2, but sigma is a 3-5x outlier vs. neighbors (1.1-1.9 GeV) -- classic alpha/n collapse (alpha1->0.27, n1->1.0 floor, n2->~0 floor). Image: `images/30-45_EE_trialA_range_only.png` |
| B. + fix alpha1/alpha2/n1/n2/bwz_Width | Forced to neighbor-informed constants | 0 | 6.92 | 1.355 | 0.028 | Sigma now consistent with neighbors, but visibly worse fit -- systematic +2..+4 sigma pull across the real 80-85 GeV shoulder the model now fits badly. Image: `images/30-45_EE_trialB_alpha_n_fixed.png` |
| C1. exp_peak float instead of exp_beta | range[80,92] only, `exp_peak` floating | 0 | -- | 0.539 | 0.0 | No improvement -- `exp_peak` didn't even move from its init value; same alpha/n collapse as A |
| C2. Trim low edge too | `range=[84,92]` (excludes the 80-85 shoulder entirely) | 302 | 5.2 | 1.003 | 0.0 | Real improvement -- sigma much closer to neighbors, pulls visibly cleaner. HESSE still failing (n2 at its 0.0 floor) |
| C3. + n2 floor | `n2: {min: 1.0}` | -- | -- (~5.2) | 1.003 | 0.0 | n2 moved off its floor but HESSE still failed -- `bwz_mZ` now pinned at its 91.0 floor instead |
| C4. + widen bwz_mZ | `bwz_mZ: {min: 89.0}` | -- | **2.56** | 1.174 | 0.0 | Big jump in fit quality. `bwz_Width` now pinned at its 1.0 floor |
| C5. + bwz_Width floor 0.3 | -- | -- | (worse, degenerate alpha1~19.8) | 1.483 | 0.0 | Too aggressive -- relocated the degeneracy onto alpha1 instead |
| C6. bwz_Width floor 0.7 | -- | -- | **1.96** | 1.241 | 0.0 | Best result -- pulls genuinely flat across the whole range |
| C7. bwz_Width floor 0.5 | -- | -- | 1.95 | 1.395 | 0.0 | Essentially tied with C6 -- confirms chi2 has *plateaued* over this bwz_Width range (0.5-0.7), i.e. found the real plateau, not a leftover degeneracy |
| C8. Fully fix bwz_mZ + bwz_Width to PDG | `const: true` on both | -- | 5.01 | 1.109 | 0.0 | Worse than C6/C7 -- too rigid, forced `mean` to overcompensate (`mean`->-3.18) |
| C9. C6 + `Minos: true` | -- | -- | 1.96 (same as C6) | 1.241 | 0.0 (unchanged) | Confirmed Minos doesn't help -- see cheat-sheet entry above |
| **Final (= C6)** | range `[84,92]` + `n2: {min:1.0}` + `bwz_mZ: {min:89.0}` + `bwz_Width: {min:0.7}` | 0 | **1.96** | **1.241** | 0.0 (known, accepted limitation) | **Applied to production.** Image: `images/30-45_EE_trialFinal_C6.png` |

## Final `fit_config.yml` recipe (already applied)

```yaml
"30-45_EE":
    observable:
        range: [84.0, 92.0]
        fit_range: [84.0, 92.0]
        full_range: [84.0, 92.0]
        fft_cache_range: [84.0, 92.0]
    params:
        n2: { min: 1.0 }
        bwz_mZ: { min: 89.0 }
        bwz_Width: { min: 0.7 }
```

## Key findings (generalizable beyond this one category)

1. **A good chi2/ndf does not guarantee a physically meaningful `sigma`.** The
   alpha/n/sigma crystal-ball degeneracy can produce an excellent-looking fit with a
   `sigma` that's 3-5x smaller than every neighboring category. Always cross-check the
   extracted resolution against neighbors (same eta at other pT, other eta at the same
   pT) before trusting a "converged" result.
2. **A structured residual pattern (not noise) in the pull plot means a real, unmodeled
   feature in the data** -- usually fixed by trimming the fit range to exclude that
   region, not by constraining signal-tail parameters to force-fit it.
3. **Fully fixing physically-constant parameters (BW pole mass/width) to PDG values is
   not always the safest choice** -- if the category's real peak is genuinely shifted
   (detector bias, resolution skew), forcing BW to the exact PDG value pushes all of that
   shift onto `mean`, which can overcompensate and produce a worse fit than a moderately
   widened (not fully free, not fully fixed) BW bound.
4. **When repeatedly widening one parameter's bound just relocates the pinning to a new
   parameter (whack-a-mole), look for a plateau** -- try two or three floor values and
   check if chi2/ndf stops improving; if it's flat across a range, you've found the real
   optimum, not a symptom to keep chasing.
5. **Floating multiple RooCMSShape background parameters together is risky** -- one at a
   time is fine; `exp_beta`+`exp_gamma` together made MIGRAD hang outright.
6. **Zero HESSE errors in an otherwise excellent fit (flat pulls, chi2/ndf~1-2) usually
   means exactly one parameter is at a hard bound** (a genuine flat likelihood
   direction). This can be an accepted, documented limitation rather than a sign the fit
   itself is wrong -- but it should be called out explicitly, not silently accepted.

## Remaining work

- Apply this same recipe (or a category-specific variant, following the same diagnostic
  order above) to `30-45_EE` for the other 13 year/sample combinations -- see the
  companion check requested 2026-09-15 (recheck all `30-45_EE` plots for wrong fits).
- Consider a small code change to `save_fit_params_to_json` to read
  `getErrorLo()`/`getErrorHi()` when Minos was used, so a category like this one could
  get a real (asymmetric) uncertainty instead of the current `0.0` -- not done here
  (config-only fix), flagged for a future session.
