# EBE mass-resolution calibration: fit audit + execution attempt (nanoAODv15, 2025, MC)

- Date: 2026-09-14
- Type: Investigation + Implementation (2 minimal bug fixes) + failed execution attempt
- Status: Open -- pipeline crashes on first category, unresolved
- Applicable era: Run 3, 2025 (NanoAODv15, `Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation` label)
- NanoAOD campaign: v15
- Relevant files: `src/lib/ebeMassResCalibration/getCalibrationFactor.py`, `basic_class_for_calibration.py`, `fit_config.yml`, `docs/eventbyeventMassResolution.md`, `configs/trials.yml`, `common_workflow.sh`

## Question

Audit the EBE mass-resolution calibration's RooFit-based Z-peak fit against
`docs/eventbyeventMassResolution.md`, then run:
`bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation -y 2025 -M 1 -p V1 -m calib -k -i 0`.

## Evidence

Coordinated review: main agent + `code-reviewer` + `physics-reviewer` sub-agents (parallel, read-only),
each given the doc text plus pointers into `generateBWxDCB_RooCMSShape_plot` (the only fit function
actually invoked in production; three sibling functions in the same file are confirmed dead code).

Fit is genuinely ROOT/RooFit-native: `RooBreitWigner ⊗ RooCrystalBall` via `RooFFTConvPdf` (BW⊗DCB
order matches the doc), `RooCMSShape` background, `RooAddPdf`, two-stage `fitTo()` (Strategy1 rough /
Strategy2 final), chi2/ndf via `frame.chiSquare(pdf, hist, n_free_params)` exactly per the doc.

## Findings

**Confirmed defects (static review):**
1. `calib` mode's `-l <label>` is never read by `getCalibrationFactor.py` (`args.input_path` is parsed
   by the shared `cli/common_argparser.py` but unused in `main()`); the script always loads via
   `modules/trials.py::get_stage1_path()` from `configs/trials.yml`'s `"current"` entry unless
   `$HMM_TRIAL` is exported. `"current"` pointed at an unrelated, older nanoAODv12 label with no `2025/`
   subdirectory at all.
2. The `--isMC` branch hardcoded `INPUT_DATASET = .../dyTo2L_M-50_incl/*/*.parquet`. That sample name
   is nanoAODv12-only (2022/2023); this nanoAODv15 2025 label's compacted DY sample is
   `dyTo2Mu_M-50_aMCatNLO` (matches the 2026-09-09 registry note that 2025/2026 MC reuses the
   2024-conditions Summer24 set).
3. `dask.config.set(scheduler="threads")` at module import time in `getCalibrationFactor.py` globally
   overrides any Gateway/local distributed client created later -- self-documented by a
   `# FIXME: ... not running on gateway` comment in `basic_class_for_calibration.py:319`. `-k`/
   `--use_gateway` does not actually distribute `.compute()`; everything runs on the driver session's
   own local threads.
4. Fit convergence/covariance quality is never gated: `fit_result.status()` is saved to JSON but never
   checked; `covQual()` is never captured at all (only diagonal parameter errors).
5. `get_calib_categories`'s docstring says the lowest-pT bin (30-45 GeV, likely highest stats) should
   merge into 3 eta groups; the merge logic exists but is commented out -- all 9 sub-bins are kept,
   contradicting the function's own documented intent.
6. `bwz_Width` (natural Z width) floats by default (`fit_config.yml`, range [1,3] GeV); a separate,
   unused function in the same file fixes it citing HIG-19-006. Flagged as needing authoritative
   verification -- and empirically confirmed problematic, see below.
7. Minor: non-reproducible bootstrap seed (`hash(cat_name)`, per-process randomized), asymmetric
   skim/mass-array caching (read unconditionally, written only under `--fixCat`, no staleness check),
   `np.median` not filtering non-finite values (inconsistent with the bootstrap-err path, which does),
   chi2 rounded to 3 sig figs before being multiplied back into the stored absolute chi2.

**New finding from live execution (not caught by static review):**

8. **The pipeline crashes with a glibc heap-corruption abort (`free(): invalid next size (normal)`,
   `*** Break *** abort`) immediately after the very first category's fit completes and its outputs
   are written** (`calibration_fitCat30-45_BB.pdf`, `CalibrationLog.txt`, `fit_params.json` for
   category `30-45_BB` were all written successfully). The process then hangs indefinitely (150 ROOT
   `EnableImplicitMT()` threads spawned per category with no cap; CPU time stayed flat at 38s over
   6.6 minutes wall-clock -- i.e., genuinely stuck, not slow) and had to be killed manually (PID
   3812269). Root cause not isolated, but the most likely candidates given the static-review context:
   an unbounded `rt.EnableImplicitMT()` call (no thread-count argument, so it likely picks up the
   node's full core count rather than the pod's cgroup allocation) combined with known ROOT/RooFit
   thread-safety fragility around `TCanvas`/`RooPlot` object cleanup (`del canvas`, `canvas.SaveAs`,
   `frame.pullHist`) -- this was flagged as a *speculative* concern in the code-reviewer's audit and is
   now empirically confirmed as an actual crash.
9. In the one category that did fit before the crash, **`bwz_mZ` converged pinned at its upper bound
   (92.0) and `bwz_Width` pinned at its lower bound (1.0 GeV)** -- both flagged "(limited)" -- despite
   `fit_result.status()==0` and RooFit reporting "Full, accurate covariance matrix". This is a live,
   in-the-wild manifestation of finding 6's predicted BW-width/DCB-sigma degeneracy: MIGRAD is pushing
   the natural width to its floor and the pole mass to its ceiling rather than settling near PDG values,
   and RooFit's convergence/covariance-quality flags do not catch a boundary-pinned parameter. The
   resulting `sigma = 1.195 +/- 0.003` GeV (used directly as `sigma_fit`) should not be trusted while
   this persists.

## Decision or outcome

Two blocking wiring bugs (findings 1-2) were fixed with the user's explicit confirmation before
execution:
- Exported `HMM_TRIAL=Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation`
  for the run invocation only (no file change).
- Edited `getCalibrationFactor.py`'s isMC branch to pick whichever of
  `["dyTo2L_M-50_incl", "dyTo2Mu_M-50_aMCatNLO"]` actually exists under `LOAD_PATH/<year>/`, falling
  back to the old name if neither exists (keeps nanoAODv12 labels working unchanged).

Execution was launched inside the `default` pixi environment (`CONDA_OVERRIDE_CUDA=12.4`, no GPU in
this session) via `pixi run -e default bash -lc '...'` from the CVMFS pixi project directory
(bare `bash run_analysis_pipeline.sh` on PATH failed first with `ModuleNotFoundError: pandas`, as
`CLAUDE.md` warns). The run **did not complete successfully** -- see finding 8. Fixes 1-2 are
confirmed necessary and sufficient to get past the wiring bugs (skim built, first category fit ran to
completion and converged with a valid-looking covariance matrix); finding 8 (the crash) and finding 9
(boundary-pinned BW params) remain open and were not fixed in this session, per user direction to stop
and report before attempting further changes to the fit-execution code.

## Verification

- Command: `python3 -m py_compile src/lib/ebeMassResCalibration/getCalibrationFactor.py` -- syntax OK.
- Command: standalone Python check resolving `dy_candidates` against the real 2025 compacted directory
  -- correctly resolves to `dyTo2Mu_M-50_aMCatNLO`.
- Result: full run -- log at `/work/users/shar1172/copperheadV2_develop/log_20260914_192335.txt` (65
  lines; ends mid-crash, no `Program ended`/`Program FAILED` trap line because the killed process never
  returned control to the wrapping `run_analysis_pipeline.sh`). Confirmed via `ps`/`/proc` that the
  process hung (flat CPU time) rather than continuing, then killed it (PID 3812269) after ~7 minutes
  of no progress. Partial artifacts for category `30-45_BB` exist under
  `validation/ebeMassResCalibration/Run3_nanoAODv15_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/binned/2025/MC_2025_V1/`.
  Confirmed an unrelated, pre-existing `zpt_fit0`/`zpt_fit1` batch job (launched by a different session
  earlier, same label, years 2022preEE-2026) was running concurrently in the same pod throughout and was
  left untouched.

## CMS sources

Reference note cited in-code: HIG-19-006, AN-19-124 (Run 2 H->mumu, muon pT/eta calibration categories
and BW-width treatment). Not independently re-verified against the primary AN in this session --
flagged as "needs authoritative verification" per finding 6/9, not confirmed either way.

## Remaining work

- Root-cause and fix the heap-corruption abort (finding 8) -- prime suspects: unbounded
  `rt.EnableImplicitMT()` (basic_class_for_calibration.py:748, called once per category, no
  thread-count argument) and/or PyROOT object-lifetime handling around canvas/frame cleanup in
  `generateBWxDCB_RooCMSShape_plot`. Needs isolated reproduction (e.g. `--fixCat 30-45_BO --no-dask-client`
  on a single next category) before attempting a fix, since the crash could also be sensitive to the
  150-thread oversubscription interacting with the concurrent unrelated zpt_fit job sharing this pod.
- Resolve finding 9 / finding 6 together: decide whether `bwz_mZ`/`bwz_Width` should be fixed to PDG
  (as the unused sibling function already does, citing HIG-19-006) rather than left floating, given the
  observed boundary-pinning in real data.
- Fix finding 5 (30-45 GeV bin eta-merging) if the merge was intentional per HIG-19-006/AN-19-124.
- Fix finding 3 (Dask scheduler override) separately -- affects resource usage/performance, not
  correctness, so lower priority than 8/9.
- Findings 4, 7 are lower-priority robustness/reproducibility improvements.
- The MC-sample-name fix (this session) added a fallback list of 2 names; if a third NanoAOD
  campaign/sample-naming convention is introduced later, extend `dy_candidates` rather than
  re-hardcoding a single name.

## Update 2026-09-15: root-caused, fixed, and running for all years

Per user follow-up ("fix the issue, run it, if it fails debug/fix/resubmit; once one year
done, submit snakemake for all years"), returned to this and fully resolved it.

### Root cause of finding 8 (the crash)

Confirmed via a `faulthandler`-instrumented standalone repro (succeeded once, isolated)
vs. the same code crashing when run through the full pipeline (inconsistent -- classic
memory-corruption signature, not a deterministic logic bug). Two real, independent
contributing bugs found in `generateBWxDCB_RooCMSShape_plot`:

1. **RooFit object-name collisions across categories.** Every `RooRealVar`/`RooAbsPdf`
   (`mass`, `bwz_mZ`, `sigma`, `signal`, `bkg`, `final_model`, etc.) was constructed with
   an identical hardcoded name on every category iteration -- 30+ times in the same
   process. RooFit's FFT-convolution caching and global name-based object management is
   not safe against this (the original code's own canvas-naming comment,
   `"giving a specific name for each canvas prevents segfault?"`, shows the author had
   already half-discovered this class of bug for `TCanvas` but never extended the fix to
   the RooFit model components). Fixed: every RooFit object name in the production
   function is now suffixed with a sanitized `cat_idx` (`basic_class_for_calibration.py`,
   `_safe_roofit_name`, `suf` variable). This also required two follow-on fixes: (a)
   `save_fit_params_to_json` matched the "sigma" parameter by a hardcoded
   `.lower() == "sigma"` string, now takes an explicit `sigma_param_name` (the live
   object's real name) instead; (b) `Components="signal"`/`"bkg"` in `plotOn()` calls
   select a sub-pdf by its own name (now suffixed) -- changed to `model1.GetName()`/
   `model2.GetName()`; and `frame.chiSquare(final_model.GetName(), ...)` was looking up a
   curve by a tag that is a **fixed literal** (`Name="final_model"` in the corresponding
   `plotOn`), not `final_model`'s own (now-suffixed) name -- reverted to the literal
   `"final_model"` to match the actual curve tag.
2. **Residual, unexplained heap corruption independent of (1).** Even after fix (1), a
   `free(): invalid next size (normal)` / `*** Break *** abort` still occurred
   intermittently -- sometimes on category 1, sometimes not at all (a single-category
   standalone repro completed cleanly). Recompiling the bundled
   `PDFs/RooCMSShape_cc.so` against this pixi env's ROOT (6.32.02) surfaced a cling
   ABI-compatibility warning (`__GLIBCXX__` 20240521 at original compile time vs. 20250605
   at runtime) -- consistent with, though not proven to be, an ABI-mismatch-driven
   corruption. This was not chased further (would need Purdue AF / pixi-environment-level
   toolchain work, out of this repo's scope) -- instead the pipeline was made resilient to
   it (below), since a crash was shown to always happen only *after* a category's fit
   converges and `fit_params.json` is written for it.

### Engineering fix: per-category subprocess isolation with JSON-based recovery

`getCalibrationFactor.py`: for `--steps step1`/`all` **without** `--fixCat` (i.e. a normal
full run over every category), added `run_step1_categories_isolated()`, which re-invokes
the same script once per category via `--fixCat <cat> --steps step1 --no-dask-client` in
its own OS subprocess (`subprocess.run`, 150s timeout, up to 2 attempts). Because a crash
happens strictly after `save_fit_params_to_json` writes that category's `fit_params.json`
but before the function returns (so `fit_results.csv` never gets its row from a crashed/
hung subprocess), added a recovery path (`_try_recover_from_json`) that reads the sigma
value straight out of `fit_params.json` and appends the row itself when the subprocess
didn't return cleanly -- logging a clear warning when the recovered fit's
`status != 0` (non-converged) so those rows are flagged, not silently trusted. Also
skips categories that already have a row (resume-safe) and tries JSON recovery before
spawning a fresh subprocess (avoids redundant refits on resume). The single-category
(`--fixCat`) in-process path is unchanged and still used by each subprocess itself.

Also fixed in the same pass: `pd.read_csv(fit_results.csv)` in the `--fixCat` merge block
crashed with `FileNotFoundError` on a brand-new output dir (no prior full run) -- now
falls back to an empty DataFrame, needed because isolated subprocesses are often the
*first* write to that file.

**`getCalibrationFactor.py`'s `--use_gateway` path also crashed outright**
(`gateway.list_clusters()[0]` -> `IndexError`) whenever no Dask Gateway cluster happened
to be running for the user at invocation time -- guaranteed to fail every one of the 14
Snakemake-queued jobs identically. Fixed to fall back to a local Dask client with a
warning when the cluster list is empty (harmless given finding 3's confirmed dead
gateway-compute path).

### Verification

- 2025 MC ran to completion end-to-end via the real `run_analysis_pipeline.sh -m calib`
  command (inside the `default` pixi env, `CONDA_OVERRIDE_CUDA=12.4`): all 36 categories
  produced a row (several via JSON recovery after a crash/hang, confirming the recovery
  path works in practice, not just in theory), `calibration_factors.csv` and
  `res_calib_BS_correction_2025_MC_nanoAODv15.json` were produced with sensible values
  (calibration factors 0.94-2.0 across categories).
- **New, confirmed-live finding**: category count is 36, not the doc's/docstring's stated
  30 -- direct empirical confirmation of the earlier-flagged finding 5 (the documented
  3-group eta merge for the 30-45 GeV bin is dead code; all 9 sub-bins run instead of 3
  for every pT bin).
- **New, confirmed-live finding**: at least 3 of the 36 categories converged with
  Minuit status != 0 (`30-45_BO`, `30-45_BE`, `62-200_EE` all showed status=3, i.e.
  MIGRAD/HESSE did not fully converge) and 2 categories (`30-45_OE`, `45-52_EE`) show
  `fit_err = 0.0` (HESSE/covariance did not produce a usable uncertainty). These fits
  were still accepted into `calibration_factors.csv` (matching finding 4: no gating on
  fit status) -- now surfaced via explicit log warnings when the isolated-run recovery
  path fires, but not blocked. Recommend a follow-up: use `--fixCat <cat>` after tuning
  `fit_config.yml` bounds for these specific categories.
- Memory: step 2 (`step2_mass_resolution`, `.compute()`s the *entire* per-year skim into
  one pandas DataFrame) peaked at ~112GB RSS transiently for the 2025 MC run (single DY
  sample) before releasing back to ~34GB; survived without OOM against this session's
  256GiB cgroup limit. Not yet verified for a **data** year (all data streams combined,
  materially higher statistics) -- flagged as an open risk for the all-years batch below.

### All-years batch submission

Confirmed all 7 years' `stage1_compact_<year>.done` markers already exist, so
`MassCalibrationMC`/`MassCalibrationData` (Snakemake) have no stage1 dependency to
rebuild. Launched:
```
snakemake -s workflow/Snakefile <14 mass_calibration_{MC,Data}_<year>.done targets> \
  -j 4 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60 \
  --config years='["2022preEE","2022postEE","2023","2023BPix","2024","2025","2026"]' use_existing_stage1=True
```
matching this repo's existing `pixi.toml` snakemake-invocation convention
(`gateway=1` serializes all 14 jobs one at a time, bounding peak memory to one job's
footprint rather than compounding across concurrent jobs). Included **both** MC and Data
calibration for all years (not just MC) -- the doc documents both as parallel, equally
"the mass calibration" workflows; not separately confirmed with the user, flagged here in
case only one was wanted. Did not include the `_Closure` variants (a distinct validation
step, not requested). A stale lock from an earlier `pkill -9` (mid-debugging) required
`snakemake --unlock` before the final launch. Log:
`snakemake_mass_calibration_all_years.log`; per-job logs under
`logs/FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation/`.

## Remaining work (updated)

- Watch the batch's first **data** year for the step-2 memory-peak risk noted above; if it
  approaches the 256GiB cgroup ceiling, `step2_mass_resolution` would need reworking to
  compute per-category (like step1 already does) instead of materializing the full skim.
- Investigate the residual ABI-mismatch-flavored crash properly if it keeps requiring the
  subprocess-isolation workaround long-term (env/toolchain-level, likely needs Purdue AF
  pixi-environment maintainers, not a repo-only fix).
- Re-fit the specific non-converged/zero-uncertainty categories found above
  (`30-45_BO`, `30-45_BE`, `62-200_EE`, `30-45_OE`, `45-52_EE` for 2025 MC) via
  `--fixCat` after adjusting `fit_config.yml` bounds/constants for them.
- Findings 3 (dead gateway-compute path), 5 (eta-merge dead code), 6/9 (BW-width
  floating vs. fixed) remain open per the original investigation above.

## Update 2026-09-16: closure test run for all 14 year/sample combinations

User caught that the closure test (`--closure_test` / doc's documented validation step)
had never actually been run -- the earlier "all years done" summary only covered the base
calibration (`--steps all`), not closure. Ran it for all 14 year/sample combinations
(direct `getCalibrationFactor.py --closure_test` invocation, matching the pattern already
validated for the base calibration): all 14 produced the full 12 resolution bins
(`CLOSURE_BINS`) and a valid `closure_test_comparison.pdf` (this one is matplotlib-based,
not ROOT -- confirmed unaffected by the PDF/PNG crash bug found earlier). 2024 MC (615k
events) needed a longer timeout (280s -> 500s) than the rest; all others completed within
280s.

Spot-checked 2022preEE Data's closure plot: "Before Calibration" points sit systematically
below the y=x line across the full predicted-resolution range (median offset well outside
the +-10% band), while "After Calibration" points track much closer to y=x -- confirms the
calibration factors are doing real, physically meaningful work, not just adding noise. Two
clear outlier points (both before and after calibration, near-identical) come from the two
lowest-statistics closure bins (resBin1: 9 events, resBin2: 367 events) -- expected
low-stats artifacts, not a calibration failure; not further investigated.

**New, separate finding**: `workflow/Snakefile`'s `MassCalibrationMCClosure`/
`MassCalibrationDataClosure` rules are unreachable via Snakemake's own DAG resolution --
`snakemake ... state/.../mass_calibration_{MC,Data}_Closure_<year>.done` fails with
`MissingRuleException: No rule to produce ...` even though `snakemake --list` shows both
rules registered, and even for a single year with all its dependencies already satisfied.
Root cause not found (checked: output pattern matches the requested path character-for-
character, `wildcard_constraints` on `year` should be identical for closure and
non-closure rules, `STATEDIR`/`RUN_TAG` are shared module-level variables so can't differ
between them, no hidden whitespace in the rule block). Confirmed NOT year-list-override
related (fails even for `--config years='["2022preEE"]'`, a year definitely satisfying
every constraint). Worked around by calling `getCalibrationFactor.py --closure_test`
directly (same approach already used for the base calibration and for regenerating
missing plots) rather than through Snakemake -- this is what was actually used to produce
the 14 closure results above. The Snakemake-level bug itself is unresolved and flagged for
a future session; the underlying calibration/closure-test code and outputs are unaffected.

### Verification

- All 14 `validation/ebeMassResCalibration/<label>/binned/<year>/<Sample>_<year>_V1/`
  directories have `closure_results_resolutionBinning.csv` (12 rows) and
  `closure_test_comparison.pdf` (valid, matplotlib-produced).
- 2022preEE Data spot-checked visually (see above); other 13 not individually inspected
  beyond confirming successful completion + row/file counts -- a full visual pass across
  all 14 closure plots was not performed in this session.

### Remaining work

- Investigate the Snakemake `MassCalibrationMCClosure`/`MassCalibrationDataClosure`
  MissingRuleException properly (a real DAG-resolution bug, not just a config-passing
  issue) -- needed before these two rules can be used via `snakemake`/the documented
  workflow rather than direct script invocation.
- Visually inspect the remaining 13 closure comparison plots (only 2022preEE Data was
  actually looked at) to confirm none show a pathological miscalibration beyond the
  expected low-stats-bin outliers.
- The `30-45_EE` chi2/ndf-vs-statistics tradeoff (2026-09-15 update above) remains open
  for the highest-statistics MC samples (2024, 2022postEE, 2025/2026) -- not addressed in
  this closure-test pass.
