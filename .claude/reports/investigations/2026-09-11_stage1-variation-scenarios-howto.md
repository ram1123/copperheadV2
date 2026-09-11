# How to run stage-1 with multiple parameter variations (muon ID/iso, jet settings, ...)

**Date:** 2026-09-11
**Type:** Investigation (read-only; "what's the best way" question via `/coordinate`)
**Scope:** `configs/parameters/*.yaml`, `src/lib/get_parameters.py`, `scripts/apply_switches_profile.py`,
`scripts/run_scenario.sh`, `run_analysis_pipeline.sh`/`common_workflow.sh`.

## How stage-1 parameters actually resolve (the mechanism any "variation" hooks into)

`src/lib/get_parameters.py::getParametersForYr(parameter_path, year)` globs **every**
`configs/parameters/*.yaml` file, OmegaConf-merges them into one dict, then for each top-level key
does `val[year]` (special-cased for `cross_sections`, `jec`, `switches`). So `muon.yaml`, `jet.yaml`,
`electron.yaml`, `trigger.yaml`, `misc.yaml`, `switches.yaml` etc. are **all** the same shape --
`<param_name>: {<year>: <value>}` -- and are all merged the same way. Confirmed `muon.yaml` (muon ID,
iso, pt/eta cuts, trigger-match settings) has the identical indentation/structure as `switches.yaml`
(4-space key line, 8-space `'year': value` lines, optional quotes). "Run stage-1 with a different
muon ID" == "change `muon_id: {<year>: mediumId}` to `tightId` for the years you're testing, in
`configs/parameters/muon.yaml`, before that stage-1 run starts" -- same mechanism for jet settings in
`jet.yaml` / `switches.yaml`.

## Existing tooling for this (already partially built, found this session)

1. **`scripts/apply_switches_profile.py`** -- safe, comment-preserving TEXT editor (not a YAML
   round-trip -- `switches.yaml`'s inline comments would be lost/reformatted by PyYAML or even
   ruamel.yaml) for exactly the `key: {year: value}` pattern. Dry-run by default (prints a unified
   diff), `--yes` to write, `--set key=value` for one-off overrides on top of a named profile,
   `--years` to restrict which year rows change. After writing, it re-parses old vs new YAML and
   refuses to write if anything outside the intended `(key, year)` set changed.
   **Gap found**: hardcoded to `SWITCHES_FILE = configs/parameters/switches.yaml`, and its post-write
   sanity check does `yaml.safe_load(...)["switches"]` -- assumes a top-level `switches:` wrapper key.
   `muon.yaml`/`jet.yaml` have NO such wrapper (keys are top-level), so this script cannot be pointed
   at them as-is. The line-matching regexes themselves (`KEY_LINE_RE`/`YEAR_LINE_RE`) don't care about
   the wrapper and would work unchanged on `muon.yaml`/`jet.yaml` -- generalizing needs only a
   `--file` argument plus making the wrapper-key sanity check optional/parameterized.
2. **`configs/parameters/switches_profiles/*.yaml`** -- named override sets already defined:
   `nominal_nozpt_jme_reco`, `nominal_nozpt_jvm_jets`, `nominal_zpt`, `pudnn_jetidvars`, `syst`.
   These only cover `switches.yaml` keys today (JME/PU/zpt/jet-ID-var toggles), not muon/jet
   object-definition parameters.
3. **`scripts/run_scenario.sh`** -- clearly intended as the one-command driver: apply a switches
   profile (Step 1), optionally reset+reprocess specific samples in place (Step 2), then invoke the
   Snakemake DAG under a scenario-specific `run_tag`/label (Step 3) so each variation's stage-1 output
   lands in its own directory. Its own header comment explains WHY the profile must be applied exactly
   once before the DAG starts (switches.yaml is one file shared by every parallel Snakemake rule --
   no safe way to toggle it per-rule mid-DAG).
   **Gap found**: Steps 2 and 3 (the `reset_stage1_samples.sh` call and the actual `snakemake`
   invocation) are **entirely commented out** in the current file. As it stands, running this script
   only executes Step 1 (apply/dry-run the switches profile) and then does nothing else -- it does
   NOT currently run stage-1. This looks like in-progress/uncommitted work (the file is untracked),
   not a maintained tool yet.
4. **`-l <label>` on `run_analysis_pipeline.sh`** (`common_workflow.sh: save_path="${save_root}/${label}"`)
   -- the actual mechanism that keeps two variations' stage-1 parquet/cutflow output from colliding.
   Every scenario this session (`..._OfficialRecomendation`, `..._OfficialRecomendation_Cutflow`, the
   `FilterEvents_Aug30_...` family) used a distinct, descriptive label for exactly this reason.

## Recommended workflow (works today with what exists, or after the two small gaps above are closed)

1. Identify which `configs/parameters/*.yaml` key(s) the variation changes (muon ID/iso -> `muon.yaml`;
   jet pT/JVM/horn-pT-cut/PUID -> `jet.yaml` + relevant `switches.yaml` flags) and which years.
2. Apply the change via `apply_switches_profile.py` (generalized with `--file`) against a named
   profile file, or `--set key=value` for a one-off -- get the dry-run diff, confirm it only touches
   the intended lines, then `--yes`.
3. Run stage-1 with a fresh, descriptive `-l`/`run_tag` for this variation (never re-run into an
   existing label's output unless intentionally reprocessing in place).
4. Because the parameter files are shared, mutable, single files: apply-profile-then-run must be
   treated as one atomic step per `run_scenario.sh`'s own header comment -- do not start a second
   variation's stage-1 run (which would flip the same file again) until the first has actually started
   reading the config for every chunk it will process (the AF's local Dask cluster reads it once per
   worker at startup, but a scenario spanning a long DAG needs the file stable for its whole run).
5. Repeat per scenario; compare using the tooling already built this session: `scripts/get_yields.py`
   (`--data-only`/`--data-glob`), `scripts/merge_cutflow_npz_file.py`, `scripts/compare_cutflow_json.py`,
   `scripts/compare_yield_csv.py`, `scripts/count_dimuon_mass_window.py`.

## Open question for the user (not yet actioned)

Whether to (a) generalize `apply_switches_profile.py` with a `--file`/wrapper-key option so it also
covers `muon.yaml`/`jet.yaml`, and (b) restore/finish `run_scenario.sh`'s commented-out Steps 2-3 so
it actually runs stage-1 end-to-end. Both are small, mechanical changes given the existing code, but
not made in this investigation (read-only per the `/coordinate` task definition) pending explicit
approval, since `run_scenario.sh` looks like someone's in-progress work that might already have a plan
for those sections.
