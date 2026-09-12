# Muon-selection implementation review (read-only)

- Date: 2026-08-31
- Type: Investigation
- Status: Open
- Applicable era: all supported (Run2: 2016preVFP/2016postVFP/2017/2018 + *_RERECO; Run3: 2022preEE/2022postEE/2023/2023BPix/2024/2025/2026)
- NanoAOD campaign: v9 (Run2 legacy), custom v12 (Run2 + Run3 2022/2023, carries `Muon_bsConstrainedPt/PtErr/Chi2`), v15 (Run2 + Run3 2024+)
- Relevant files:
  - `src/copperhead_processor.py` muon block ~L900-1330, `build_muon_kinematic_variation_block` ~L234-270, `get_mass_resolution` ~L2569-2600
  - `src/corrections/{rochester,MuonScaRe,fsr_recovery,geofit,muon_sf}.py`
  - `src/lib/get_parameters.py`, `modules/classify_year.py`
  - `configs/parameters/{muon,trigger,switches}.yaml`, `configs/parameters/{SF_filelist,correction_filelist}.yaml`
  - Reference compared against: `.claude/skills/cms-object-guidelines/references/muons.md`

## Question

Compare the existing muon-selection implementation with the stored muon guidance;
separate confirmed defects from items needing authoritative CMS verification; and
list what the local reference does not cover. No files modified.

## Evidence

Physics-reviewer and code-reviewer ran independently (read-only). Main agent
spot-checked the load-bearing claims:

- `grep` confirms `muon_trigmatch_id` and `muon_trigmatch_iso` appear only in
  `configs/parameters/muon.yaml` and `muons.md` — never in any `.py`.
- `muon_trigmatch_pt` is read into `pt_threshold` and logged
  (`copperhead_processor.py:1015-1017`) but the `TrigObj.pt >= pt_threshold` term is
  commented out; `trigger_cands_filter = pass_id & pass_filterbit` with
  `pass_filterbit = (events.TrigObj.filterBits & 8) > 0` hardcoded (`:1033-1034`).
- `apply_geofit` is never called from the processor (only its def + comments).
- `src/lib/get_parameters.py:40-42` sets top-level `do_roccor/do_fsr/do_geofit=True`;
  processor reads `self.config["switches"][...]` instead → those three lines are inert.
- `modules/classify_year.py`: `is_run2()` = `year.startswith("2016") or year in ("2017","2018")`
  → `is_run2("2017_RERECO")` / `is_run2("2018_RERECO")` return `False` → routed to
  `apply_KitMuScaleRe_Run3` (Run3 JSON) with a Run2 Rochester text payload.
- `src/corrections/MuonScaRe.py:247-266` `filter_boundaries`: `outside_bounds = (pt<26)|(pt>200)`
  → `pt_corr` reset to `pt` for those muons; only NaN/None otherwise guarded (no `isfinite`).
  `if n_pt_outside > 0:` on `ak.sum(...)` forces an eager reduction inside the lazy graph.
- `copperhead_processor.py:914` `doing_BS_correction = switches["do_beamConstraint"]`
  (not gated by branch presence); passed to `get_mass_resolution` (`:1320`, `:2586-2594`)
  which then loads `BS_res_calib_path` and applies `BS_ebe_mass_res_calibration`.
- `configs/parameters/SF_filelist.yaml:80` marks the 2017 muon trigger SF
  `FIXME: input binning error`; `src/corrections/muon_sf.py:9-42` `_evaluate_nom_up_down`
  mirrors the opposite variation for the whole chunk on failure.

## Findings

### Confirmed (provable from code + stored guidance)

1. **Cutflow rows do not match the applied per-muon mask.**
   - `muon_pT_roch` cutflow uses `pt_roch >= cut`; the mask uses `pt_raw > cut`
     (`copperhead_processor.py:951` vs `:967`).
   - `muon_eta` cutflow uses `<= cut`; mask uses `< cut` (`:952` vs `:968`).
   - `muon_selection` cutflow row is registered (`:971`) before the isolation term is
     AND-ed into the mask (`:983`), so it omits iso.
   Cutflow / N-1 efficiencies for the muon steps therefore do not correspond to the
   selection actually applied.

2. **Trigger matching is much looser than config and guidance imply.** Only `ΔR < 0.4`
   is applied. `muon_trigmatch_id` (tightId), `muon_trigmatch_iso` (0.15) are never
   referenced; the trig-object `muon_trigmatch_pt` cut is read/logged but not applied.
   `filterBits & 8` is hardcoded for every era/HLT path (the per-HLT bit loop and the
   2016 `IsoMu24` leg are commented out). The matched muon is only required to pass the
   base selection + `pt_roch >= 26/29`.

3. **AN-19-124 baseline impact-parameter cuts (`|dxy|<0.05`, `|dz|<0.10`, `SIP3D<8`)
   are not applied** anywhere — the branches are written only as output columns.

4. **`is_run2()` misclassifies `2017_RERECO` / `2018_RERECO` as Run 3**, sending them to
   the Run 3 KIT JSON path with a Run 2 Rochester text payload. Latent unless a RERECO
   sample is processed; `2016_RERECO` works only by the `startswith("2016")` match.

5. **Dead / misleading correction config.** `get_parameters.py:40-42` top-level
   `do_roccor/do_fsr/do_geofit=True` are shadowed by the `switches` block and unused;
   `do_geofit` is `false` everywhere and `apply_geofit` is dead code.

6. **2017 muon-trigger systematic is chunk-biased.** `_evaluate_nom_up_down` falls back
   to a symmetric mirror for the entire chunk when any `(eta,pt)` bin lacks
   `systup`/`systdown` — triggered by the payload flagged `FIXME: input binning error`.

7. **pT thresholds are cut on `pt_raw`** (post-BS, pre-Rochester/KIT), with the author's
   own `FIXME: Why pt_raw`. Internally inconsistent (BS in, Rochester out) and not the
   MUO convention of cutting on the fully corrected pt. ~per-mille acceptance effect,
   correlated with the mass-scale systematic. Physics-requirement call.

8. **Run 3 KIT correction has a 26 GeV floor.** `MuonScaRe.filter_boundaries` reverts
   muons with `pt < 26` (or `> 200`) to uncorrected pt, but the subleading muon is
   accepted from `muon_pt_cut = 20`. So 20-26 GeV muons receive no scale/resolution
   correction (and no KIT systematic) in Run 3; Run 2 `apply_roccor` has no such floor
   → Run 2 / Run 3 treat the same event differently.

9. **`doing_BS_correction` is decoupled from whether the BS branch exists.** Set from the
   switch alone; the EBE dimuon-mass-resolution BS calibration is then applied even when
   `Muon_bsConstrainedPt` was never used (e.g. Run 2 NanoAODv9). Impact depends on
   whether `BS_res_calib_path` is populated for that path; in practice Run 2 is run on
   custom v12, which carries the branch, so this bites only the v9 config.

10. **Systematic coverage is era-asymmetric.** Run 2 emits `mu_roccor_up/down` only;
    Run 3 emits `mu_scale_*` + `mu_resol_*` only. The BS-constraint replacement has no
    scale/resolution nuisance; FSR recovery has no systematic and no data/MC split.

### Confirmed, lower severity / robustness

- `MuonScaRe.pt_scale`: `pt_corr = 1/(m/pt + charge*a)` has no `isfinite` guard;
  `inf` from a vanishing denominator with in-range `pt` is not caught.
- `filter_boundaries`: `if n_pt_outside > 0:` / `if n_nan > 0:` on `ak.sum` force an
  eager compute inside the dask-awkward graph (perf / backend-determinism hazard).
- Run 3 MC scale/resolution *variations* take the resolution-smeared `ptcorr` as their
  base input rather than `ptscalecorr` (in-code `TODO` at `rochester.py:125`).
- Three different "leading muon" sort keys: `pt_roch` (trigger-match branch),
  `pt_raw` (fallback branch), `pt_fsr` (dimuon build) — FSR can reorder the top two.
- `ΔR < 0.4` trigger-match cone is loose vs the usual ~0.1 (compounds finding 2).
- `applied_fsr` is assigned in both branches and never read (dead variable).
- `build_muon_kinematic_variation_block` scales `pt` but not `ptErr`, so
  `dimuon_ebe_mass_res_*` in the *variation* columns uses a varied denominator with an
  unvaried numerator.
- `fsr_recoveryV1` isolation denominator uses the muon+photon combined pt, not the
  muon pt (tiny, near-collinear, but differs from the textbook definition).

### Requires authoritative CMS verification ([Verify])

- `mediumId` + `pfRelIso04_all < 0.25` = current MUO recommendation for H→µµ, per era
  (especially each Run 3 era).
- Whether the AN-19-124 baseline actually requires the IP cuts, and tightId + iso<0.15
  on the trigger-matched muon.
- Reco/tracking SF requirement per era (currently not applied).
- Run 3 momentum-calibration payload versions: Summer22/22EE/23/23BPix JSONs, the
  2024 `Summer24` local copy "from Hyeon", the 2025 VXBS JSON; 2026 is a placeholder
  reusing 2025. Also the MuonScaRe nominal+variation call convention vs upstream KIT.
- `TrigObj.filterBits` bit semantics per NanoAOD campaign (v9 / custom v12 / v15) for
  the `IsoMu24` / `IsoMu27` / `IsoTkMu24` legs — needed to judge the hardcoded `& 8`.
- Whether the custom NanoAODv15 (2024+) carries `Muon_bsConstrainedPt/PtErr/Chi2`
  (determines if `do_beamConstraint=true` for 2024-2026 is active or a silent no-op).
- Muon SF input conventions per era `muon_Z.json`: signed `eta` vs `abseta`; SFs
  derived vs corrected or uncorrected pt (code passes signed `eta_raw` / `pt_raw`).
- Trigger-SF denominator (`CutBasedIdTight_and_PFIsoTight` Run2 /
  `...Medium...` Run3) vs the looser offline selection the SF multiplies.
- MUO "LooseRelIso" (Run2) / "LoosePFIso" (Run3) numeric WP == `pfRelIso04_all < 0.25`?
- Is the MuonScaRe `[26, 200]` GeV restriction mandatory for this selection?
- FSR-recovery numeric cuts (`relIso03<1.8`, `dROverEt2<0.012`, pT ratio < 0.4,
  `|eta|<2.4`) vs an authoritative recipe; cross-year nuisance correlation scheme.

### Gaps in the local muon reference (`muons.md`)

- Numeric MUO ID/iso WP recommendation per era; reco/tracking SF requirement per era.
- Run 3 momentum-calibration prescription and correctionlib payload versions.
- `TrigObj.filterBits` semantics per NanoAOD campaign and per HLT leg.
- FSR-recovery numeric selection and any authoritative recipe for it.
- Whether custom NanoAODv15 (2024+) carries the `Muon_bsConstrained*` branches.
- Muon SF input conventions (signed eta vs abseta; corrected vs raw pt) per era.
- The EBE dimuon mass-resolution calibration formula, its inputs, and Z→µµ validation.
- `_RERECO` era handling: Run2/Run3 classification, missing `muSFFileList` keys,
  `dummy` BS-calibration entries.
- Cross-year correlation scheme for muon scale / resolution / ID / iso / trigger
  nuisances in a combined Run2+Run3 fit.
- That `apply_geofit` is currently dead code and that `get_parameters.py`'s top-level
  `do_*` flags are shadowed by the `switches` block.
- Trigger-SF treatment for two-muon events (leading-muon-only vs OR-of-legs).

## Decision or outcome

Review only — no code changed. The two reviews are mutually consistent; no
disagreements to resolve. Highest-priority confirmed items for follow-up:
(1) cutflow/mask consistency, (2) trigger-matching tightening + filterBits, and
(4) the `is_run2` RERECO misclassification. Items 7 and 8 need a physics decision.

## Verification

- Command: `grep`-based spot checks (see Evidence); no tests run (read-only review).
- Result: all four spot-checked claims confirmed against the source.
- Not run: stage-1 sync/regression (`scripts/update_sync_references.sh`) — would be the
  way to quantify findings 1, 2, 7, 8 if any fix is attempted.

## CMS sources

- Stored: `.claude/skills/cms-object-guidelines/references/muons.md` (local review
  2026-08-31; AN-19-124 §3.2-3.3, CERN-THESIS-2021-201, HIG-19-006 twiki — none in repo).
- Pending authoritative: MUO POG Run 2 UL and Run 3 muon ID/iso/SF recommendation
  TWikis; MUO scale-smearing (KIT `MuonScaRe`) GitLab; NanoAOD `TrigObj.filterBits`
  documentation per campaign. Current POG recommendation: not verified this session.

## Remaining work

- Physics decision on cut reference (`pt_raw` vs corrected pt) and the MuonScaRe
  `[26,200]` floor for sub-26 GeV subleading muons.
- Authoritative verification of every `[Verify]` item above.
- If any fix is made: rerun the stage-1 sync test and regenerate `test/reference/`.
- Extend `muons.md` to close the listed gaps.
