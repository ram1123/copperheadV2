# Some functionality

- When we commit it should ensure that each switches have similar structure and keys.

# Systematics issues found in the template audit (2026-10-05)

Found with `scripts/plot_template_systematics.py` on the `Oct04_2026_Syst` VBF templates of
`Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit`
(all three jj regions). Plots + `systematics_summary.csv` are under
`validation/stage3_templates/<label>/Oct04_2026_Syst_<region>/template_systematics/`.

## 1. EWK LHE scale weights are wrong (needs stage-1 rerun of EWK)

- **Symptom:** `LHERen_EWK` Up and Down are identical to nominal in every bin of all 36
  template files (6 years x SR/SB x 3 regions), so the nuisance does nothing in the fit.
  `LHEFac_EWK` does vary, but its weights look unphysical.
- **Evidence (stage-1, 2023 `ewk_mmjj_mll_105_160_mjj120`, first 5 files):**
  `wgt_LHERen_up/nominal` and `wgt_LHERen_down/nominal` are exactly 1.0 for every event;
  `wgt_LHEFac_up/nominal` ranges 0.61-12.8 and `wgt_LHEFac_down/nominal` 0.0-1.74
  (a muF variation is normally within ~0.5-2).
- **Suspected cause:** `lhe_weights()` in `src/corrections/evaluator.py` hardcodes the
  `LHEScaleWeight` positions: 9-entry (7/1/5/3), 8-entry (6/1/4/3) and, for more than 30
  entries, 34/5/24/15. The EWK sample probably has a different count or ordering, so the
  "renormalisation" indices point at the nominal weight and the "factorisation" ones at
  something else.
- **To do:** open one EWK NanoAOD file, check `nLHEScaleWeight` and the `LHEScaleWeight`
  branch title (which lists the muR/muF ordering); fix the index mapping per sample; rerun
  stage-1 for EWK, then stage-2/3 and the stats.

## 2. Muon variations in the sidebands: Up == Down != nominal

- **Symptom:** 55 (file, process) pairs of `mu_scale<year>` / `mu_resol<year>` in the SB
  (h-sidebands) templates have Up identical to Down, a 0% yield change, but a different
  bin-by-bin shape than nominal. Examples (`jj_both_central`): 2022preEE SB TT+ST
  `mu_scale2022preEE`, 2022preEE SB EWK `mu_resol2022preEE`, 2022postEE SB ggH
  `mu_scale2022postEE` / `mu_resol2022postEE`.
- **Why it is suspicious:** in the sidebands stage-2 pins every `dimuon_mass*` field to
  125 GeV before DNN scoring (`run_stage2_vbf.py`, `if region == "h-sidebands"` block), so a
  muon pT variation should only reach the DNN through the other dimuon/muon inputs, and Up
  and Down should then move the score in opposite directions -- not identically.
- **Likely cause:** a DNN input that is pinned or rebuilt for nominal but not for the varied
  muons (e.g. `dimuon_ebe_mass_res_rel` or another mass-derived feature computed from the
  unpinned varied mass), or a variation-specific feature column that is missing so both Up
  and Down fall back to the same non-nominal column.
- **To do:** for one sample/year, log `feature_name_for_variation()` resolutions for
  `mu_scale_up/down` in the SB and compare the DNN inputs event-by-event to nominal; fix the
  pinning or the feature mapping so Up/Down are consistent with nominal.

## 3. b-tag counts never change under JES/JER variations

- **Symptom:** in stage-1, `nBtagLoose_<variation>` and `nBtagMedium_<variation>` are
  identical to `nBtag*_nominal` for every JES/JER variation, including `Absolute_up` (2023
  `ttjets_dl` part65: 0.000 of events differ), while `njets_Absolute_up` differs in 2.7% of
  events. The VBF/ggH selection uses these counts for the b-jet veto via `varcol()` in
  `modules/selection.py`.
- **Why it matters:** if the b-tag jet pT threshold is applied to the varied jet pT, jets
  crossing it under JES should change the b-tag count, and with it the b-veto acceptance;
  freezing the count to nominal ignores that migration. Freezing it may be intentional
  (the b-tag SF and its uncertainties are defined on nominal jets).
- **To do:** find where stage-1 fills `nBtag*_<variation>` and confirm whether the count
  is deliberately taken from nominal jets. If not, compute it from the varied jet pT; if
  yes, document it here and in `docs/known_issues.md`.
- **Related:** the jet tagger-score variation columns (e.g. `jet1_btagPNetQvG_HF_up`) are
  filled with the -1 placeholder instead of the nominal score. Not used by the VBF DNN or
  the selection today, but anything that reads them for a variation would get -1.

## 4. Follow-ups from the `Oct04_2026_Syst_JESfix` rerun (2026-10-05)

- **Item 2 above, more evidence:** stage-2 worker logs for `mu_scale_up` warn that many
  varied DNN inputs do not exist and fall back (`dimuon_cos_theta_cs_mu_scale_up`,
  `dimuon_phi_cs_mu_scale_up`, `jet1_pt_mu_scale_up`, `jj_mass_mu_scale_up`, `rpt_mu_scale_up`, ...
  -- `feature_name_for_variation`, `modules/systematics.py:96`). Check which fallback column
  each resolves to; if Up and Down both fall back to the same column, that explains
  Up == Down. In the combined Run3 impacts `mu_scale2025_2026` and `mu_resol2025_2026` have
  exactly the same impact (0.089), which also looks like they share one non-varied input.
- **Residual Up == Down from float noise in other varied columns:** after the eta rounding
  fix, JES/JER sources with no effect in a region (e.g. `HF`, `jer3-6` in `jj_both_central`;
  `jer1`, `jer2`, `BBEC1` in `jj_both_fwd25`) still show Up == Down != nominal, now with only
  0.000% to -0.08% yield change. Cause: stage-1 recomputes the varied `jj_mass`, `jj_dEta`, ...
  with ~1e-8 relative noise, which moves a few DNN scores across a bin edge. Fix options:
  round the varied DNN inputs in stage-2, or have stage-1 copy the nominal value when a jet
  is not changed by the variation.
- **MC statistics dominate the systematic loss:** the top impacts in the combined Run3 fit
  (r=1) are autoMCStats `prop_bin*` parameters of the high-score SR bins of
  `jj_one_fwd25_one_central` and `jj_both_fwd25` in 2024 and 2025_2026 (impacts 0.25 down
  to 0.08), ahead of every named systematic. Removing the double-counted JES `Total` moved
  the combined Run3 significance only from 1.628 to 1.620 sigma. More MC in those bins, or a
  coarser binning at high DNN score per region, is where the sensitivity is.

## 5. DNN bin scan: account for background MC statistics (2026-10-05)

Context: `MVA_training/VBF_run3/scan_bins_for_dnn.py --stage2-scores` on the
`Oct05_2026_ScoreDump` stage-2 dumps (one per jj region). Reports and plots:
`validation/dnn_binning_scan/score_<label>_Oct05_2026_ScoreDump_<region>_NoSyst/`.

- **Problem:** the scan maximises the plain Asimov Z, which ignores MC statistics, and
  its guards (raw S+B >= 15, weighted S >= 0.05, weighted B >= 0.05) do not bound them.
  In the fit, autoMCStats is the largest uncertainty after data statistics
  (r = 1 +0.656/-0.627; MCStat +0.432/-0.412 vs Stat +0.461/-0.445, Syst +0.125/-0.111).
- **A relative-error cut (e.g. "B MC rel. error <= 30%") is the wrong guard.** What hurts the
  fit is the MC uncertainty compared with the data's Poisson fluctuation, sigma_MC/sqrt(B),
  not sigma_MC/B. Scanned-bin examples:
  `jj_one_fwd25_one_central` bin 9: B = 1334, rel. error 4%, sigma_MC/sqrt(B) = 1.53;
  bin 13: B = 135, 10%, 1.19 (MC-dominated, and the top MCStat impacts in the fit);
  `jj_both_fwd25` bin 17: B = 0.74, 75%, 0.64 (fairly harmless). A 30% cut would remove
  the harmless bin and keep the MC-dominated ones.
- **Rebinning cannot fix MC-dominated bins:** sigma_MC^2 / B = sum(w^2)/sum(w), i.e. the
  typical background MC event weight. Merging bins only averages it. In
  `jj_one_fwd25_one_central` it is 1.2-1.5 in all top bins, so one background MC event stands
  for ~1.5-2.3 data events. Only more MC (smaller per-event weights) helps there; the
  LumiSplit per-year MC split raises the weights. Rebinning does fix empty/negative bins
  and single-event "golden" bins (e.g. current `jj_both_fwd25` bin 2.57-2.87: S = 0.079,
  B = 0.0012 +- 5600%, fake Asimov Z = 0.72).
- **To do:**
  1. Use the Asimov significance with a background uncertainty as the scan's figure of
     merit, with sigma_B = sigma_MC per bin (Cowan formula, general statistics, not a POG
     recommendation):
     Z = sqrt(2[(s+b) ln((s+b)(b+s2)/(b^2+(s+b)s2)) - (b^2/s2) ln(1 + s2 s/(b(b+s2)))]),
     s2 = sigma_MC^2 = sum(w^2) of the background in the bin. Each bin's MC uncertainty is
     independent, as with autoMCStats, so per-bin Z^2 still add. The per-event weights are
     already in the score dumps, so sum(w^2) per bin is available.
  2. Optional hard guard with a meaning: sigma_MC/sqrt(B) <= k (k = 1: MC uncertainty no
     larger than the data's), instead of a relative-error threshold.
  3. Find which background sample dominates sum(w^2) in the top bins of each region
     (likely a DY sample) to judge whether requesting more MC is worthwhile.

## 6. Signal cross section x BR was ~20% too high (2026-10-08)

- **Problem:** every Run-3 signal entry in `configs/datasets/dataset_nanoAODv1{2,5}_run3.yaml`
  used BR(H->mumu) = 2.6e-4 (ggH 51.96 x 2.6e-4 = 0.0135096 pb; VBF 4.067 x 2.6e-4 = 0.00105742 pb).
  LHCHXSWG gives 2.1542e-4 at mH = 125.38. The stage-1 `separate_wgt_xsec` column confirms the old
  values were applied (checked: 2024 and 2023BPix ggH, 2024 VBF, 2022preEE VBF dipole).
- **Done 2026-10-08:** dataset YAMLs (the only cross-section source; the obsolete `configs/parameters/cross_sections.yaml` was deleted) now use
  ggH 0.0110834 pb (51.45 x 2.1542e-4) and VBF 0.00088319 pb (4.0998 x 2.1542e-4), from LHCHXSWG1
  `crosssections` @cbb6cf65 (R5 v1.1.3 + YR4 BR). Sources and uncertainties:
  `.claude/skills/stats/references/signal-xsec-br.md`.
- **Interim:** `run_stage3_vbf.py --signal_xsec_rescale` (pipeline: `SIGNAL_XSEC_RESCALE=1`) scales
  the signal templates by 0.8204 (ggH) / 0.8352 (VBF) and writes `stage3_datacards_<postfix>_SigXS`.
  Use it only on stage-1 made before the fix, or the signal gets scaled twice.
- **To do:**
  1. Rerun stage-1 for the signal samples only (`ggh_powhegPS`, `vbf_powheg`, `vbf_powheg_dipole`;
     reset them with `scripts/reset_stage1_samples.sh`), all Run-3 years, then compact, stage-2 and
     stage-3 **without** `--signal_xsec_rescale`. Check `separate_wgt_xsec` is 0.0110834 / 0.00088319.
  2. Then remove `SIGNAL_XSEC_BR_RESCALE` / `--signal_xsec_rescale` and the `KNOWN_ISSUE` in `configs/trials.yml`.
  3. ~~Datacard: add `BR_hmm` lnN 0.983/1.017 (ggH+qqH) and `QCDscale_ggH` lnN 0.930/1.040, both correlated
     across years~~ done 2026-10-08 (`stage3/make_datacards.py` `signal_theory_lnN`).
