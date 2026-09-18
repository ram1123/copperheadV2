# Stage-1 correction / scale-factor / reweighting inventory — Run 3 (2024 & 2025)

**Date:** 2026-09-03
**Branch:** `Week_June10`
**Scope:** Run 3, data-taking years **2024, 2025 and 2026**. 2022/2023 and Run 2 excluded except
where a 2024–2026 path is only defined by falling through to shared code. NanoAODv15.
**Method:** static reading of code + YAML as they stand on `Week_June10`; no code executed.

---

## Runnable status of 2026 (read this first)

**2026 is mechanically runnable end-to-end but calibration-incomplete ("half scaffold").**

- Every per-year config key resolves for `"2026"` (nothing `KeyError`s), and
  `configs/datasets/dataset_nanoAODv15_run3.yaml:1135` has a fully populated 2026 block:
  **data eras A, B, D active** (`/Muon{0..3}/Run2026{A,B,D}-PromptReco-v1/NANOAOD`, `:1139-1193`),
  **era C excluded** (`data_C: skip_sample: True # low-PU era`, `:1167-1168`), golden JSON
  `Cert_Collisions2026_401624_403937_golden.json`, `total_lumi_pb: 25950.0` (A+B+D). **MC blocks
  are populated and un-skipped** but **every 2026 MC sample reuses the
  `RunIII2024Summer24NanoAODv15` (2024-conditions) files** — "2026-conditions MC not yet available"
  (`:1199-1201` etc.). So 2026 MC == 2025 MC == 2024 Summer24 simulation.
- **JME side is real for 2026:** JES payload/tags (`Summer24Prompt26_V1_MC/DATA`), JER tag
  (`Summer24Prompt26_RunBD_JRV1_MC`), jet-ID payload, jet-veto-map payload/tag
  (`Summer24Prompt26_RunBCD_V1`), and the `Collisions2026` lumi mask are genuine
  `Run3-26Prompt-Summer24-NanoAODv15` objects.
- **MUO + LUM side is placeholder for 2026:** muon Rochester/scale-smear, muon ID/Iso/Trig SF,
  pileup reweighting, and the EBE mass-resolution BS calibration all fall through to 2025 (or 2024)
  payloads, with explicit config comments saying the real `MUO/Run3-26Prompt-...` and `LUM/*2026*`
  payloads do not yet exist on cvmfs. 2026 physics output would be produced with 2025/2024 muon and
  pileup calibrations.
- All `switches.yaml` `"2026"` values are **identical to `"2025"`** (verified key-by-key). In
  particular `do_jet_horn_ptcut` is `false` for both 2025 and 2026 (only 2024 = `50`).

---

## 0. Scope / framework facts

- **Implemented Run 3 year strings.** `modules/classify_year.py:3` treats anything containing
  `22/23/24/25` as Run 3; `is_run3()` (`classify_year.py:22-24`) is "not Run 2".
  `configs/parameters/switches.yaml` and every `configs/parameters/*.yaml` define keys for
  `2022preEE, 2022postEE, 2023, 2023BPix, 2024, 2025, 2026`.
  `configs/datasets/dataset_nanoAODv15_run3.yaml` has real data+MC blocks for `"2024"` (`:2`),
  `"2025"` (`:573`) and `"2026"` (`:1135`). So **2024 and 2025 are fully wired; 2026 exists too**
  (data DAS paths + MC), but essentially every 2026 correction payload is a copy-of-2025
  placeholder (see §3). 2025 MC is itself the `RunIII2024Summer24NanoAODv15` campaign
  (e.g. `dataset_nanoAODv15_run3.yaml:657`), i.e. **2025 MC = 2024 Summer24 simulation**.
- **Config assembly.** `src/lib/get_parameters.py:16-39` merges *all* `configs/parameters/*.yaml`,
  then slices every block by the year string. `switches:` and `jec_parameters:` are sliced
  key-by-key (`get_parameters.py:25-36`). `configs/parameters/switches_profiles/*.yaml` are **not**
  loaded here — they are applied by `scripts/apply_switches_profile.py` rewriting `switches.yaml`,
  so the effective switches are whatever is in `switches.yaml` at run time.
- **NanoAOD.** 2024/2025 run on NanoAODv15; `getJetType` -> `AK4PFPuppi` for all Run 3
  (`copperhead_processor.py:218-226`).
- **do_geofit** is forced `True` in `get_parameters.py:42` but `apply_geofit`
  (`src/corrections/geofit.py`) is never called from `copperhead_processor.py`; `switches.yaml
  do_geofit` is `false` for all years. **GeoFit is not applied.**

---

## 1. Pipeline-ordered correction inventory

Order follows `EventProcessor.process` in `src/copperhead_processor.py`.

### 1.1 Pileup reweighting (`pu`)
- **Corrects:** event weight for MC pileup profile.
- **Data/MC:** MC only (`copperhead_processor.py:881`). Active 2024 and 2025.
- **Call site / order:** weights computed pre-filter at `copperhead_processor.py:870-889`
  (`pu_evaluator`), added to `Weights` at `:1610-1613` (right after gen/xsec/lumi, before muon SF).
  Impl `src/corrections/evaluator.py:151-187`; Run 3 branch `:174-183` uses `correctionlib`, first
  key in the file, `evaluate(nTrueInt, "nominal"/"up"/"down")` on `events.Pileup.nTrueInt`.
- **Payload (`pu_file_mc`, `configs/parameters/SF_filelist.yaml`):**
  - 2024 `:66` `data/cat/metadata/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/puWeights_BCDEFGHI.json.gz`
  - 2025 `:67` `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-05/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`
  - `pu_file_data` is `dummy_only_pu_file_mc_is_needed` for both (`SF_filelist.yaml:49-50`) — not used on the Run 3 path.
- **WP/tag:** correction key auto-selected as `list(ceval.keys())[0]` (`evaluator.py:180`); nominal 69200 ub implied by the 2025 filename only.
- **Weight vs kinematic:** weight.
- **Systematics:** `up` / `down` (`evaluator.py:182-183`), pushed as `weightUp/weightDown` of `"pu"` (`:1612`).
- **Switch:** `do_pu_wgt` (`switches.yaml:429-440`); default **true** for 2024 and 2025.

### 1.2 Normalization weights (`genWeight`, `genWeight_normalization`, `xsec`, `lumi`, k-factor)
- **Corrects:** MC normalization to luminosity x cross-section.
- **Data/MC:** MC only (`copperhead_processor.py:1580-1608`); data gets a `"ones"` weight (`:1627`). Active 2024/2025.
- **Order:** first weights added.
- **Values:** `genWeight` = `events.genWeight`, or `np.sign(genWeight)` if `"MiNNLO"` in dataset
  name (`:1582-1585`). `genWeight_normalization` = `1/sumGenWgts` (`:1587`, from
  `events.metadata['sumGenWgts']`). `xsec`, `lumi` from `get_sample_info(dataset_yaml_file,
  dataset, year)` (`:1591-1607`); `cross_section *= kfactor` (`:1600-1601`) where `kfactor_value`
  comes from the dataset YAML (`kfactor.value`, `= 1.0` for the 2024 blocks inspected). Integrated
  lumi also in `configs/parameters/lumi.yaml:13-14` (`2024: 109820.0`, `2025: 110840.0 pb^-1`).
- **Weight vs kinematic:** weight.
- **Systematics:** none.
- **Switch:** none (always on for MC).

### 1.3 L1 prefiring weight (`l1prefiring`)
- **Status:** **inactive** for 2024/2025. `switches.yaml do_l1prefiring_wgts` `2024/2025: false`
  (`:453-454`); code also requires `"L1PreFiringWeight" in events.fields`
  (`copperhead_processor.py:1615`). Not applicable to Run 3.

### 1.4 Beam-spot-constrained muon pT (`bsConstrainedPt`)
- **Corrects:** muon `pt` / `ptErr` (kinematic), replaced by `Muon.bsConstrainedPt/bsConstrainedPtErr`
  where `Muon.bsConstrainedChi2 < 30` (hard-coded).
- **Data/MC:** both (`copperhead_processor.py:914-922`). Active 2024/2025.
- **Order:** before Rochester; raw muon kinematics saved right after (`:927-930`).
- **Payload:** none (uses NanoAOD BS-constrained branches directly). Not a POG SF.
- **Weight vs kinematic:** modifies kinematic.
- **Systematics:** none.
- **Switch:** `do_beamConstraint` (`switches.yaml:110-121`); default **true** 2024 and 2025.
  Guarded by `"bsConstrainedChi2" in events.Muon.fields`.

### 1.5 Muon momentum scale & resolution — "KIT MuScaRe" (`pt_roch`)
- **Corrects:** muon `pt` (kinematic). MC: scale correction to gen Z peak **and** stochastic
  resolution smearing; Data: scale correction only.
- **Data/MC:** both. Active 2024 and 2025.
- **Call site / order:** `copperhead_processor.py:935-948`; Run 3 -> `apply_KitMuScaleRe_Run3`
  (`src/corrections/rochester.py:111-205`). `events.Muon.pt` is then set to `pt_roch` (`:945`).
  Applied before base muon selection uses `pt_raw`, before trigger matching, before FSR.
- **Payload (`roccor_file`, `SF_filelist.yaml`):**
  - 2024 `:14` `data/roch_corr/2024_Summer24.json` — comment: "From Hyeon (Copied from GitLab)";
    commented-out official cvmfs path
    `MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-11-27/muon_scalesmearing.json.gz` (`:13`).
  - 2025 `:16` `data/roch_corr/2025_muon_scalesmearing_VXBS.json`; commented-out cvmfs path
    `MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_scalesmearing_VXBS.json.gz` (`:15`).
- **Implementation:** `src/corrections/MuonScaRe.py` (vendored from `cms-muonPOG/muonscarekit`,
  `MuonScaRe.py:11`). MC uses `pt_scale(0,...)` then `pt_resol(...)` (`rochester.py:115-132`);
  data uses `pt_scale(1,...)` (`rochester.py:175-183`).
- **Weight vs kinematic:** modifies kinematic.
- **Systematics defined:** `pt_roch_scale_up/down` (`pt_scale_var`), and MC-only
  `pt_roch_resol_up/down` (`pt_resol_var`) — `rochester.py:170-201`. These feed the optional
  `build_muon_kinematic_variation_block` outputs (`copperhead_processor.py:1825-1909`) **only if**
  `save_muon_roccor_variations` is true — it is `false` for 2024/2025 (`switches.yaml:505-516`), so
  no roccor-varied branches are written and roccor is not added to `Weights`.
- **Switch:** `do_roccor` (`switches.yaml:74-85`), default **true** 2024/2025; also forced true in
  `get_parameters.py:40`.

### 1.6 FSR photon recovery (`pt_fsr`, `iso_fsr`)
- **Corrects:** muon `pt/eta/phi/mass` and `pfRelIso04_all` (kinematic + isolation) by adding a
  matched FSR photon.
- **Data/MC:** both. Active 2024/2025.
- **Call site / order:** computed at `copperhead_processor.py:975-980` (`fsr_recoveryV1`,
  `src/corrections/fsr_recovery.py:61-129`) after Rochester; iso overwrite applied immediately
  (`:980`), kinematics applied after trigger matching (`:1090-1093`).
- **Selection (hard-coded, `fsr_recovery.py:62-68`):** `fsrPhotonIdx>=0`, `relIso03<1.8`,
  `dROverEt2<0.012`, `pt(gamma)/pt(mu)<0.4`, `|eta(gamma)|<2.4`.
- **Payload:** none (NanoAOD `FsrPhoton` branches).
- **Weight vs kinematic:** modifies kinematic.
- **Systematics:** none.
- **Switch:** `do_fsr` (`switches.yaml:86-97`), default **true** 2024/2025; also forced true in
  `get_parameters.py:41`.

### 1.7 Event-by-event dimuon mass-resolution calibration (BS resolution calibration)
- **Corrects:** the per-event dimuon mass-resolution estimate (`dimuon_ebe_mass_res`) via a
  multiplicative `calibration` factor.
- **Data/MC:** both, when `doing_BS_correction` (= `do_beamConstraint`) is true
  (`copperhead_processor.py:1320-1322`, `2575-2606`).
- **Call site:** `EventProcessor.get_mass_resolution` (`copperhead_processor.py:2575`);
  `correction_set["BS_ebe_mass_res_calibration"].evaluate(mu1.pt, |mu1.eta|, |mu2.eta|)` (`:2600-2604`).
- **Payload (`BS_res_calib_path`, `SF_filelist.yaml`):**
  - 2024 `:48-50` — MC **and** Data both point at
    `data/res_calib/res_calib_BS_correction_2024_Data_nanoAODv15.json` (the "MC" key uses the
    *Data*-derived file).
  - 2025 `:51-54` — "Copied from 2024 until the EBE mass-resolution calibration pipeline is
    rerun"; MC and Data both `res_calib_BS_correction_2024_Data_nanoAODv15.json`.
- **Weight vs kinematic:** modifies a derived quantity (mass-resolution estimate), not a weight
  and not a 4-vector.
- **Systematics:** none.
- **Switch:** tied to `do_beamConstraint` (true 2024/2025).

### 1.8 NNLOPS reweighting (`nnlops`)
- **Corrects:** ggH signal Higgs-pT spectrum to NNLOPS.
- **Data/MC:** MC only, and only when `"ggh" in dataset` (`copperhead_processor.py:1650`).
- **Call site / order:** `copperhead_processor.py:1650-1655` (`nnlops_weights`,
  `src/corrections/evaluator.py:336-348`), before muon SF.
  `NNLOPS_Evaluator.evaluate(HTXS.Higgs_pt, HTXS.njets30, generator)` where generator is
  `"mcatnlo"` if `"amc"` in dataset name else `"powheg"` (`evaluator.py:338-345`).
- **Payload (`nnlops_file`, `SF_filelist.yaml:13-14`):** `data/NNLOPS_reweight.root` — **identical
  file for every year, Run 2 and Run 3** (Run 2-derived; TGraphs
  `gr_NNLOPSratio_pt_{mcatnlo,powheg}_{0..3}jet`).
- **Weight vs kinematic:** weight.
- **Systematics:** none.
- **Switch:** `do_nnlops` (`switches.yaml:393-404`), default **true** 2024/2025.

### 1.9 Muon ID / Iso / Trigger scale factors (`muID`, `muIso`, `muTrig`)
- **Corrects:** event weight for muon reco/ID, isolation, and single-muon trigger efficiency.
- **Data/MC:** MC only (`copperhead_processor.py:1658-1682`). Active 2024 and 2025.
- **Call site / order:** `add_muon_sfs_correctionlib(mu1, mu2, config)`
  (`src/corrections/muon_sf.py:45-150`), added to `Weights` at `copperhead_processor.py:1667-1681`,
  after NNLOPS, before LHE/THU/PDF.
- **Convention:** ID/Iso event SF = `SF(mu1)*SF(mu2)` on `(eta_raw, pt_raw)` (`muon_sf.py:91-110`);
  Trigger SF = **leading muon only**, `SF=1` outside `pt_raw>muon_leading_pt (26 GeV)` and
  `|eta_raw|<muon_eta_cut (2.4)` (`muon_sf.py:118-148`, config from
  `configs/parameters/muon.yaml:76-90, 16-30`).
- **Payload (`muSFFileList`, `SF_filelist.yaml`):**
  - 2024 `:110-114` `data/POG/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-11-27/muon_Z.json.gz`
  - 2025 `:115-119` `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_Z.json.gz`
- **SF set names (identical for 2024 and 2025):** `id: NUM_MediumID_DEN_TrackerMuons`,
  `iso: NUM_LoosePFIso_DEN_MediumID`, `trig: NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium`.
  Analysis muon ID is `mediumId` (`muon.yaml:58-59`), iso cut `pfRelIso04_all < 0.25`
  (`muon.yaml:43-44`), HLT `IsoMu24` (`configs/parameters/trigger.yaml:29-32`) — SF DEN/trigger
  names are consistent with those choices.
- **Weight vs kinematic:** weight.
- **Systematics:** `up`/`down` for each of `muID`, `muIso`, `muTrig`
  (`copperhead_processor.py:1667-1681`), from `"systup"/"systdown"` categories; `muon_sf.py:9-42`
  mirrors a missing variation symmetrically if a bin lacks one (a workaround noted for "an
  early-calibration muon trigger SF payload").
- **Switch:** none dedicated (runs whenever `is_mc` and year is Run 2 or Run 3,
  `copperhead_processor.py:1660`).

### 1.10 LHE renormalization / factorization scale weights (`LHERen`, `LHEFac`)
- **Corrects:** QCD scale (muR, muF) uncertainty, weight-only (nominal = 1).
- **Data/MC:** MC only, when `"LHEScaleWeight" in events.fields` and `"nominal" in pt_variations`
  (`copperhead_processor.py:1685-1701`). Active 2024/2025 for samples with the branch.
- **Impl:** `src/corrections/evaluator.py:517-561`. `lhefactor = 2.0` only for `dy_m105_160_amc`
  in 2017/2018 (`evaluator.py:518`), so **= 1.0** for all 2024/2025.
- **Payload:** NanoAOD `LHEScaleWeight`; index scheme hard-coded (`evaluator.py:525-552`).
- **Weight vs kinematic:** weight (variations only).
- **Systematics:** `LHERen` up/down, `LHEFac` up/down.
- **Switch:** none (automatic when branch present). Related: `do_THU` gates the STXS block, not this.

### 1.11 THU / STXS VBF acceptance uncertainties (`THU_VBF_*`)
- **Corrects:** theory uncertainty on VBF (qqH) STXS bins, weight-only (nominal = 1).
- **Data/MC:** MC only, and only when `"vbf" in dataset`, `"dy" not in dataset`, and
  `HTXS.stage1_1_fine_cat_pTjet30GeV` present (`copperhead_processor.py:1705-1718`).
- **Impl:** `add_stxs_variations` (`src/corrections/evaluator.py:831-882`); hard-coded `stxs_acc`,
  `uncert_deltas`, `powheg_xsec` tables (`evaluator.py:567-777`); names from
  `configs/parameters/misc.yaml sths_names` (`:123-133` for 2024, `:134-144` for 2025 — 10 names
  `Yield,PTH200,Mjj60,...,JET01`, identical to every other year).
- **Weight vs kinematic:** weight (up/down only).
- **Systematics:** one up/down pair per STXS name (`evaluator.py:855-859`).
- **Switch:** `do_THU` (`switches.yaml:417-428`), default **true** 2024/2025.

### 1.12 PDF variations (`pdf_2rms`)
- **Corrects:** PDF uncertainty (2*RMS of replicas), weight-only (nominal = 1).
- **Data/MC:** MC only, dataset must contain `dy|ewk|ggh|vbf` and not `mg`, and `LHEPdfWeight`
  present (`copperhead_processor.py:1721-1741`).
- **Impl:** `add_pdf_variations` (`src/corrections/evaluator.py:921-965`):
  `pdf_std = std(LHEPdfWeight[:, :n_pdf_variations])`, `up = 1+2*std`, `down = 1-2*std`.
  `n_pdf_variations = 33` for 2024 and 2025 (`configs/parameters/misc.yaml:415-416`).
- **Weight vs kinematic:** weight (up/down only).
- **Systematics:** `pdf_2rms` up/down.
- **Switch:** `do_pdf` (`switches.yaml:405-416`), default **true** 2024/2025.

### 1.13 Jet Energy Corrections — JES (`pt_jec`, `mass_jec`)
- **Corrects:** jet `pt`/`mass` (kinematic).
- **Data/MC:** both. Active 2024 and 2025.
- **Call site / order:** `copperhead_processor.py:1500-1533` -> `do_jec_scale`
  (`src/corrections/jet.py:767-865`). Applied after `prepare_jets` (`:1405`, sets `pt_raw`,
  `mass_raw`, `PU_rho`) and after optional jet-veto-map jet filtering; before JER; jets re-sorted
  by `pt` after (`:1556-1557`).
- **Levels:** nominal uses compound key `{jec_tag}_L1L2L3Res_AK4PFPuppi` (`jet.py:812-822`).
  `jec_levels_mc = [L1FastJet, L2Relative, L3Absolute]`, `jec_levels_data` adds `L2L3Residual`
  (`configs/parameters/jec.yaml:658-665, 726-735`).
- **Payload (`jec_parameters.jerc_load_path`, `jec.yaml`):**
  - 2024 `:30` `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-07-16/jet_jerc.json.gz`
  - 2025 `:31` `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16/jet_jerc.json.gz`
- **Tags (`jec.yaml`):**
  - MC `jec_tags` — 2024 `:769` `Summer24Prompt24_V5_MC`; 2025 `:770` `Summer24Prompt25_V3_MC`.
  - Data `jec_data_tags` — 2024 `:905-913` `Summer24Prompt24_V5_DATA` (runs C,D,E,F,G,H,I);
    2025 `:914-921` `Summer24Prompt25_V3_DATA` (runs B,C,D,E,F,G). Data tag chosen by matching a
    run letter in the dataset name (`jet.py:784-792`, `getJecDataTag` `:681-698`).
  - `runs` list for 2024 (`jec.yaml:592-600`) includes `B`, but `jec_data_tags` 2024 has no `B`
    entry -> a 2024 Run-B data file would raise `ValueError` ("JEC tag not found"). No `Run2024B`
    dataset is configured, so currently latent.
- **Weight vs kinematic:** modifies kinematic.
- **Systematics:** `do_jec_unc` is **false** for 2024/2025 (`switches.yaml:378-379`), and
  force-disabled on data (`copperhead_processor.py:1490-1492`). The `Regrouped_*` source machinery
  (`jet.py:753-765`, `get_jec_sources`), `jec_unc_to_consider`/`jec_variations`
  (`jec.yaml:181-216, 471-516`; 2025 reuses `*_2024` source labels, `:194-203`) is therefore
  **not exercised** for 2024/2025.
- **Switch:** `do_jec` (`switches.yaml:333-344`), default **true** 2024/2025.

### 1.14 Jet Energy Resolution smearing — JER (`pt_jer_nom`)
- **Corrects:** jet `pt` (kinematic) — MC only.
- **Data/MC:** **MC only** (`copperhead_processor.py:1543`; requires `jer_strat >= 0`). Active
  2024 and 2025 (MC).
- **Call site / order:** `do_jer_smear` (`src/corrections/jet.py:964-1135`), after JES, before
  final jet sort. Nominal-only run passes `syst_l=["nom"]` (`copperhead_processor.py:1550`).
- **Strategy:** `jer_strat = 4` for 2024 and 2025 (`switches.yaml:366-367`) -> `applyStrat4`
  (`jet.py:916-927, 1114-1115`): scaling method for gen-matched jets; **no stochastic smearing for
  unmatched jets, unconditionally** (no PUID/pT/eta gating). Comment cites the JME "option (b)"
  ad-hoc mitigation for Run 3 2.5<|eta|<3.0 data/MC (`jet.py:917-925`, `switches.yaml:345-368`).
- **Payload / tags:**
  - `jersmear_load_path` (`jec.yaml:45-46`): `data/POG/.../JME/jer_smear.json.gz` (shared, all years).
  - Resolution + SF read from the same `jet_jerc.json.gz` as JES (`jet.py:977`).
  - `jer_tags` (`jec.yaml`): 2024 `:784` `Summer24Prompt24_JRV2_MC`; 2025 `:785`
    `Summer24Prompt25_JRV2_MC`.
  - Keys: `{jer_tag}_ScaleFactor_AK4PFPuppi`, `{jer_tag}_PtResolution_AK4PFPuppi`
    (`jet.py:990, 1023`). For Summer24 the ScaleFactor payload has **no `systematic` axis**; up/down
    come from a separate `{jer_tag}_SFUncertainty_AK4PFPuppi` correction as `SF +/- SFUncertainty`
    (`jet.py:1001-1052`).
- **Weight vs kinematic:** modifies kinematic.
- **Systematics:** `do_jer_unc` **false** for 2024/2025 (`switches.yaml:381-392`, header comment
  "FIXME: Not validated yet for Run-3"). The 6-bin `jer1..jer6` variation columns
  (`jet.py:930-961 apply_jer_unc`; `jer_variations` `jec.yaml:1072-1097`) are **not built** on a
  nominal run (`jet.py:1129-1130`).
- **Switch:** `jer_strat` (`switches.yaml:345`), value `4` for 2024/2025; `-1` would disable
  smearing entirely.

### 1.15 Jet veto maps — event filter
- **Corrects:** removes whole events with any (minimally selected) jet in a vetoed (eta,phi) region.
- **Data/MC:** both (`compute_jet_veto_eventfilter`, `copperhead_processor.py:608-660`; called
  `:1257-1264` before the event skim). Active 2024 and 2025 in "filter events" mode.
- **Mode:** `do_jet_veto_maps_filterEvents` = **true** for 2024/2025 (`switches.yaml:203-204`);
  `do_jet_veto_maps_filterJets` = **false** for 2024/2025 (`switches.yaml:215-216`) (2022/2023 use
  the opposite — jet-level filtering).
- **Minimal jet pre-selection (per JME doc, `copperhead_processor.py:619-624`):** `pt>15`,
  `tightLepVeto` jet ID (`jet_veto_map_jet_id`, `configs/parameters/jet.yaml:94-95`),
  `(chEmEF+neEmEF)<0.9`.
- **Payload (`configs/parameters/jet.yaml`):**
  - `jet_veto_maps` — 2024 `:319` `.../JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-07-17/jetvetomaps.json.gz`;
    2025 `:320` `/cvmfs/.../JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16/jetvetomaps.json.gz`
  - `jet_veto_maps_tag` — 2024 `:331` `Summer24Prompt24_RunBCDEFGHI_V1`; 2025 `:332`
    `Summer24Prompt25_RunCDEFG_V1`
- **Weight vs kinematic:** event filter (neither).
- **Systematics:** none.
- **Switch:** `do_jet_veto_maps_filterEvents` (default true 2024/2025).

### 1.16 Jet-horn pT cut (HE region)
- **Corrects:** selection — in the HE region `2.5<|eta|<=3.0`, require `jet.pt > 50`
  (`copperhead_processor.py:2778, 2804-2819`).
- **Data/MC:** both.
- **2024 vs 2025:** `do_jet_horn_ptcut` = **`50` for 2024**, **`false` for 2025**
  (`switches.yaml:235-236`). The switch's own comment says 2025 is "treated the same as 2024 ...
  per explicit user decision" but the value differs — this is a **2024/2025 inconsistency to
  confirm** (`switches.yaml:218-237`). It "never touches HF (3<|eta|<5)".
- **Related switches, both `false` for 2024/2025:** `add_pt_cut_for_HE_HF_jets` (`:305-306`),
  `add_asymmetric_pt_cut_for_HE_HF_jets` (`:317-318`), `do_jet_HE_nConstCut` (`:247-248`),
  `do_reject_high_rawFactor_jets` (`:264-265`), `do_jet_horn_puid` (`:330-331`),
  `do_jet_PUID_cut` (`:167-168`).
- **Weight vs kinematic:** selection.
- **Systematics:** none.

### 1.17 Muon-jet overlap removal (jet cleaning)
- **Selection, not a correction:** jets with `dR(jet, mu_raw) <= 0.4` to either selected leading
  muon are dropped (`copperhead_processor.py:2651-2673`, cites AN-19-124 line 465).
  `min_dr_mu_jet: 0.4` (`configs/parameters/jet.yaml:155-156`). Electron veto (`nelectrons==0`) at
  `copperhead_processor.py:1167-1196`. Identical for 2024 and 2025.

### 1.18 b-tag working-point counting (`nBtagLoose`, `nBtagMedium`)
- **Selection quantity, not a weight.** For NanoAODv15 Run 3 uses `btagUParTAK4B`
  (`copperhead_processor.py:3320-3323`) with WPs from `configs/parameters/jet.yaml`:
  - `btag_loose_wp_UParT` 2024 `:260` `0.0246`, 2025 `:261` `0.0246` ("PLACEHOLDER copied from
    2024, pending BTV POG 2025 WPs")
  - `btag_medium_wp_UParT` 2024 `:275` `0.1272`, 2025 `:276` `0.1272` (same placeholder note)
- b-jet selection region: `pt>25` (`jet.yaml:43-44`), `|eta|<2.5` (`jet.yaml:58-59`;
  `btag_jet_selection` `src/corrections/jet.py:399-402`), tight jet ID (`btag_jet_id`,
  `jet.yaml:110-111`).

### 1.19 b-tag scale-factor weight (`btag`, `btag_*`)
- **Status:** **inactive** for 2024/2025. `do_btag_wgt` = `false` (`switches.yaml:490-491`). Code
  path `copperhead_processor.py:3225-3296` -> `btag_weights_jsonKeepDim`
  (`src/corrections/evaluator.py:1312-1400`) would use `btag_sf_json` (`SF_filelist.yaml:25-26`:
  2024 `.../BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-08-19/btagging.json.gz`;
  2025 `/cvmfs/.../BTV/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26/btagging.json.gz`), key
  `deepJet_shape`, systs `jes,lf,hfstats1/2,cferr1/2,hf,lfstats1/2`
  (`configs/parameters/misc.yaml:267-286`). `btag_sf_csv` for those years is a stale
  `data/btag/DeepCSV_106XUL18SF.csv` with `# FIXME: Update SF` (`SF_filelist.yaml:37-38`) but is
  only read on `RERECO`.

### 1.20 QGL / PNet-QvG weight (`qgl`)
- **Status:** **inactive** for 2024/2025. `do_qgl_wgt` = `false` (`switches.yaml:478-479`; comment
  "Run-3: Recommendation does not exist, yet"). Code `copperhead_processor.py:3206-3219` ->
  `qgl_weights_V2` (`src/corrections/evaluator.py:970-1111`) would, for Run 3, use `btagPNetQvG`
  with hard-coded Run 2 QGL polynomial coefficients — not used for these years.

### 1.21 Jet PU-ID scale-factor weight (`jetpuid`)
- **Status:** **inactive** for 2024/2025. Guarded by `is_run2(year)` and `hasattr(jets,"puId")`
  (`copperhead_processor.py:2901-2910`); `apply_jet_PUID_wgt` = `false` for all years
  (`switches.yaml:465-466`). Payload `jetpuid_sf_file` (`SF_filelist.yaml:201-202`) is not consumed
  for Run 3.

### 1.22 Region PU/HS jet-ID DNN cleanup (`pu_dnn_pass`)
- **Status:** **inactive** for 2024/2025. `do_use_pu_dnn_score` = `false`
  (`switches.yaml:291-292`). If enabled it would drop forward jets failing per-region torch models
  under `pu_dnn_model_dir` (`SF_filelist.yaml:168-169`: 2024
  `validation/pu_dnn/run2024_dy_top_ewk_RemoveMuon_OnlyDMetJet`; **2025 = same dir, "placeholder
  until a 2025-trained model exists"**). Impl `src/corrections/pu_dnn.py`. Mutually exclusive with
  `do_use_pySR_score` (also false, `switches.yaml:277-278`).

### 1.23 Z-pT reweighting (`zpt`, `zpt_wgt_reco/gen`)
- **Status:** **inactive** for 2024/2025. `do_zpt` = `false` (`switches.yaml:143-144`); block
  `copperhead_processor.py:2288-2404` requires `'dy' in dataset and is_mc and do_zpt`.
  `save_zpt_variations` also false (`switches.yaml:155-156`).
- If enabled: `getZptWgts_3region` (`copperhead_processor.py:298-390`) with piecewise-polynomial
  config `new_zpt_weights_file_aMCatNLO` — 2024 `SF_filelist.yaml:139`
  `data/zpt_rewgt/zpt_rewgt_params_IncDY_aMCatNLO_Run3_nanoAODv15_FilterJets_July08_tightPassLepVeto_PUDNN_TrainOn2024.yaml`;
  **2025 `SF_filelist.yaml:128` is a placeholder "copied from 2024 only so getParametersForYr's
  per-year key lookup doesn't crash"**. There is also a dead `whichMethod=="DNN"` branch gated on
  `str(year)=="2024"` with hard-coded `/depot/cms/private/users/shar1172/...` TorchScript paths
  (`copperhead_processor.py:2339-2377`), fully commented out.

### 1.24 DY M-100-200 LHE stitching cut
- **Status:** **inactive** for 2024/2025. `do_remove_dy_M100to200` = `false`
  (`switches.yaml:131-132`); code `copperhead_processor.py:810-842`. (Run 2 only.)

### 1.25 HEM veto
- **Status:** **inactive** for 2024/2025. `do_HemVeto` = `false` (`switches.yaml:179-180`,
  2018-only). Code `copperhead_processor.py:1172-1181`, `src/corrections/jet.py:619-678`.

### 1.26 ECAL bad-calibration event-filter recipe
- **Corrects:** rejects data events per the MET `ECal_BadCalibration` recipe.
- **Data/MC:** **data only** (no-op on MC by prescription, `copperhead_processor.py:512-513`).
- **2024/2025 relevance:** hard-coded run window `362433-367144`
  (`copperhead_processor.py:475-476`), described in the docstring as "later part of 2022 and early
  2023". For 2024/2025 runs the `in_run_range` mask is always False, so the recipe is effectively a
  no-op — but it is still executed unconditionally for all data years (`:907`). Separately,
  `event_flags` for 2024 includes `ecalBadCalibFilter` (`configs/parameters/misc.yaml:384`) while
  2025 has it **commented out** (`misc.yaml:393`) — a 2024/2025 MET-filter-list difference to
  confirm.
- **Switch:** none.

### 1.27 Run 3 jet ID (correctionlib) — object selection payload
- Not a correction/SF but a per-year correctionlib payload: for NanoAODv15 with `jetId` present
  the code uses the bit-decoded branch (`src/corrections/jet.py:282-314`); the correctionlib
  fallback (`jet.py:327-360`) would read `jet_id_json_files` (`configs/parameters/jet.yaml:125-126`:
  2024 `.../JME/Run3-24...-NanoAODv15/2025-12-02/jetid.json.gz`; 2025
  `/cvmfs/.../JME/Run3-25...-NanoAODv15/2026-07-16/jetid.json.gz`). Analysis WP `tightPassLepVeto`
  for 2024/2025 (`jet.yaml:75-76`).

---

## 1A. Year 2026 — correction-by-correction

Section numbers match §1.1–§1.27 above. "Same as 2025" means identical switch value **and** the
payload is 2025's (which is itself frequently a 2024 copy).

- **§1.1 Pileup reweighting** — **active** (`do_pu_wgt` `2026: true`, `switches.yaml:440`).
  Payload `SF_filelist.yaml:68` = `LUM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-05/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`.
  **PLACEHOLDER** — config: *"no LUM/*2026* payload exists on cvmfs yet, reusing 2025 pileup
  weights"*. This is the 2025 file. `pu_file_data` `2026` = dummy `# FIXME` (`SF_filelist.yaml:51`).
- **§1.2 gen/xsec/lumi/k-factor** — **active** (MC only). `lumi.yaml:15` `2026: 25950.0` — *"Golden
  JSON lumi for eras A+B+D (0.64+15.28+10.03 /fb); era C (low-PU) excluded ... Preliminary Offline
  Results"*. **REAL 2026 value, preliminary.** Cross-sections/k-factors = shared Summer24 values.
- **§1.3 L1 prefiring** — **inactive** (`do_l1prefiring_wgts` `2026: false`, `switches.yaml:455`).
- **§1.4 BS-constrained muon pT** — **active** (`do_beamConstraint` `2026: true`, `switches.yaml:121`).
  No payload (NanoAOD `bsConstrainedPt`, chi2 < 30). Same as 2025.
- **§1.5 KIT muon scale & resolution (`pt_roch`)** — **active** (`do_roccor` `2026: true`,
  `switches.yaml:85`). Payload `correction_filelist.yaml:19` = `data/roch_corr/2025_muon_scalesmearing_VXBS.json`.
  **PLACEHOLDER** — config (`correction_filelist.yaml:17-18`): *"MUO/Run3-26Prompt-Summer24-NanoAODv15/
  exists on cvmfs but is empty (no Rochester payload yet); reusing the 2025 ... corrections"*.
  Identical file to 2025. `save_muon_roccor_variations` `2026: false` (`switches.yaml:516`).
- **§1.6 FSR recovery** — **active** (`do_fsr` `2026: true`, `switches.yaml:97`). Hard-coded cuts.
- **§1.7 EBE dimuon mass-resolution BS calibration** — **active** (via `do_beamConstraint`).
  Payload `correction_filelist.yaml:55-59` MC **and** Data both =
  `data/res_calib/res_calib_BS_correction_2024_Data_nanoAODv15.json`. **PLACEHOLDER (2026 <- 2025 <-
  2024)** — config: *"EBE mass-resolution calibration pipeline has not been rerun for 2026; copied
  from 2024"*.
- **§1.8 NNLOPS** — **active** (`do_nnlops` `2026: true`, `switches.yaml:404`; ggH only).
  `data/NNLOPS_reweight.root` — year-independent, Run 2-derived (all years).
- **§1.9 Muon ID / Iso / Trigger SF** — **active** (MC only, no switch). Payload
  `SF_filelist.yaml:120-124` = `MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_Z.json.gz`.
  **PLACEHOLDER** — config: *"MUO/Run3-26Prompt-Summer24-NanoAODv15/ ... is empty, reusing 2025
  muon SFs"*. SF-set names identical to 2024/2025 (`NUM_MediumID_DEN_TrackerMuons`,
  `NUM_LoosePFIso_DEN_MediumID`, `NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium`).
- **§1.10 LHE muR/muF scale** — **active** (MC, branch-gated). Factor 1.0 for Run 3.
- **§1.11 THU / STXS VBF** — **active** (`do_THU` `2026: true`, `switches.yaml:428`; vbf only).
  `misc.yaml:145-155` `sths_names["2026"]` = same 10 STXS bins as every year.
- **§1.12 PDF 2*RMS** — **active** (`do_pdf` `2026: true`, `switches.yaml:416`).
  `n_pdf_variations` `2026: 33` (`misc.yaml:417`) — same as 2022–2025.
- **§1.13 JES / JEC** — **active** (`do_jec` `2026: true`, `switches.yaml:344`); data & MC.
  Payload `jec.yaml:32` = `JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jet_jerc.json.gz` —
  **REAL 2026 path**. MC tag `jec.yaml:771` = `Summer24Prompt26_V1_MC` — **REAL, 2026-specific**.
  Data tags `jec.yaml:922-927` = single flat `Summer24Prompt26_V1_DATA` for runs A,B,C,D — **REAL**
  (*"verified via jet_jerc.json.gz correction names on cvmfs"*). `runs` `2026` = A,B,C,D
  (`jec.yaml:608-612`). **Systematics latent** (`do_jec_unc` `2026: false`, `switches.yaml:380`) but
  the source labels ARE 2026-specific (`Absolute_2026`, `BBEC1_2026`, ...; `jec.yaml:205-216,
  :517-539`) — unlike 2025 which reuses `*_2024` labels (main Unknown #7).
- **§1.14 JER smearing, strategy 4** — **active** (`jer_strat` `2026: 4`, `switches.yaml:368`); MC
  only. `applyStrat4` (no stochastic smearing for unmatched jets). JER tag `jec.yaml:786-791` =
  `Summer24Prompt26_RunBD_JRV1_MC` — **REAL, 2026-specific**, with a documented limitation:
  *"era C is a dedicated low-PU run and has its own JER (Summer24Prompt26_RunC_JRV1_MC) ... This
  config only supports one JER tag for the whole '2026' year bucket, so we use the RunBD tag ...
  a real limitation to be aware of, not a placeholder"*. Era-C data is `skip_sample: True`.
  **`do_jer_unc` `2026: false`** (`switches.yaml:392`).
- **§1.15 Jet veto maps (event filter)** — **active** (`do_jet_veto_maps_filterEvents` `2026: true`,
  `switches.yaml:205`; `filterJets` `2026: false`, `:217`). Payload `jet.yaml:321` =
  `JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetvetomaps.json.gz` — **REAL 2026 path**.
  Tag `jet.yaml:333` = `Summer24Prompt26_RunBCD_V1` — **REAL, 2026-specific**.
- **§1.16 HE jet-horn pT>50** — **inactive** (`do_jet_horn_ptcut` `2026: false`, `switches.yaml:237`).
  Same as 2025; unlike 2024 (=50). Switch comment says 2025/2026 "treated the same as 2024", which
  contradicts the value (main Unknown #1). Related HE/HF switches all `false` for 2026.
- **§1.17 Muon-jet overlap removal** — **active** (selection). `min_dr_mu_jet` `2026: 0.4`
  (`jet.yaml:157`). Identical to 2024/2025.
- **§1.18 b-tag WP counting** — **active** (selection quantity). `btagUParTAK4B`.
  `btag_loose_wp_UParT` `2026: 0.0246`, `btag_medium_wp_UParT` `2026: 0.1272` (`jet.yaml:262, 277`)
  — **PLACEHOLDER**: *"copied from 2025 (itself copied from 2024), pending BTV POG 2026 WPs"*.
- **§1.19 b-tag SF weight** — **inactive** (`do_btag_wgt` `2026: false`, `switches.yaml:492`).
  Would use `SF_filelist.yaml:27` = `BTV/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26/btagging.json.gz`
  — **PLACEHOLDER** (*"no BTV/*2026* payload ... reusing 2025"*). `btag_sf_csv` `2026` = stale
  `DeepCSV_106XUL18SF.csv # FIXME` (`SF_filelist.yaml:39`).
- **§1.20 QGL / PNet-QvG weight** — **inactive** (`do_qgl_wgt` `2026: false`, `switches.yaml:480`).
- **§1.21 Jet PU-ID SF weight** — **inactive** for Run 3 (`is_run2` guard; `apply_jet_PUID_wgt`
  `2026: false`, `switches.yaml:467`). `jetpuid_sf_file["2026"]` (`SF_filelist.yaml:203`) is a REAL
  `Run3-26Prompt` `jetid.json.gz` path but not consumed.
- **§1.22 PU/HS jet-ID DNN** — **inactive** (`do_use_pu_dnn_score` `2026: false`,
  `switches.yaml:293`). `pu_dnn_model_dir["2026"]` (`SF_filelist.yaml:170`) =
  `validation/pu_dnn/run2024_dy_top_ewk_RemoveMuon_OnlyDMetJet` — **PLACEHOLDER** (*"placeholder
  until a 2026-trained model exists"*).
- **§1.23 Z-pT reweighting** — **inactive** (`do_zpt` `2026: false`, `switches.yaml:145`;
  `save_zpt_variations` `2026: false`, `:157`). `new_zpt_weights_file_aMCatNLO["2026"]`
  (`SF_filelist.yaml:129-130`) = `data/zpt_rewgt/zpt_rewgt_params_aMCatNLO.yaml` — **PLACEHOLDER**
  (*"placeholder copied from 2025/2024 ... so ... key lookup doesn't crash"*).
- **§1.24 DY M-100-200 LHE stitching** — **inactive** (`do_remove_dy_M100to200` `2026: false`,
  `switches.yaml:133`). Run 2 only.
- **§1.25 HEM veto** — **inactive** (`do_HemVeto` `2026: false`, `switches.yaml:181`;
  `HemVeto_ratio["2026"] = 0.0`, `jet.yaml:309`). 2018 only.
- **§1.26 ECAL bad-calibration recipe** — **executed but no-op for 2026** (run window
  `362433-367144` hard-coded; data only). MET-filter list `event_flags["2026"]`
  (`misc.yaml:394-402`) has `ecalBadCalibFilter` **commented out** — same as 2025, differs from
  2024 (which includes it).
- **§1.27 Run 3 jet ID payload** — analysis WP `jet_id["2026"] = tightPassLepVeto` (`jet.yaml:77`).
  Bit-decoded path used when `jetId` present; correctionlib fallback reads `jet_id_json_files["2026"]`
  (`jet.yaml:127`) = `JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetid.json.gz` — **REAL 2026
  path**.
- **Selection scalars** (all identical to 2024/2025): `muon_pt_cut 20`, `muon_eta_cut 2.4`,
  `muon_iso_cut 0.25`, `muon_id mediumId`, `muon_leading_pt 26`, `muon_trigmatch_pt 24`,
  `muon_trigmatch_dr 0.4`; `jet_pt_cut 25`, `jet_eta_cut 4.7`, `btag_jet_eta_cut 2.5`; HLT `IsoMu24`.

---

## 2. Summary table (pipeline order)

"on"/"off" = switch state. Payload column shows 2026 (see §1.13-§1.15 etc. for 2024/2025 payloads
in full). **R** = real 2026 payload, **P** = placeholder (copied from 2025/2024).

| # | Name | Data/MC | 2024 | 2025 | 2026 | Weight/kinematic | 2026 payload (R/P) | Tag / SF-set / WP | Systematics | Switch (2024/2025/2026) |
|---|---|---|---|---|---|---|---|---|---|---|
|1.1|Pileup wgt|MC|on|on|on|weight|`LUM/Run3-25.../2026-06-05/puWeights_2025pp_...69200ub` (**P**, "reusing 2025")|first key in file|nom/up/down|`do_pu_wgt` (T/T/T)|
|1.2|gen/xsec/lumi/k-factor|MC|on|on|on|weight|`lumi.yaml` 2026=25950 pb⁻¹ A+B+D (**R**, preliminary); xsec = Summer24|—|none|none|
|1.3|L1 prefiring|MC|off|off|off|weight|—|—|nom/up/down|`do_l1prefiring_wgts` (F/F/F)|
|1.4|BS-constrained muon pT|both|on|on|on|kinematic|NanoAOD `bsConstrainedPt` (chi2<30)|—|none|`do_beamConstraint` (T/T/T)|
|1.5|KIT muon scale+resolution|both|on|on|on|kinematic|`data/roch_corr/2025_muon_scalesmearing_VXBS.json` (**P**, "Run3-26Prompt empty")|MuonScaRe kit|scale_up/down (both), resol_up/down (MC) — not written|`do_roccor` (T/T/T)|
|1.6|FSR recovery|both|on|on|on|kinematic|NanoAOD `FsrPhoton`|hard-coded cuts|none|`do_fsr` (T/T/T)|
|1.7|EBE mass-res BS calibration|both|on|on|on|derived quantity|`res_calib_BS_correction_2024_Data_nanoAODv15.json` (**P**, 2026←2025←2024; MC & Data)|`BS_ebe_mass_res_calibration`|none|via `do_beamConstraint`|
|1.8|NNLOPS|MC (ggh)|on|on|on|weight|`data/NNLOPS_reweight.root` (year-independent, Run 2-derived)|`mcatnlo`/`powheg` graphs|none|`do_nnlops` (T/T/T)|
|1.9|Muon ID/Iso/Trig SF|MC|on|on|on|weight|`MUO/Run3-25.../2026-04-28/muon_Z.json.gz` (**P**, "Run3-26Prompt empty")|`NUM_MediumID_DEN_TrackerMuons`, `NUM_LoosePFIso_DEN_MediumID`, `NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium`|up/down each|none (auto MC)|
|1.10|LHE muR/muF scale|MC|on|on|on|weight (var only)|NanoAOD `LHEScaleWeight`|factor 1.0 (Run 3)|LHERen/LHEFac up/down|none|
|1.11|THU / STXS VBF|MC (vbf)|on|on|on|weight (var only)|hard-coded tables; `sths_names["2026"]` (10 bins)|10 STXS bins|up/down per bin|`do_THU` (T/T/T)|
|1.12|PDF 2*RMS|MC|on|on|on|weight (var only)|`LHEPdfWeight[:, :33]`|n=33|pdf_2rms up/down|`do_pdf` (T/T/T)|
|1.13|JES (JEC)|both|on|on|on|kinematic|`JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jet_jerc.json.gz` (**R**)|MC `Summer24Prompt26_V1_MC`; DATA `Summer24Prompt26_V1_DATA` (A-D); L1L2L3Res, AK4PFPuppi|`do_jec_unc` **false** (labels are `*_2026`, not `*_2024` like 2025)|`do_jec` (T/T/T)|
|1.14|JER smearing (strat 4)|MC|on|on|on|kinematic|`jer_smear.json.gz` + 2026 `jet_jerc.json.gz` (**R** tag)|`Summer24Prompt26_RunBD_JRV1_MC` (era-C low-PU JER not selectable); ScaleFactor+SFUncertainty, AK4PFPuppi|`do_jer_unc` **false**|`jer_strat` (4/4/4)|
|1.15|Jet veto maps (event filter)|both|on|on|on|event filter|`JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetvetomaps.json.gz` (**R**)|`Summer24Prompt26_RunBCD_V1`|none|`do_jet_veto_maps_filterEvents` (T/T/T)|
|1.16|HE jet-horn pT>50|both|**on (50)**|**off**|**off**|selection|—|—|none|`do_jet_horn_ptcut` (50/false/false)|
|1.17|µ-jet overlap removal|both|on|on|on|selection|`min_dr_mu_jet` 0.4|—|ΔR≤0.4|—|
|1.18|b-tag WP counting|both|on|on|on|selection quantity|`btagUParTAK4B`; WPs 0.0246 / 0.1272 (**P**, "copied from 2025 (itself from 2024)")|loose/medium UParT|none|—|
|1.19|b-tag SF weight|MC|off|off|off|weight|`BTV/Run3-25.../2026-06-26/btagging.json.gz`, key `deepJet_shape` (**P**)|—|jes,lf,hf,hfstats1/2,lfstats1/2,cferr1/2|`do_btag_wgt` (F/F/F)|
|1.20|QGL weight|MC|off|off|off|weight|Run 2 polynomials on `btagPNetQvG`|—|up/down|`do_qgl_wgt` (F/F/F)|
|1.21|Jet PU-ID SF weight|MC|off (Run 2 only)|off|off|weight|`jetpuid_sf_file` (2026 path is **R** but unused)|—|—|`apply_jet_PUID_wgt` (F/F/F)|
|1.22|PU/HS jet-ID DNN|both|off|off|off|selection|`validation/pu_dnn/run2024_...` (**P**)|per-region torch|score only|`do_use_pu_dnn_score` (F/F/F)|
|1.23|Z-pT reweighting|MC (dy)|off|off|off|weight|`zpt_rewgt_params_aMCatNLO.yaml` (**P**)|3-region polynomial|reco/gen ±1σ (not saved)|`do_zpt` (F/F/F)|
|1.24|DY M-100-200 LHE cut|MC (dy)|off|off|off|event cut|—|—|none|`do_remove_dy_M100to200` (F/F/F)|
|1.25|HEM veto|both|off|off|off|event filter|`HemVeto_ratio["2026"]=0.0`|—|none|`do_HemVeto` (F/F/F)|
|1.26|ECAL bad-calib recipe|data|no-op|no-op|no-op (run window 2022-23)|event filter|—|MET-filter list: `ecalBadCalibFilter` commented out (= 2025, ≠ 2024)|none|none|
|1.27|Run 3 jet ID payload|both|on|on|on|selection|`JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetid.json.gz` (**R**)|WP `tightPassLepVeto`|—|—|

---

## 3. Unknowns / needs authoritative verification

1. **`do_jet_horn_ptcut` 2024 vs 2025 mismatch.** `switches.yaml:235-236` sets `2024: 50` but
   `2025: false`, while the same switch's comment (`switches.yaml:222-226`) says 2025 is "treated
   the same as 2024". Either the value or the comment is wrong.
2. **2025 muon Rochester/scale-smear payload** (`data/roch_corr/2025_muon_scalesmearing_VXBS.json`)
   — provenance not pinned in-repo; official cvmfs equivalent only present commented-out
   (`SF_filelist.yaml:15`). Confirm it matches `MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/`.
3. **2024 Rochester payload** labeled "From Hyeon (Copied from GitLab)" (`SF_filelist.yaml:14`)
   rather than the official `muon_scalesmearing.json.gz` — verify equivalence/version.
4. **EBE mass-resolution BS calibration for 2025 is literally the 2024 file**
   (`SF_filelist.yaml:51-54`). Also for **2024 the "MC" key points at the *Data*-derived JSON**
   (`SF_filelist.yaml:48-50`) — confirm intentional.
5. **NNLOPS file is year-independent** (`data/NNLOPS_reweight.root`) and Run 2-derived; no Run 3 /
   13.6 TeV NNLOPS ratio provided. Validate applying Run 2 NNLOPS to a Run 3 ggH sample.
6. **JER ScaleFactor uncertainty handling for Summer24** relies on a code-side assumption that the
   payload has no `systematic` axis and exposes `{jer_tag}_SFUncertainty_{algo}` with a symmetric
   absolute delta (`jet.py:1001-1052`). Check against the actual `jet_jerc.json.gz` schema. Only
   matters if `do_jer_unc` is turned on (currently `false`, "Not validated yet for Run-3").
7. **JEC uncertainty source labels for 2025 reuse `*_2024` names** (`jec.yaml:194-203`, `497-514`).
   Latent (`do_jec_unc` false), but wrong if JES systematics are enabled for 2025.
8. **2024 `runs` list includes `B` but `jec_data_tags["2024"]` covers only C-I** (`jec.yaml:592-600`
   vs `:905-913`); a `Run2024B` data file would raise `ValueError`. No such dataset configured today.
9. **`event_flags` (MET filters) differ between 2024 and 2025**: 2024 includes `ecalBadCalibFilter`,
   2025 has it commented out (`misc.yaml:376-393`). Confirm intended recommendation per year.
10. **`apply_ECALBadCalib_EventFilter_recipe` runs unconditionally for all data years** with a
    hard-coded 2022-2023 run window (`copperhead_processor.py:469-556`, `:907`); no-op for
    2024/2025 but not year-guarded.
11. **b-tag WPs for 2025 are placeholder copies of 2024** (`jet.yaml:261, 276`). Only affects
    `nBtagLoose/nBtagMedium` counting today (b-tag SF weight off), so any b-veto category built on
    these for 2025 uses 2024 numbers.
12. **b-tag SF payloads / `deepJet_shape` key**: off for 2024/2025, but if enabled the code assumes
    a `deepJet_shape` correction while the NanoAODv15 discriminant is `btagUParTAK4B`
    (`copperhead_processor.py:3252-3258, 3320-3323`) — discriminant/SF mismatch to resolve.
    `btag_sf_csv` still `DeepCSV_106XUL18SF.csv` with `# FIXME` (`SF_filelist.yaml:33-39`).
13. **2025 Z-pT and PU-DNN artifacts are 2024 placeholders** (`SF_filelist.yaml:126-128, 169`).
    Both features off for 2025, so no current impact.
14. **Pileup correction key** chosen as `list(ceval.keys())[0]` (`evaluator.py:180`) rather than a
    named key/WP; 69200 ub assumption inferred only from the 2025 filename. Verify the 2024 file's
    first key is the intended nominal.
15. **`pu_file_data` = `dummy_only_pu_file_mc_is_needed`** for 2024/2025 with `# FIXME`
    (`SF_filelist.yaml:49-50`) — harmless on the Run 3 path but unresolved.
16. **2026 payload split (superseded by §1A — kept for the record):** JME payloads for 2026 ARE
    real `Run3-26Prompt` objects (JES/JER/veto-map/jet-ID); MUO + LUM payloads are placeholders
    reusing 2025/2024 (`SF_filelist.yaml:27, 68, 120-124`; `correction_filelist.yaml:17-19, 55-59`).

17. **2026 MUO calibrations are 2025/2024 stand-ins (authoritative verification required).**
    `roccor_file["2026"]` (`correction_filelist.yaml:19`) and `muSFFileList["2026"]`
    (`SF_filelist.yaml:120-124`) both point at 2025 `Run3-25Prompt` payloads; config states the real
    `MUO/Run3-26Prompt-Summer24-NanoAODv15/` cvmfs directory "is empty". Any 2026 run applies 2025
    muon momentum scale/smear and 2025 muon ID/Iso/Trig SF.

18. **2026 pileup weights are the 2025 file** (`pu_file_mc["2026"]`, `SF_filelist.yaml:68`;
    "no LUM/*2026* payload exists on cvmfs yet"). 2026 MC PU reweighting uses the 2025 data PU
    profile.

19. **2026 EBE mass-resolution BS calibration is a two-generation copy** (2026 <- 2025 <- 2024):
    `BS_res_calib_path["2026"]` MC and Data both = `res_calib_BS_correction_2024_Data_nanoAODv15.json`
    (`correction_filelist.yaml:55-59`). The "MC" slot again uses the *Data*-derived file (as 2024/2025,
    Unknown #4).

20. **2026 JER single-tag limitation is real and documented, not a placeholder.**
    `jer_tags["2026"] = Summer24Prompt26_RunBD_JRV1_MC` (`jec.yaml:786-791`): the framework cannot
    select the dedicated era-C low-PU JER `Summer24Prompt26_RunC_JRV1_MC`. Era-C data is
    `skip_sample: True`, so latent unless era C is re-enabled.

21. **2026 b-tag WPs are placeholders** (`btag_loose/medium_wp_UParT["2026"]` = `0.0246`/`0.1272`,
    "copied from 2025 (itself from 2024)", `jet.yaml:262, 277`). Affects `nBtagLoose/nBtagMedium`
    counting only (b-tag SF weight off).

22. **2026 MC is 2024-conditions Summer24 simulation** (every 2026 MC sample in
    `dataset_nanoAODv15_run3.yaml:1195-1493` reuses `RunIII2024Summer24NanoAODv15`;
    "2026-conditions MC not yet available"). Combined with real 2026 data + real 2026 JME payloads,
    2026 data/MC comparisons rely on the JME corrections absorbing the conditions difference and on
    2025-stand-in MUO/LUM calibrations. Needs validation before 2026 results are used.

23. **2026 lumi is preliminary.** `integrated_lumis["2026"] = 25950.0` (`lumi.yaml:15`), "Preliminary
    Offline Results"; per-era `lumi_pb` (`dataset_nanoAODv15_run3.yaml:1146,1160,1174,1188`) likewise.

24. **2026 has an `ext1` VBF sample pair** (`vbf_aMCatNLO`, `vbf_powheg` each list a `_ext1` dataset,
    `dataset_nanoAODv15_run3.yaml:1468, 1485`) — confirm stage-1 `sumGenWgts`/normalization sums base
    + ext1.

---

## Files inspected

- `src/copperhead_processor.py`
- `src/corrections/`: `rochester.py`, `MuonScaRe.py`, `muon_sf.py`, `fsr_recovery.py`, `jet.py`,
  `evaluator.py`, `pu_dnn.py`, `geofit.py`
- `src/lib/get_parameters.py`
- `modules/classify_year.py`
- `configs/parameters/`: `switches.yaml`, `SF_filelist.yaml`, `correction_filelist.yaml`,
  `jec.yaml`, `jet.yaml`, `muon.yaml`, `misc.yaml`, `lumi.yaml`, `trigger.yaml`
- `configs/datasets/dataset_nanoAODv15_run3.yaml`
- `run_stage1.py`
