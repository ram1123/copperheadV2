---
title: Run-3 HMuMu synchronisation ntuples (2024, Oct 2026)
---

# Run-3 HMuMu synchronisation ntuples — Purdue (copperheadV2)

First synchronisation exercise of the Run-3 H→µµ groups (request of 2026-10, due 2026-10-13).
Purdue covers ggF and VBF; ntuples are also provided for ttH/ZH/WH signals with the same definition.

## Inputs

| Sample key | Dataset(s) |
| --- | --- |
| `data_H` | `/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD`, `/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD` |
| `vbf_powheg` | `/VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` |
| `ggh_powhegPS` | `/GluGluH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` |
| `tth_powheg` | `/TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` |
| `zh_powheg` | `/ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` |
| `wph_powheg` | `/WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` |

Only the listed datasets are used (no `_ext1`). Config: `configs/datasets/sync_dataset_nanoAODv15_2024.yaml`.

## Ntuple preselection (the only cuts)

- Data: golden JSON `Cert_Collisions2024_378981_386951_Golden.json`.
- Muon pair: two muons with raw $p_T > 20$ GeV and $|\eta| < 2.4$, and $50 < m_{\mu\mu}^{raw} < 200$ GeV.
  "Raw" = beam-spot-constrained $p_T$ (NanoAOD `Muon_bsConstrainedPt` when `bsConstrainedChi2 < 30`, else `Muon_pt`),
  before ScaRe and FSR. This is the $p_T$ our analysis cuts on; the NanoAOD `Muon_pt` is stored as `mu*_pt_nano`.
- Pair choice: the two leading-raw-$p_T$ muons among the analysis-selected muons (ID+iso, below) if at least two pass,
  otherwise among muons passing only the preselection (`mu_pair_from_selected` tells which). This is identical to the
  analysis pair whenever the event passes the analysis selection.
- An event is also kept if the *IP pair* (below) passes the same window, even when the default pair does not;
  `pass_analysis_preselection` / `pass_analysis_preselection_ip` each include their own pair's window.

Everything else is stored as flags. `pass_analysis_preselection` reproduces exactly the event-level selection of our
standard stage-1 (checked event-by-event and column-by-column against the standard output).

### Lepton/IP-veto variant (`*_ip` branches)

`pass_analysis_preselection_ip` reproduces our stage-1 with switch `do_lepton_ip_veto: true` (checked event-by-event
on one data_H and one ZH file: identical event sets). Differences from the default selection:
- muons: $|d_{xy}| < 0.05$ cm, $|d_z| < 0.1$ cm on top of ID+iso; the pair is the two leading-raw-$p_T$ such muons
  (`mu_pair_ip_is_sync_pair` = same two muons as the default pair; `mu*_` branches always describe the default pair,
  the IP pair only through `dimuon_mass_raw_ip`, `dimuon_mass_ip`, `dimuon_pt_ip`);
- extra-muon veto: exactly 2 loose muons (looseId, raw $p_T > 10$ GeV, $|\eta| < 2.4$, iso < 0.25, same IP cuts);
- electron veto from $p_T > 10$ GeV with `pfRelIso03_all` < 0.25 and IP cuts (barrel $|d_{xy}|<0.05$, $|d_z|<0.1$;
  endcap $|d_{xy}|<0.1$, $|d_z|<0.2$ cm, split on supercluster $|\eta| = 1.479$).
Cut values: `configs/parameters/{muon,electron}.yaml`. Jet-level quantities are computed with the default pair.

## Content (TTree `Events`, one entry per event)

| Group | Branches |
| --- | --- |
| Event | `run`, `luminosityBlock`, `event`, `sample_id` (0 Muon0, 6 Muon1, 1 VBF, 2 ggH, 3 ttH, 4 ZH, 5 WplusH), `PV_npvs`, `PV_npvsGood` |
| Trigger | `HLT_IsoMu24`, `pass_HLT`, `pass_trigger_match` (≥1 of the two leading selected muons matched, ΔR<0.4 to a muon TrigObj with filterBit 8, and ScaRe $p_T$ ≥ 26 GeV), `mu{1,2}_trigMatched` |
| Event flags | `pass_MET_filters`, `pass_PV`, `pass_nmuons` (exactly 2 selected muons), `pass_OS`, `nelectrons`, `pass_electron_veto`, `pass_jet_veto_map`, `pass_analysis_preselection`, `nmuons_selected`, `nmuons_presel` |
| IP-veto flags | `pass_analysis_preselection_ip`, `pass_muon_ip` (default pair passes IP), `mu{1,2}_passIP`, `nmuons_selected_ip`, `nloose_muons_ip`, `pass_nmuons_ip`, `pass_OS_ip`, `pass_trigger_match_ip`, `nelectrons_ip`, `pass_electron_veto_ip`, `mu_pair_ip_is_sync_pair`, `dimuon_mass_raw_ip`, `dimuon_mass_ip`, `dimuon_pt_ip` |
| Category | `pass_bveto`, `pass_vbf_cuts`, `cat_vbf`, `cat_ggh`; mass regions `z_peak`, `h_sidebands`, `h_peak` (on corrected mass) |
| Muons | `mu{1,2}_charge`, `_eta`, `_phi` (after FSR), `_eta_raw`, `_phi_raw`, `_pt_nano`, `_pt_bsc` (raw), `_pt_scare` (after ScaRe), `_pt_corr` (after ScaRe+FSR, = `mu*_pt`), `_mediumId`, `_isGlobal`, `_isTracker`, `_passId`, `_pfRelIso04_all_raw`, `_pfRelIso04_all_fsrCorr`, `_passIso`, `_passSelection`, `_hasFsr`, `_dxy`, `_dz`, ... ; mu1 = leading in corrected $p_T$ |
| Dimuon | `dimuon_mass_raw`/`dimuon_pt_raw` (before corrections), `dimuon_mass_scare`/`_pt_scare`, `dimuon_mass_corr`/`_pt_corr` (after ScaRe+FSR, = `dimuon_mass`/`dimuon_pt`) |
| Jets (all NanoAOD jets, $p_T$-sorted after JEC+JER) | `nJet`, `Jet_pt`, `Jet_eta`, `Jet_phi`, `Jet_mass`, `Jet_pt_raw` (before JEC), `Jet_pt_jec` (after JEC, before JER), `Jet_rawFactor`, `Jet_passJetId`, `Jet_isCleanFromMuons`, `Jet_passHornPtCut`, `Jet_inJetVetoMap`, `Jet_isSelected`, `Jet_isBtagCandidate`, `Jet_btagUParTAK4B`, `Jet_passBtagLoose`, `Jet_passBtagMedium`, MC: `Jet_hadronFlavour`, `Jet_genJetIdx` |
| Selected jets / dijet | `njets_nominal`, `jj_idx1`, `jj_idx2` (index into `Jet_*` of the selected pair = two leading selected jets, -1 if none), `jet{1,2}_*_nominal`, `jj_mass_nominal`, `jj_pt_nominal`, `jj_eta_nominal`, `jj_dEta_nominal`, `nBtagLoose_nominal`, `nBtagMedium_nominal` |
| MET | `PuppiMET_pt`, `PuppiMET_phi`, `PuppiMET_sumEt` — NanoAOD PuppiMET, no re-correction |
| MC weights | `separate_wgt_genWeight`, `separate_wgt_genWeight_normalization` (1/Σw), `separate_wgt_xsec`, `separate_wgt_lumi` (full 2024, 109.82 fb⁻¹), `separate_wgt_pu`, `separate_wgt_muID`, `separate_wgt_muIso`, `separate_wgt_muTrig`, `separate_wgt_nnlops` (ggH), LHE/THU entries are 1 (variations only); `wgt_nominal` = product. No DY/dimuon-$p_T$ reweighting (signals only). Data: `separate_wgt_ones`. |

Missing values (e.g. no second jet) are `-999`. Booleans are stored as `int8`. File-level JSON metadata is in the
`TObjString` `sync_info` (DAS dataset, git commit, configs). Data are processed as one sample (`data_H`) and split
into Muon0 / Muon1 outputs by input file.

FatJets are not stored (Purdue does not run the VH-hadronic boosted channel).

## Analysis selection outline (separate from the ntuple preselection)

Full table with corrections and payloads: `docs/Run3_selection_and_samples.md` (official baseline variant).
For 2024 (`configs/switches/switches_official.yaml`, `configs/parameters/*.yaml`):

- Trigger `HLT_IsoMu24`; trigger matching as above.
- Muons: beam-spot constraint → ScaRe (MUO correctionlib `muon_scalesmearing`) → FSR recovery; raw $p_T > 20$ GeV,
  $|\eta| < 2.4$, `mediumId`, global or tracker, `pfRelIso04_all` < 0.25 (FSR photons removed); exactly two selected
  muons, opposite charge; no muon IP cuts, no extra loose-lepton veto in the baseline.
- Electron veto: `mvaIso_WP90`, $p_T > 20$ GeV, $|\eta| < 2.5$ excluding 1.44–1.57.
- MET filters: goodVertices, globalSuperTightHalo2016Filter, EcalDeadCellTriggerPrimitiveFilter, BadPFMuonFilter,
  BadPFMuonDzFilter, hfNoisyHitsFilter, eeBadScFilter, ecalBadCalibFilter (+ ECAL bad-calib recipe); ≥1 good PV.
- Jets: AK4 PUPPI, JEC + JER (JME Run-3 recipe, `jer_strat=4`), $p_T > 25$ GeV, $|\eta| < 4.7$, `tightPassLepVeto`
  jet ID, ΔR(j, µ) > 0.4 to the two muons, horn cut $p_T > 50$ GeV for $2.5 < |\eta| \le 3.0$ (2024).
- Jet veto map `Summer24Prompt24_RunBCDEFGHI_V1`: event rejected if any jet with $p_T > 15$, tightLepVeto ID,
  EM fraction < 0.9 is in the vetoed region.
- **B-tag: UParTAK4 (`btagUParTAK4B`), loose WP 0.0246, medium WP 0.1272 (BTV Summer24)**; b-tag jets
  $p_T > 25$ GeV, $|\eta| < 2.5$, tight ID, muon-cleaned. B-veto: reject if $N_b^{L} \ge 2$ or
  ($N_b^{M} \ge 1$ and $N_{jets} \ge 2$).
- VBF: $m_{jj} > 400$ GeV, $|\Delta\eta_{jj}| > 2.5$, leading-jet $p_T > 35$ GeV (+ b-veto); ggH: rest (+ b-veto).

## Differences from the standard analysis configuration

- Selection stored as flags instead of cuts (switch `do_sync_ntuple`, `configs/switches/switches_official_sync.yaml`;
  all other switches identical to `switches_official.yaml`).
- Extra preselection $50 < m_{\mu\mu}^{raw} < 200$ GeV (standard stage-1 has no mass cut).
- When fewer than two muons pass ID/iso, the pair is formed from preselected muons (not used in the analysis).
- `_ext1` signal datasets not used (standard analysis includes them); MC normalised with the non-ext Σw.

## Reproduce

```bash
./run_in_pixi.sh default python run_prestage.py --chunksize 600000 -y 2024 --NanoAODv 15 \
    --yaml configs/datasets/sync_dataset_nanoAODv15_2024.yaml --data H --signal Higgs \
    --use_gateway --prestage_output ./prestage_output_sync2024
cp prestage_output_sync2024/processor_samples_2024_NanoAODv15.json prestage_output/processor_samples_2024_NanoAODv15_sync2024.json
./run_in_pixi.sh default python -W ignore run_stage1.py -y 2024 --NanoAODv 15 --max_file_len 900 \
    --yaml configs/datasets/sync_dataset_nanoAODv15_2024.yaml --switches-yaml configs/switches/switches_official_sync.yaml \
    --prestage-tag sync2024 --save_path /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_sync2024 --use_gateway
./run_in_pixi.sh default python scripts/make_sync_ntuple.py \
    --stage1 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_sync2024/stage1_output/2024 \
    --out /depot/cms/hmm/shar1172/hmm_sync/run3_sync_2024/Purdue
```

Output directories follow the requested `<GROUP>-<DATASETNAME>` mapping, e.g.
`Purdue-VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/`, `Purdue-Muon0_Run2024H-MINIv6NANOv15-v1/`, and are
copied with `xrdcp` to `/eos/cms/store/group/phys_higgs/cmshmm/run3_sync/Purdue/` (same layout as the other groups).
