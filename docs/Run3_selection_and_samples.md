---
title: Run-3 Selection, Corrections and Samples
---

# Run3 $H \rightarrow \mu\mu$ synchronization Purdue


# Object Selection

### Muon Selection & Corrections

| Object                | Selection / Correction Criterion                                                   |
| --------------------- | ---------------------------------------------------------------------------------- |
| **Muon**              | Beam-spot constrained muon $p_T$ *(revert to default if beam-spot $\chi^2 \geq 30$)* |
| -                     | Muon scale & resolution correction (MUO Run-3 correctionlib `muon_scalesmearing`) |
| -                     | Trigger: `HLT_IsoMu24` (all Run-3 eras)                                            |
| -                     | At least one of the two leading muons matched ($\Delta R < 0.4$) to the HLT muon object, with $p_T > 26$ GeV |
| -                     | $p_T > 20$ GeV, $\lvert\eta\rvert < 2.4$                                           |
| -                     | Pass Medium ID, global or tracker muon                                             |
| -                     | Loose PF isolation: `pfRelIso04_all` < 0.25 (FSR photons removed from the isolation sum) |
| -                     | Impact parameter: $\lvert d_{xy}\rvert < 0.05$ cm and $\lvert d_{z}\rvert < 0.10$ cm |
| -                     | At least 2 selected muons, leading pair with opposite charge                       |
| Loose-muon veto       | Exactly 2 loose muons in the event: loose muon = $p_T > 10$ GeV, $\lvert\eta\rvert < 2.4$, Loose ID, `pfRelIso04_all` < 0.25, same IP cuts as above *(the two selected muons are a subset, so any extra muon vetoes the event)* |
| -                     | Number of good primary vertices > 0                                                |
| -                     | FSR recovery                                                                       |
| Filters/Corrections   | Correction order: Beam-spot → Scale & resolution → FSR                            |

*IP cuts and the loose-lepton veto (muon above, electron below) are enabled with `do_lepton_ip_veto: true`
(`configs/switches/switches_official_lepton_ip_veto.yaml`); cut values are in `configs/parameters/muon.yaml`.
No MUO scale factor covers the IP cuts yet.*


### Electron Selection

| Object       | Selection / Correction Criterion                                            |
| ------------ | --------------------------------------------------------------------------- |
| **Electron** | $p_T >$ 10 GeV and $\lvert\eta\rvert < 2.5$                                  |
| -     | Ignore electrons in 1.44 < $\lvert\eta\rvert$ < 1.57                                 |
| -     | Pass WP90 MVA-based ID (`mvaIso_WP90`)                                               |
| -     | Isolation: `pfRelIso03_all` < 0.25                                                   |
| -     | Impact parameter: barrel ($\lvert\eta + \Delta\eta_{SC}\rvert < 1.479$) $\lvert d_{xy}\rvert < 0.05$ cm, $\lvert d_{z}\rvert < 0.10$ cm; endcap $\lvert d_{xy}\rvert < 0.10$ cm, $\lvert d_{z}\rvert < 0.20$ cm |
| -     | Veto events with electrons passing this pre-selection                                |


### Jet Selection & Corrections

Common to all jet selections:

| Object                  | Selection / Correction Criterion                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Jet**                 | AK4 PUPPI jets with $p_T > 25$ GeV and $\lvert\eta\rvert < 4.7$                                                       |
| -                       | Tight Lepton Veto jet ID (JME Run-3 recipe)                                                                          |
| -                       | Clean from selected muons ($\Delta R > 0.4$)                                                                        |
| Corrections             | JEC applied; JER smearing only for gen-matched jets, unmatched jets not smeared *(except variant 2b given below: no JER smearing)* |
| B-tag                   | B-tag jets: $p_T > 25$ GeV, $\lvert\eta\rvert < 2.5$, Tight jet ID, cleaned from selected muons                     |
| B-tag                   | B-tag: loose and medium WPs of ParticleNet (2022, 2023; NanoAODv12) or UParT (2024, 2025, 2026; NanoAODv15)                     |
| Filters/Corrections     | Jet veto maps (JVM): only jets with $p_T > 15$ GeV, Tight Lepton Veto ID and EM fraction < 0.9 are checked; event veto or jet veto depending on the variant (below) |

Forward (HE/HF) jet treatment and jet-veto-map (JVM) mode — variants:

| Selection                        | Criterion                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **1) Official (baseline)**       | 2022preEE, 2022postEE, 2023, 2023BPix: jets with $\lvert\eta\rvert > 2.5$ (HE + HF) require $p_T > 50$ GeV |
| -                                | 2024: jets with $2.5 < \lvert\eta\rvert \leq 3.0$ (HE only) require $p_T > 50$ GeV; no extra cut in HF       |
| -                                | 2025, 2026:  HE $p_T > 50$ GeV |
| -                                | JVM: **reject the whole event** if any checked jet is in the veto region                                    |
| -                                | Config: `configs/switches/switches_official.yaml`                                                           |
| **2) PU DNN** (both variants)    | No extra forward $p_T$ cut; instead reject pileup jets with a DNN in HE ($2.5 \leq \lvert\eta\rvert \leq 3.0$) and HF ($\lvert\eta\rvert > 3.0$), separate models per region and $\eta$ sign |
| -                                | Applied only to jets with $25 \leq p_T < 50$ GeV; working point = 80% hard-scatter jet efficiency          |
| -                                | Models: 2022/2023 eras use the 2022postEE-trained model (no `puIdDisc` input); 2024 uses the 2024-trained model (includes `puIdDisc`); 2025, 2026 reuse the 2024 model (placeholder until dedicated models exist) |
| **2a) PU DNN, JVM event veto**   | JVM: **reject the whole event** (as in the official selection); JER as official                            |
| -                                | Config: `configs/switches/switches_JetPt25GeV_PUDNN_EventFilter_JetStrat4.yaml`                             |
| **2b) PU DNN, JVM jet veto**     | JVM: **reject only the jets** in the veto region and keep the event (PuppiMET $p_T$ and $\Sigma E_T$ set to 0 in such events); no JER smearing |
| -                                | Config: `configs/switches/switches_JetPt25GeV_PUDNN.yaml`                                                   |

*Both PU-DNN switches files also turn off the NNLOPS and THU signal weights, which do not change the selection.*


### Correction files

Each payload below is byte-identical (md5) to the listed `/cvmfs` file (2022–2024 are read from local copies under `data/`;
2025/2026 mostly straight from `/cvmfs`, the 2025 muon scale & resolution from `data/roch_corr/`). 2025 and 2026 use Summer24 MC.
Source configs: `configs/parameters/{correction_filelist,SF_filelist,jec,jet,lumi}.yaml`.

| Correction | Year | `/cvmfs` path |
| --- | --- | --- |
| Muon scale & resolution | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz` |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz` |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-11-27/muon_scalesmearing.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_scalesmearing_VXBS.json.gz` |
| - | 2026 | same as 2025  |
| Muon ID / Iso / Trigger SF (`muon_Z`) | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_Z.json.gz` |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_Z.json.gz` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_Z.json.gz` |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_Z.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-11-27/muon_Z.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_Z.json.gz` |
| - | 2026 | same as 2025  |
| Pileup weights | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22CDSep23-Summer22-NanoAODv12/2024-01-31/puWeights.json.gz` |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2024-01-31/puWeights.json.gz` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23CSep23-Summer23-NanoAODv12/2024-01-31/puWeights.json.gz` |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23DSep23-Summer23BPix-NanoAODv12/2024-01-31/puWeights.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/puWeights_BCDEFGHI.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-05/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz` |
| - | 2026 | same as 2025 *(placeholder: no 2026 LUM payload yet)* |
| JEC / JER (`jet_jerc`) | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jet_jerc.json.gz` |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jet_jerc.json.gz` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jet_jerc.json.gz` |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jet_jerc.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-07-16/jet_jerc.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16/jet_jerc.json.gz` |
| - | 2026 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jet_jerc.json.gz` |
| JER smearing formula | all | `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/jer_smear.json.gz` |
| Jet ID | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jetid.json.gz` |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetid.json.gz` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetid.json.gz` |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetid.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/jetid.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16/jetid.json.gz` |
| - | 2026 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetid.json.gz` |
| Jet veto maps | 2022preEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jetvetomaps.json.gz` (`Summer22_23Sep2023_RunCD_V1`) |
| - | 2022postEE | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetvetomaps.json.gz` (`Summer22EE_23Sep2023_RunEFG_V1`) |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetvetomaps.json.gz` (`Summer23Prompt23_RunC_V1`) |
| - | 2023BPix | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetvetomaps.json.gz` (`Summer23BPixPrompt23_RunD_V1`) |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-07-17/jetvetomaps.json.gz` (`Summer24Prompt24_RunBCDEFGHI_V1`) |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16/jetvetomaps.json.gz` (`Summer24Prompt25_RunCDEFG_V1`) |
| - | 2026 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15/jetvetomaps.json.gz` (`Summer24Prompt26_RunBCD_V1`) |
| B-tag SF (not applied: b-tag weights off) | 2022preEE | `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz` |
| - | 2022postEE | `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz` |
| - | 2023 | `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23/btagging.json.gz` |
| - | 2023BPix | `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23BPix/btagging.json.gz` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-08-19/btagging.json.gz` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26/btagging.json.gz` |
| - | 2026 | same as 2025 *(placeholder: no 2026 BTV payload yet)* |
| Golden JSON | 2022 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions22/latest/Cert_Collisions2022_355100_362760_Golden.json` |
| - | 2023 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions23/latest/Cert_Collisions2023_366442_370790_Golden.json` |
| - | 2024 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions24/2026-05-27/Cert_Collisions2024_378981_386951_Golden.json` |
| - | 2025 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions25/2026-05-27/Cert_Collisions2025_391658_398903_Golden.json` |
| - | 2026 | `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions26/latest/Cert_Collisions2026_401624_403937_golden.json` |

Not on `/cvmfs` (analysis-specific): beam-spot resolution calibration (`data/res_calib/`), Z-$p_T$ reweighting
(`data/zpt_rewgt/`), NNLOPS (`data/NNLOPS_reweight.root`), PU-DNN models (`validation/pu_dnn/`).



# Event Categorization

Applied in stage-2 after the object/event selection above. All cuts use the nominal jets (leading two jets after
the jet selection, including the forward-jet treatment of the chosen variant). Numbers follow AN-19-124 (L827/L830).

### Vetoes common to the categories

| Cut | Criterion |
| --- | --- |
| b-jet veto | Reject the event if $N_{b}^{\text{loose}} \geq 2$, or $N_{b}^{\text{medium}} \geq 1$ and $N_{\text{jets}} \geq 2$ (b-tag jets and WPs as in the jet table) |
| VH veto | Not applied (optional: no fat jet away from the muons and $p_T^{\text{miss}} < 150$ GeV) |

### Categories

| Category | Definition |
| --- | --- |
| **VBF** | $m_{jj} > 400$ GeV and $\lvert\Delta\eta_{jj}\rvert > 2.5$ and leading-jet $p_T > 35$ GeV, and pass the b-jet veto |
| **ggH** | Fail the VBF requirement (any jet multiplicity, including 0 and 1 jet), and pass the b-jet veto |

VBF and ggH are exclusive and together equal the b-jet-vetoed sample. Events failing the b-jet veto (VH/ttH-like)
are not used in either category.

### Mass regions

| Region | $m_{\mu\mu}$ [GeV] |
| --- | --- |
| Signal region (h-peak) | $115 \leq m_{\mu\mu} < 135$ |
| Sidebands (h-sidebands) | $110 \leq m_{\mu\mu} < 115$ or $135 \leq m_{\mu\mu} < 150$ |
| Z-peak (control) | $70 \leq m_{\mu\mu} < 110$ |

### Sub-categorization and fit

| Channel | Discriminant and fit |
| --- | --- |
| **ggH** | BDT score in the 110–150 GeV window; events split into BDT-score sub-categories using per-year bin edges (`BDT_edges.yaml` given to `run_stage2.py --edge_cfg_path`); signal extracted from a fit to $m_{\mu\mu}$ in each sub-category |
| **VBF** | DNN score (`dnn_vbf_score_atanh`), trained in the h-peak region; binned per production and dijet-$\eta$ region (`configs/MVA/VBF/dnn_binning.yaml`); template fit to the DNN-score distribution in the signal region and sidebands |

VBF dijet-$\eta$ regions, defined on the two leading jets (central = $\lvert\eta\rvert \leq 2.5$, forward = $\lvert\eta\rvert > 2.5$).
The three regions are mutually exclusive and together cover the full VBF category:

| Region | Code name | Definition |
| --- | --- | --- |
| Both central | `jj_both_central` | Both jets $\lvert\eta\rvert \leq 2.5$ |
| One central, one forward | `jj_one_fwd25_one_central` | One jet $\lvert\eta\rvert \leq 2.5$, the other $\lvert\eta\rvert > 2.5$ |
| Both forward | `jj_both_fwd25` | Both jets $\lvert\eta\rvert > 2.5$ |

A DNN is trained and binned per region, and should only be applied in the region it was trained on. 



# Integrated Luminosity



| Year       | Integrated Luminosity [pb⁻¹] | Source                                                                                                                                                                                                                                     |
| ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2022preEE  | 7980.4                       | —                                                                                                                                                                                                                                          |
| 2022postEE | 26671.70                     | —                                                                                                                                                                                                                                          |
| 2023       | 17794.0                      | https://docs.google.com/presentation/d/1TjPem5jX0fzqvTGl271_nQFoVBabsrdrO0i8Qo1uD5E/edit?slide=id.g289f499aa6b_2_58                                                                                                                         |
| 2023BPix   | 9451.0                       | —                                                                                                                                                                                                                                          |
| 2024       | 108960.0                     | https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0                                                                                                                           |
| 2025       | 110840.0                     | https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#Year_2025 (eras B–G) |
| 2026       | 25950.0                      | `configs/parameters/lumi.yaml` (eras A, B, D; era C low-pileup run excluded, preliminary) |


The values actually used by the code are in [configs/parameters/lumi.yaml](../configs/parameters/lumi.yaml) (`integrated_lumis`); the 2022–2024 rows above are older numbers and differ slightly from it (e.g. 2024 = 109820 pb⁻¹ there).

**2024 and 2025/26 (PC guideline, Aug 2026):** 2025 and 2026 are treated as one year, and the Summer24 MC is shared between 2024 and 2025/26, split by luminosity fraction. See [2024 / 2025-26 Year Grouping and MC Split](Run3_2024_2025_2026_MC_split.md).


# Lumimask

| Year       | Lumimask     | Reference                                                                                       |
|------------|--------------|-------------------------------------------------------------------------------------------------|
| 2022preEE  | Golden JSON  | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4                                                                                               |
| 2022postEE | Golden JSON  | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4                                                                                               |
| 2023       | Golden JSON  | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN2                                                                                               |
| 2023BPix   | Golden JSON  | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN2                                                                                               |
| 2024       | Golden JSON  | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN1                              |
| 2025       | Golden JSON  | https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/Cert_Collisions2025_391658_398903_Golden.json |
| 2026       | Golden JSON  | https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions26/Cert_Collisions2026_401624_403937_golden.json |

# MET/ Noise Filters

[Official recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#Run_3_2022_and_2023_data_and_MC)


# Sample Tables

NOTE: We take the cross section (Xsec) from the MC generator and depending on the order of the calculation we either multiply it with a k-factor or just use the Xsec calculated by a dedicated program. If a k-factor is mentioned, please multiply it with the Xsec to obtain the final Xsec:

$Xsec_{final} = kFactor*Xsec$


# 2022preEE
NanoAOD version: v12: [data](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_EGamma_Run2022F-22Sep2023-v1.html), [MC](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html)
## Data
| ERA | Dataset Paths | Golden JSON Luminosity [fb⁻¹] | Notes / Source |
|-----|----------------|-------------------------------|----------------|
| C   | /Muon/Run2022C-16Dec2023-v1/NANOAOD<br>/SingleMuon/Run2022C-16Dec2023-v1/NANOAOD | 5.0104 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4 |
| D   | /Muon/Run2022D-16Dec2023-v1/NANOAOD | 2.9700 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4 |
| **Total** | — | **7.9804** | - |


## MC

| **Production mode**       | **Cross section [pb]** | **Perturbative Order**    | **k-factor** | **Reference**                   | **Datasets** |
|---------------------------|------------------------|----------------------------|--------------|----------------------------------|--------------|
| ggH                       | 0.0110834              | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |
| VBF                       | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |
| EWK                       | 1.421                  | LO                         | –            | GenXAnalyzer script              | /EWK_2L2J_TuneCH3_13p6TeV_madgraph-herwig7/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM |
| DYto2L_incl (M-50)        | 5991.00                | NNLO QCD + NLO EW          | –            | DY Turbo calculation from Filipo  (Pisa)           | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v5/NANOAODSIM<br>/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| dyTo2L_M-50_2jets         | 359.05                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |
| dyTo2L_M-50_1jets         | 952.29                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |
| dyTo2L_M-50_0jets         | 5034.65                | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM |


| **Production mode** | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference** | **Datasets** |
|----------------------|------------------------|--------------------------|---------------|----------------|--------------|
| ZZ_4l               | 1.39                   | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| ZZ_2l2q             | 6.788                  | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| ZZ_2l2nu            | 1.031                  | NLO                      | 1.16          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| WZ_1l1nu2q          | 15.87                  | NLO                      | 1.00          | XS from xsecdb | — |
| WZ_2l2q             | 7.568                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| WZ_3lnu             | 4.924                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3/NANOAODSIM |
| WW_2l2nu            | 11.79                  | NLO                      | –             | XS from xsecdb | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |
| TTto4Q              | 419.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | — |
| TTto2L2Nu           | 98.04                  | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | — |
| TTtoLNu2Q           | 405.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | — |
| TT_inclusive (remove all other TT samples if using this sample)       | 923.6                 | NNLO + NNLL              | 1.0           |  https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO | /TT_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM<br>/TT_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM |



# 2022postEE
NanoAOD version: v12: [data](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_EGamma_Run2022F-22Sep2023-v1.html), [MC](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html)
## Data
| ERA | Dataset Paths | Golden JSON Luminosity [fb⁻¹] | Notes / Source |
|-----|----------------|-------------------------------|----------------|
| E   | /Muon/Run2022E-16Dec2023-v1/NANOAOD | 5.8070 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4 |
| F   | /Muon/Run2022F-16Dec2023-v1/NANOAOD | 17.7819 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4 |
| G   | /Muon/Run2022G-16Dec2023-v1/NANOAOD | 3.0828 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN4 |
| **Total** | — | **26.6717** | - |


## MC

| **Production mode**       | **Cross section [pb]** | **Perturbative Order**    | **k-factor** | **Reference**                   | **Datasets** |
|---------------------------|------------------------|----------------------------|--------------|----------------------------------|--------------|
| ggH                       | 0.0110834              | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM |
| VBF                       | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBFHto2Mu_M-125_TuneCP5_withDipoleRecoil_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM |
| EWK                       | 1.421                  | LO                         | –            | GenXAnalyzer script              | /EWK_2L2J_TuneCH3_13p6TeV_madgraph-herwig7/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM |
| DYto2L_incl (M-50)        | 5991.00                | NNLO QCD + NLO EW          | –            | DY Turbo calculation from Filipo (Pisa)                    | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5/NANOAODSIM<br>/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| dyTo2L_M-50_2jets         | 359.05                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM |
| dyTo2L_M-50_1jets         | 952.29                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM |
| dyTo2L_M-50_0jets         | 5034.65                | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM |


| **Production mode** | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference** | **Datasets** |
|----------------------|------------------------|--------------------------|---------------|----------------|--------------|
| ZZ_4l               | 1.39                   | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| ZZ_2l2q             | 6.788                  | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| ZZ_2l2nu            | 1.031                  | NLO                      | 1.16          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| WZ_1l1nu2q          | 15.87                  | NLO                      | 1.00          | XS from xsecdb | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| WZ_2l2q             | 7.568                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| WZ_3lnu             | 4.924                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| WW_2l2nu            | 11.79                  | NLO                      | –             | XS from xsecdb | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| TTto4Q              | 419.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| TTto2L2Nu           | 98.04                  | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| TTtoLNu2Q           | 405.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM<br>/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM |
| TT_inclusive (remove all other TT samples if using this sample) | 923.6 | NNLO + NNLL | 1.0 | https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO | — |



# 2023
NanoAOD version: v12: [data](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_EGamma_Run2022F-22Sep2023-v1.html), [MC](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html)
## Data
| Key       | Dataset Paths | Golden JSON Luminosity [fb⁻¹] | Notes / Source |
|-----------|----------------|-------------------------------|----------------|
| data_C    | /Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD<br>/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD<br>/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD<br>/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD<br>/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD<br>/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD<br>/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD<br>/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD | 18.063 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN2 |
| **Total** | —             | **18.063**                    | - |


## MC

| **Production mode**       | **Cross section [pb]** | **Perturbative Order**    | **k-factor** | **Reference**                   | **Datasets** |
|---------------------------|------------------------|----------------------------|--------------|----------------------------------|--------------|
| ggH                       | 0.0110834              | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM |
| VBF                       | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBFHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM |
| EWK                       | 1.421                  | LO                         | –            | GenXAnalyzer script              | /EWK_2L2J_TuneCH3_13p6TeV_madgraph-herwig7/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM |
| DYto2L_incl (M-50)        | 5991.00                | NNLO QCD + NLO EW          | –            | DY Turbo calculation from Filipo (Pisa)                    | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| dyTo2L_M-50_2jets         | 359.05                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM |
| dyTo2L_M-50_1jets         | 952.29                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM |
| dyTo2L_M-50_0jets         | 5034.65                | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM |


| **Production mode** | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference** | **Datasets** |
|----------------------|------------------------|--------------------------|---------------|----------------|--------------|
| ZZ_4l               | 1.39                   | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM |
| ZZ_2l2q             | 6.788                  | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM |
| ZZ_2l2nu            | 1.031                  | NLO                      | 1.16          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM |
| WZ_1l1nu2q          | 15.87                  | NLO                      | 1.00          | XS from xsecdb | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| WZ_2l2q             | 7.568                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM |
| WZ_3lnu             | 4.924                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM<br>/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15_ext1-v2/NANOAODSIM |
| WW_2l2nu            | 11.79                  | NLO                      | –             | XS from xsecdb | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM |
| TTto4Q              | 419.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| TTto2L2Nu           | 98.04                  | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| TTtoLNu2Q           | 405.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM |
| TT_inclusive (remove all other TT samples if using this sample) | 923.6 | NNLO + NNLL | 1.0 | https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO | — |


# 2023BPix
NanoAOD version: v12: [data](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_EGamma_Run2022F-22Sep2023-v1.html), [MC](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html)
## Data
| Key     | Dataset Paths | Golden JSON Luminosity [fb⁻¹] | Notes / Source |
|---------|----------------|-------------------------------|----------------|
| data_D  | /Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD<br>/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD<br>/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD<br>/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD | 9.693 | https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA_AN2 |
| **Total** | —           | **9.693**                     | - |



## MC
| **Production mode**       | **Cross section [pb]** | **Perturbative Order**    | **k-factor** | **Reference**                   | **Datasets** |
|---------------------------|------------------------|----------------------------|--------------|----------------------------------|--------------|
| ggH                       | 0.0110834              | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /GluGluHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM |
| VBF                       | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBFHto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM |
| EWK                       | 1.421                  | LO                         | –            | GenXAnalyzer script              | /EWK_2L2J_TuneCH3_13p6TeV_madgraph-herwig7/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM |
| DYto2L_incl (M-50)        | 5991.00                | NNLO QCD + NLO EW          | –            | DY Turbo calculation from Filipo (Pisa)                    | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM |
| dyTo2L_M-50_2jets         | 359.05                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5/NANOAODSIM |
| dyTo2L_M-50_1jets         | 952.29                 | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM |
| dyTo2L_M-50_0jets         | 5034.65                | NNLO QCD + NLO EW          | –            | XS from SMP-25-013               | /DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM |

| **Production mode** | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference** | **Datasets** |
|----------------------|------------------------|--------------------------|---------------|----------------|--------------|
| ZZ_4l               | 1.39                   | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto4L_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| ZZ_2l2q             | 6.788                  | NLO                      | 1.19          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| ZZ_2l2nu            | 1.031                  | NLO                      | 1.16          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| WZ_1l1nu2q          | 15.87                  | NLO                      | 1.00          | XS from xsecdb | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM |
| WZ_2l2q             | 7.568                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM |
| WZ_3lnu             | 4.924                  | NLO                      | 1.08          | XS from xsecdb, kfactor from Maxtrix (SMP-25-013) | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM<br>/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6_ext1-v2/NANOAODSIM |
| WW_2l2nu            | 11.79                  | NLO                      | –             | XS from xsecdb | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| TTto4Q              | 419.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| TTto2L2Nu           | 98.04                  | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| TTtoLNu2Q           | 405.69                 | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM |
| TT_inclusive (remove all other TT samples if using this sample) | 923.6 | NNLO + NNLL | 1.0 | https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO | — |

# 2024
NanoAOD version: v15: [data](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv15/2024/doc_EGamma1_Run2024F-MINIv6NANOv15-v1.html), [MC](https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv15/2024/doc_TTH-Hto2G_Par-M-125_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2.html)
## Data

| Key       | Dataset Paths                                                                                  | Luminosity (fb⁻¹) | Notes / Source                                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| data_C    | `/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD`       | 7.24              | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_D    | `/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD`       | 7.96              | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_E    | `/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD`       | 11.32             | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_F    | `/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD`       | 27.76             | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_G    | `/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD`       | 37.77             | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_H    | `/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD`<br>`/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD`       | 5.44              | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_I    | `/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD`<br>`/Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 11.47             | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| **Total** | —                                                                                              | **108.96**        | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |


## MC



| **Production mode**  | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference**                                                                                                                                                                                                                                       | **Datasets**                                                                                                                                                                                                                                                         |
| -------------------- | ---------------------- | ---------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ggH                  | 0.0110834              | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /GluGluH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM                                                                                                                                    |
| VBF  powheg          | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM /VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM   |
| VBF  aMCatNLO        | 0.00088319             | N3LO QCD + NLO EW | –            | σ × BR: [LHCHXSWG1 R5 v1.1.3](https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections) (13.6 TeV, $m_H$=125.38) × YR4 BR($\mu\mu$)=2.1542e-4 | /VBF-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM /VBF-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM | 
| EWK mmjj mll 105_160 | 0.06445                | LO                     | –            | GenXAnalyzer script                                                                                                                                                                                                                                 | /EWK-2Mu2J_Bin-MLL-105to160_TuneCH3_13p6TeV_madgraph-herwig7/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM                                                                                                                                |
| DYto2Mu_incl (M-50)  | 1997.0                 | NNLO QCD + NLO EW      | –            | DY Turbo calculation from Filipo (Pisa)                                                                                                                                                                                                             | /DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6/NANOAODSIM                                                                                                                              |
| dyTo2Mu_M-10To50     | 6744                   | –                      | –            | [XSDB: DYto2Mu_Bin-MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8](https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=process_name%20%3D%20DYto2Mu_Bin-MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8)     | /DYto2Mu_Bin-MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM                                                                                                                                      |
| dyTo2Mu_M-50To120    | 2219                   | –                      | –            | [XSDB: DYto2Mu_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8](https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=process_name%20%3D%20DYto2Mu_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8)   | /DYto2Mu_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM                                                                                                                                     |
| dyTo2Mu_M-120To200   | 21.65                  | –                      | –            | [XSDB: DYto2Mu_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8](https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=process_name%20%3D%20DYto2Mu_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8) | /DYto2Mu_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM                                                                                                                                    |

| **Production mode** | **Cross section [pb]** | **Perturbative Order** | **k-factor** | **Reference** | **Datasets** |
|----------------------|------------------------|--------------------------|---------------|----------------|--------------|
| TTto2L2Nu           | 98.04                  | NNLO + NNLL              | –             | Top++v2.0 program (ATLAS-CMS recommended twiki) | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM |
| st_tw_top            | 35.99                 | –                        | –             | [XSDB: TWminusto2L2Nu](https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=process_name%20%3D%20TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8) | /TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM |
| st_tw_antitop        | 36.05                 | –                        | –             | [XSDB: TbarWplusto2L2Nu](https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=process_name%20%3D%20TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8) | /TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM |


# 2025
NanoAOD version: v15 (data: PromptReco)
## Data

| Key | Dataset Paths | Luminosity (fb⁻¹) | Notes / Source |
| --- | --- | --- | --- |
| data_B | `/Muon0/Run2025B-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025B-PromptReco-v1/NANOAOD` | 0.26 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_C | `/Muon0/Run2025C-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025C-PromptReco-v1/NANOAOD`<br>`/Muon0/Run2025C-PromptReco-v2/NANOAOD`<br>`/Muon1/Run2025C-PromptReco-v2/NANOAOD` | 21.63 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_D | `/Muon0/Run2025D-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025D-PromptReco-v1/NANOAOD` | 25.52 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_E | `/Muon0/Run2025E-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025E-PromptReco-v1/NANOAOD` | 14.15 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_F | `/Muon0/Run2025F-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025F-PromptReco-v1/NANOAOD`<br>`/Muon0/Run2025F-PromptReco-v2/NANOAOD`<br>`/Muon1/Run2025F-PromptReco-v2/NANOAOD` | 26.89 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_G | `/Muon0/Run2025G-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2025G-PromptReco-v1/NANOAOD` | 22.40 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| **Total** | — | **110.84** | `configs/parameters/lumi.yaml` |

*Era luminosities sum to 110.85 fb⁻¹, 0.01 fb⁻¹ above the total used for normalization.*


## MC

Same MC as 2024 (`RunIII2024Summer24NanoAODv15`; same samples, cross sections and k-factors). Each sample's files are split between 2024 and 2025+2026 by relative integrated luminosity (`scripts/split_prestage_by_year_lumi.py`, prestage tag `LumiSplit`), so every MC event is used once.


# 2026
NanoAOD version: v15 (data: PromptReco)
## Data

| Key | Dataset Paths | Luminosity (fb⁻¹) | Notes / Source |
| --- | --- | --- | --- |
| data_A | `/Muon0/Run2026A-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2026A-PromptReco-v1/NANOAOD`<br>`/Muon2/Run2026A-PromptReco-v1/NANOAOD`<br>`/Muon3/Run2026A-PromptReco-v1/NANOAOD` | 0.64 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_B | `/Muon0/Run2026B-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2026B-PromptReco-v1/NANOAOD`<br>`/Muon2/Run2026B-PromptReco-v1/NANOAOD`<br>`/Muon3/Run2026B-PromptReco-v1/NANOAOD` | 15.28 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| data_C | `/Muon0/Run2026C-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2026C-PromptReco-v1/NANOAOD`<br>`/Muon2/Run2026C-PromptReco-v1/NANOAOD`<br>`/Muon3/Run2026C-PromptReco-v1/NANOAOD` | 2.11 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) — **not used** (`skip_sample: True`, low-pileup run) |
| data_D | `/Muon0/Run2026D-PromptReco-v1/NANOAOD`<br>`/Muon1/Run2026D-PromptReco-v1/NANOAOD`<br>`/Muon2/Run2026D-PromptReco-v1/NANOAOD`<br>`/Muon3/Run2026D-PromptReco-v1/NANOAOD` | 10.03 | [Source](https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0) |
| **Total** | — | **25.95** | `configs/parameters/lumi.yaml` |

*2026 luminosity is preliminary.*


## MC

Same MC as 2024 (`RunIII2024Summer24NanoAODv15`; same samples, cross sections and k-factors). Each sample's files are split between 2024 and 2025+2026 by relative integrated luminosity (`scripts/split_prestage_by_year_lumi.py`, prestage tag `LumiSplit`), so every MC event is used once.
