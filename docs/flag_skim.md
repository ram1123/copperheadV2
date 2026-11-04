# Flag-based stage-1 skim (`do_flag_skim`)

One stage-1 ntuple replaces the per-scenario ones (official, official + lepton/IP veto, ...). The
preselection is the loosest variant of each choice. The tighter choices are stored as flags, and the
muon pair of every muon-ID/IP working point is stored as positions in a stored muon list.
Code: `src/flag_skim.py` and `src/copperhead_processor.py` (`do_flag_skim` blocks). Default is off,
which leaves the production selection unchanged.

## Preselection (hard cuts)
Unchanged: lumi mask, HLT, event-quality flags, PV, jet veto maps, HEM veto, muon pT/eta/global-or-tracker/iso.
Loosened: an event is kept if **any** WP pair (below) has >= 2 muons, opposite sign, and passes the
trigger match at the loosest dR (0.4). There is no electron veto, no muon multiplicity cut and no muon IP cut.
Requires `do_trigger_match: true`, `do_lepton_ip_veto: false`, `do_muon_ip_cut: false`.

## Rows: one per distinct muon pair
All existing `mu1_*`, `mu2_*`, `dimuon_*`, jet (cleaning), DNN-input and weight columns are built from the
row's muon pair. Row `pair_row == 0` uses the **primary pair**: the configured-WP (`muon_id`, e.g. mediumId)
no-IP pair when it exists, else the loose no-IP pair (`primary_is_official_pair` tells which). An event
whose WPs pick other leading pairs (e.g. 3 Medium muons, one of the leading two failing IP) gets one more
row per distinct pair (`pair_row` 1, 2, ...; `n_pair_rows` per event), built from that pair. Every
`pass_sel_*` and `pair_<wp>_isPrimary` is true only on the row built from its pair, so each selection
takes at most one row per event. Count events with `pair_row == 0`; the cutflow counts events.
The event weight keeps the configured-WP muon SFs.

## Columns
| Column | Meaning |
|---|---|
| `mulist{0..3}_{pt,ptRoch,eta,phi,charge,iso,dxy,dz,looseId,mediumId,tightId,passIP,trigMinDR,nanoIdx}` | leading 4 base muons (loose ID, no IP, iso), pt-sorted after FSR; floats stored as float32; padded with -999 / 0 / False / -1 |
| `pair_row`, `n_pair_rows` | row number within its event (0 = primary pair) and the event's number of rows |
| `nMuons_base` | number of base muons |
| `nMuons_<wp>` | muons passing `<wp>` ∈ {looseID,mediumID,tightID}_{noIP,IP} |
| `pair_<wp>_pos1/pos2` | positions of that WP's leading pair in `mulist` (-1 = none) |
| `pair_<wp>_OS`, `pair_<wp>_mass` | opposite sign; FSR-corrected pair mass (-1 = none) |
| `pair_<wp>_trigMatch_dR{0p4,0p3,0p2,0p15,0p1}` | production trigger-match rule (either leading muon, pt_roch >= leading cut) at that dR. Combine it with `nMuons_<wp> >= 2` |
| `pair_<wp>_isPrimary` | that WP's pair is this row's pair, so the mu1/mu2/dimuon/jet columns apply to it |
| `nLooseVetoMuons` | loose-veto muons (`loose_muon_*` cuts, with IP), as in `do_lepton_ip_veto` |
| `nElectrons_official`, `nElectrons_lepIPVeto` | electrons of the official veto / of the lepton-IP veto (iso + IP, lower pT) |
| `pass_officialEleVeto` | `nElectrons_official == 0` |
| `pass_looseLepVeto` | `nLooseVetoMuons == 2 && nElectrons_lepIPVeto == 0` (the `do_lepton_ip_veto` veto) |
| `pass_invLooseLepVeto` | inverted: `nLooseVetoMuons > 2 \|\| nElectrons_lepIPVeto > 0` (ttH/VH-style) |
| `extraMu_*`, `extraEle_*` | leading loose-veto muon not in the primary pair / leading lepton-IP-veto electron (pt, eta, phi, charge) |
| `muIDIso_sf_{Loose,Medium,Tight}ID` | MC only: nominal ID x Iso SF of the primary pair for that WP (no Loose-ID trigger SF exists) |

## Stage-2 (one pass per selection)
`run_stage2_vbf.py --flag_skim_selection <name>` (env `FLAG_SKIM_SELECTION`; Snakemake config `flag_skim_selection`)
filters events with `modules.selection.FLAG_SKIM_SELECTIONS[<name>]` before any region cut or systematic. For
Tight/Loose-ID selections it rescales every `wgt_*` by `muIDIso_sf_<WP>ID / muIDIso_sf_MediumID`, while the ID/Iso
nuisances keep the Medium-ID uncertainties. A flag-skim input without a selection is refused, and the
`save_postfix` must contain the selection name. Selections: `official`, `lepIPVeto`, `nMu2NoLepVeto`, `tightID`,
`looseID`, `ipOnly` (Medium + IP, official electron veto), `trigDR0p1`, `tightIDlepIPVeto`, `tightIDIP` / `looseIDIP`
(Tight/Loose + IP, official electron veto), `looseIDlepIPVeto` (Loose + IP + lepton veto). 2024 chain:
`logs/plans/snakemake_flagskim_2024_*.yaml` + `logs/plans/run_flagskim_2024.sh`.

## Scenario recipes
| Scenario | Flag |
|---|---|
| official (`do_lepton_ip_veto: false`) | `pass_sel_official` = `nMuons_mediumID_noIP == 2 && pair_mediumID_noIP_OS && trigMatch(prod dR) && pass_officialEleVeto && pair_mediumID_noIP_isPrimary` |
| official + lepton/IP veto | `pass_sel_lepIPVeto` = `nMuons_mediumID_IP >= 2 && pair_mediumID_IP_OS && trigMatch && pass_looseLepVeto && pair_mediumID_IP_isPrimary` |
| no lepton veto, exactly 2 muons | `pass_sel_nMu2NoLepVeto` (official without the electron veto) |
| other WP / dR / inverted veto | combine `nMuons_<wp>`, `pair_<wp>_OS`, `pair_<wp>_trigMatch_dR*`, the veto flags and `pair_<wp>_isPrimary` (picks the row built from that WP's pair) |

## Checks: commands (2024 v2)
Run from the worktree root (`/work/users/shar1172/copperhead_test`). All are read-only on the ntuple and take
seconds to minutes. `IN` is the v2 label:
```bash
IN=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit/stage1_output
```
Lepton rule: every signal selection has an electron veto; the loose-muon veto is optional and only with IP.
`nMu2NoLepVeto` (no electron veto) is a diagnostic only.

1. **Chain status** (stage-1 + 8 stage-2/3/stats cases): `grep -E "START|OK|FAIL" logs/detached/fsv2_2024.log`;
   per-job results `cat logs/FlagSkimV2_Oct30_2024_*/failed_jobs.log | grep -v SUCCESS` (empty = all OK).
2. **Inclusive yields vs Oct26/Oct23** (stage-1 level, all cases):
   `/work/pixi/global/.pixi/envs/default/bin/python logs/plans/flagskim_yields_2024.py Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit validation/flag_skim/2024_v2`
   (both args needed: without them it reads the old v1 label; loops over every `FLAG_SKIM_SELECTIONS` entry, ~2 min)
   -> `validation/flag_skim/2024_v2/yields_per_group.csv` (`n`/`w` = raw/weighted, `_win` = 110-150 GeV).
3. **VBF h-sideband yields per selection** (`scripts/get_yields.py --flag_skim_selection`, ~5 min each):
   `bash logs/plans/run_get_yields_fsv2_2024.sh [sel ...]` (no args = all 11) ->
   `validation/flag_skim/2024_v2/get_yields/yield_summary_FS<sel>_2024.csv`. Single selection:
   ```bash
   ./run_in_pixi.sh default python scripts/get_yields.py --input $IN --years 2024 --flag_skim_selection looseIDIP \
       --output-csv validation/flag_skim/2024_v2/get_yields/yield_FSlooseIDIP_2024.csv \
       --summary-output-csv validation/flag_skim/2024_v2/get_yields/yield_summary_FSlooseIDIP_2024.csv
   ```
   Region/category are hardcoded in the script (`vbf`, `h-sidebands`).
4. **Cutflow + trigger-dR scan** (`scripts/flag_skim_cutflow.py`, ~25 s; official e-veto, exactly 2 muons, cutbased
   VBF steps; `--wp` = `{looseID,mediumID,tightID}_{IP,noIP}`; `--region` = `z-peak`, `h-sidebands`, `signal`, `h-peak`):
   ```bash
   for r in z-peak h-sidebands signal; do
     ./run_in_pixi.sh default python scripts/flag_skim_cutflow.py --input $IN -y 2024 --wp looseID_IP --region $r \
         --out validation/flag_skim/2024_v2/cutflow_looseIDIP_$r
   done
   ```
   -> `cutflow_wide.csv` (per group, per step, per dR) / `cutflow_long.csv`. Blinded: in `signal`/`h-peak` data is
   not counted from the mass step on; the last row (VBF in h-peak) is always MC only.
5. **Significance / limits per case** (after step 1):
   ```bash
   B=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit
   for d in $B/stage3_datacards_Oct30_2026_Syst_DNNRay050_FSv2_*; do
     echo "${d##*FSv2_} $(tail -1 $d/score_*/vbf_significance_summary_*.csv | cut -d, -f3,4) $(tail -1 $d/score_*/vbf_expected_limit_summary_*.csv | cut -d, -f3,4)"
   done   # case_region  sig,sig_statonly  limit,limit_statonly
   ```
6. **Control plots** (data/MC + selection comparison, dimuon/muon/jet/dijet + all ggH-BDT / VBF-DNN inputs;
   9 selections x 5 phase spaces x 40 variables; fill ~5 min, plotting ~1 min):
   ```bash
   ./run_in_pixi.sh default python scripts/flag_skim_control_plots.py --input $IN -y 2024 \
       --out validation/flag_skim/2024_v2/control_plots            # add --plot-only to redraw from hists.pkl
   ```
   -> `control_plots/compare/<ps>/<var>.{pdf,png}`, `control_plots/stack/{official,looseIDIP}/<ps>/<var>.{pdf,png}`.
   Phase spaces `zpeak_incl`, `zpeak_ggh`, `zpeak_vbf`, `hsb_ggh`, `hsb_vbf` (peak blinded). MC keeps the Z-pT weight.
7. **Report deck** `reports/FlagSkim_2024_v2/FlagSkim_2024_v2.pdf` (tables generated from the CSVs above, plots copied in):
   ```bash
   /work/pixi/global/.pixi/envs/default/bin/python reports/FlagSkim_2024_v2/make_report_inputs.py
   cd reports/FlagSkim_2024_v2 && export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/latest/bin/x86_64-linux:$PATH && \
     latexmk -pdf -interaction=nonstopmode FlagSkim_2024_v2.tex && latexmk -c
   ```
8. **H-peak VBF yields, MC only** (data never read in h-peak: blinded), all cases, about 4 min per case on a local Dask:
   ```bash
   IN=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit/stage1_output
   OUT=validation/flag_skim/2024_v2/get_yields_hpeak; mkdir -p $OUT
   for sel in official ipOnly lepIPVeto trigDR0p1 looseID tightID tightIDlepIPVeto nMu2NoLepVeto; do
     ./run_in_pixi.sh default python scripts/get_yields.py --input $IN --years 2024 --flag_skim_selection $sel --regions h-peak \
       --output-csv $OUT/yield_FS${sel}_2024.csv --summary-output-csv $OUT/yield_summary_FS${sel}_2024.csv
   done
   ```
9. **No-Z-pT Data/MC plots, baseline** (Dask Gateway cluster with PYTHONPATH = this worktree):
   `./run_in_pixi.sh default snakemake -s workflow/Snakefile plot_all -j 1 --rerun-incomplete --restart-times 3 --latency-wait 60 --configfile logs/plans/snakemake_fsv2_2024_official_nozpt.yaml`
10. **Baseline: n_b / jj pT plots (opt-in `jet_multiplicity` group) + Z-pT test** (VBF and nocat >=2 jets, with/without Z-pT):
   `./run_in_pixi.sh default snakemake -s workflow/Snakefile plot_all -j 2 --rerun-incomplete --restart-times 3 --latency-wait 60 --configfile logs/plans/snakemake_fsv2_2024_official_jetmult_zpttest.yaml`
   Compare: `grep -E "^(Data|DY|Total bkg MC):|Integrated|chi2" <figs>/Reg_z-peak/Cat_{nocat,vbf}/njet_2/{no,default}_zpt_weights/dimuon_pt.txt`
11. **Z-pT re-derivation, 2024, baseline, (Data - non-DY)/DY** (outputs under `validation/zpt_rewgt/<label>/IncDY_aMCatNLO_FSv2official_BkgSub/`;
    production `data/zpt_rewgt/*.yaml` untouched). Step 0 on the Gateway (~10 min), steps 1-2 local (~2 min):
   ```bash
   L=Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit; T=IncDY_aMCatNLO_FSv2official_BkgSub
   ./run_in_pixi.sh default python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py -l $L -y 2024 \
     --input_path /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/$L -dy_sample $T \
     --flag_skim_selection official --subtract_nondy --use_gateway
   ./run_in_pixi.sh default python src/copperhead/zpt_rewgt/derive/do_f_test.py -l $L -y 2024 --dy_sample $T --njet 0 1 2 --save_postfix Nov03_2026
   ./run_in_pixi.sh default python src/copperhead/zpt_rewgt/derive/get_polyFit.py -l $L -y 2024 --dy_sample $T --njet 0 1 2 \
     --save_postfix Nov03_2026 --input_path /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/$L
   ```
   Bad fits (njet2 [15,170]: f1 order 8 ran away; njet1 [21,120]: kink at 120) were fixed in
   `src/copperhead/zpt_rewgt/derive/bin_definitions.py` (2024: njet0 [15,110], njet1 [21,110], njet2 [25,100], 5 GeV bins 80-120 for
   njet1/2), then steps 1-2 rerun with the same postfix -> `gof_Nov03_2026/`, `zpt_rewgt_params_${T}_TunedBins.yaml`.
