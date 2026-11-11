#!/usr/bin/env python3
"""
Convert stage-1 sync-mode parquet (switch do_sync_ntuple, configs/switches/switches_official_sync.yaml)
into flat ROOT TTrees ("Events") for the HMuMu Run-3 synchronisation exercise.

Round-0 2nd iteration (default, --layout files): one file <GROUP>-<DATASETNAME>.root per dataset directly in
--out (data_H split into Muon0/Muon1); written as --max-events parts (uproot cannot write files beyond ~2 GB) that
are merged with ROOT's hadd (needs hadd on PATH, e.g. the default pixi env). --layout dirs reproduces the 1st-iteration
layout: one directory <GROUP>-<DATASETNAME>/ per dataset, files split every --max-events events.
Data are blinded here, not in stage-1 (agreed for the 2nd iteration): an event is dropped if any stored pair
(default, IP, agreed) has 115 <= m_mumu(raw) <= 135 GeV; --no-blind writes unblinded data (local checks only). Jet_* jagged columns become a ROOT jagged collection with
counter nJet. Run metadata (sample, DAS datasets, git commit, switches, MET/b-tag definitions) is
stored in the file as TObjString "sync_info" (JSON). See docs/sync_2024_ntuples.md for the content.

--round round1 (HMuMu WG "Synchronization update", 2026-11-05): keeps only events passing the Round-1 baseline
(agreed muons/electron veto/trigger match/b-veto), writes a slim tree of corrected quantities with plain branch
names (ROUND1 maps below), only central MC weights plus xsec/lumi normalisation, region flags on the corrected
m_mumu, and <GROUP>_round1_corrections.txt (correction files and JERC tags) in --out. Data are blinded if any
stored pair has corrected (ScaRe+FSR) m_mumu in [115, 135] GeV. Upload area: round1_sync/<GROUP>/.

How to run (repo root, default pixi env; input is a finished stage-1 run of the sync switches):
    ./run_in_pixi.sh default python scripts/make_sync_ntuple.py --stage1 <save_path>/stage1_output/2024 \
        --out <outdir> [--units <unit> ...] [--max-events 1000000] [--layout files|dirs] [--round round0|round1]

Example:
    ./run_in_pixi.sh default python scripts/make_sync_ntuple.py \
        --stage1 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_sync2024_r0v2/stage1_output/2024 \
        --out /depot/cms/hmm/shar1172/hmm_sync/run3_sync_2024_r0v2/Purdue/update_after_Hmm_15Oct
    ./run_in_pixi.sh default python scripts/make_sync_ntuple.py --round round1 \
        --stage1 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_sync2024_r1/stage1_output/2024 \
        --out /depot/cms/hmm/shar1172/hmm_sync/round1_sync/Purdue
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import awkward as ak
import numpy as np
import uproot

from modules.git_utils import get_git_state
from src.lib.get_parameters import getParametersForYr

GROUP = "Purdue"
# output unit -> (stage-1 sample key, DAS dataset, integer id stored per event); output dir = <GROUP>-<unit>
UNITS = {
    "Muon0_Run2024H-MINIv6NANOv15-v1": ("data_H", "/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD", 0),
    "Muon1_Run2024H-MINIv6NANOv15-v2": ("data_H", "/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD", 6),
    "VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": (
        "vbf_powheg", "/VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM", 1),
    "GluGluH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": (
        "ggh_powhegPS", "/GluGluH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM", 2),
    "TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": (
        "tth_powheg", "/TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM", 3),
    "ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8": (
        "zh_powheg", "/ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM", 4),
    "WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8": (
        "wph_powheg", "/WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM", 5),
}
DATASET_YAML = "configs/datasets/sync_dataset_nanoAODv15_2024.yaml"
SWITCHES_YAML = "configs/switches/switches_official_sync.yaml"
BLIND_MASS_RAW = (115.0, 135.0)  # data only, inclusive edges
BLIND_COLUMNS = ("dimuon_mass_raw", "dimuon_mass_raw_ip", "dimuon_mass_raw_agreed")
# Round 1 blinds the signal region on the ScaRe+FSR-corrected m_mumu of every stored pair
BLIND_COLUMNS_R1 = ("dimuon_mass", "dimuon_mass_ip", "dimuon_mass_agreed")

# Round-1 output branch -> stage-1 sync column. Muons/dimuon/jets are after ScaRe+FSR / JEC+JER. The default
# pair (mu1/mu2) equals the agreed pair for every baseline event (checked per file, see round1_select).
ROUND1_EVENT = {
    "run": "run", "luminosityBlock": "luminosityBlock", "event": "event",
    "HLT_IsoMu24": "HLT_IsoMu24", "pass_trigger_match": "pass_trigger_match_agreed",
    "pass_MET_filters": "pass_MET_filters", "nmuons_good": "nmuons_agreed",
    "nelectrons_veto": "nelectrons_agreed", "pass_electron_veto": "pass_electron_veto_agreed",
    "pass_jet_veto_map_event": "pass_jet_veto_map",  # info only: Round-1 jets inside the map are dropped, not the event
    "njets": "njets_agreed", "nbjets_loose": "nBtagLoose_agreed", "nbjets_medium": "nBtagMedium_agreed",
    "pass_bveto": "pass_bveto_agreed", "pass_vbf": "pass_vbf_agreed",
    "cat_vbf": "cat_vbf_agreed", "cat_ggf": "cat_ggh_agreed",
    "dimuon_mass": "dimuon_mass_agreed", "dimuon_pt": "dimuon_pt_agreed",
    "dijet_mass": "jj_mass_agreed", "dijet_dEta": "jj_dEta_agreed", "dijet_pt": "jj_pt_agreed", "dijet_eta": "jj_eta_agreed",
    "PuppiMET_pt": "PuppiMET_pt", "PuppiMET_phi": "PuppiMET_phi",
}
ROUND1_MUON = {  # mu{i}_<key> <- mu{i}_<value>
    "pt": "pt", "eta": "eta", "phi": "phi", "charge": "charge", "index": "index",
    "mediumId": "mediumId", "tightId": "tightId", "pfRelIso04_fsr": "pfRelIso04_all_fsrCorr",
    "passId": "passIdAgreed", "passIso": "passIsoAgreed", "hasFsr": "hasFsr", "trigMatched": "trigMatched",
}
ROUND1_JET = {  # selected (agreed) jets only
    "pt": "pt", "eta": "eta", "phi": "phi", "mass": "mass", "btagUParTAK4B": "btagUParTAK4B",
    "passBtagLoose": "passBtagLooseAgreed", "passBtagMedium": "passBtagMediumAgreed", "hadronFlavour": "hadronFlavour",
}
ROUND1_WEIGHTS = {  # MC only, central values
    "weight_gen": "separate_wgt_genWeight", "weight_norm": "separate_wgt_genWeight_normalization",
    "xsec_pb": "separate_wgt_xsec", "lumi_pb": "separate_wgt_lumi", "weight_pu": "separate_wgt_pu",
    "weight_muID": "separate_wgt_muID", "weight_muIso": "separate_wgt_muIso", "weight_muTrig": "separate_wgt_muTrig",
}
# Round-1 mass regions on the corrected m_mumu (open intervals as on the slide)
ROUND1_REGIONS = {
    "in_signal_region": [(115, 135)], "in_higgs_sideband": [(110, 115), (135, 150)], "in_signal_extended": [(110, 150)],
    "in_z_sideband": [(70, 110)], "in_z_sideband_restricted": [(76, 106)], "in_mass_inclusive": [(60, 160)],
}


def blind_mask(arr, columns=BLIND_COLUMNS):
    """True for events to drop: any listed pair mass inside BLIND_MASS_RAW (115-135 GeV, inclusive)."""
    drop = np.zeros(len(arr), dtype=bool)
    for col in columns:
        if col in arr.fields:
            m = ak.to_numpy(ak.fill_none(arr[col], -999.0))
            drop |= (m >= BLIND_MASS_RAW[0]) & (m <= BLIND_MASS_RAW[1])
    return drop


def to_tree_dict(arr):
    """Flat columns as branches; Jet_* (jagged) zipped into one collection -> nJet + Jet_<var>."""
    flat, jet = {}, {}
    for f in arr.fields:
        col = arr[f]
        if f.startswith("Jet_"):
            jet[f[len("Jet_"):]] = col
            continue
        if col.ndim > 1:
            continue  # no other jagged columns expected
        col = ak.fill_none(col, -999)  # padded (missing jet/pair) values
        flat[f] = ak.values_astype(col, np.int8) if str(ak.type(col)).endswith("bool") else col
    if jet:
        flat["Jet"] = ak.zip({k: (ak.values_astype(v, np.int8) if str(ak.type(v)).endswith("bool") else v)
                              for k, v in jet.items()})
    return flat


def round1_baseline(arr):
    """Round-1 baseline: agreed selection (muons, HLT, MET filters, PV, OS, electron veto) + trigger match + b-veto."""
    return ak.to_numpy(ak.fill_none(arr["pass_agreed_selection"] & arr["pass_trigger_match_agreed"]
                                    & arr["pass_bveto_agreed"], False))


def round1_select(arr, is_mc):
    """Baseline events of one parquet shard as a Round-1 record array (Jet_* = selected jets)."""
    arr = arr[round1_baseline(arr)]
    if len(arr) == 0:
        return arr
    # mu1/mu2 branches are the default pair; it must be the agreed pair for the slim tree to be right
    same = ak.to_numpy(ak.fill_none(arr["mu_pair_agreed_is_sync_pair"], False))
    if not same.all():
        raise RuntimeError(f"{int((~same).sum())} baseline events whose default muon pair is not the agreed pair")
    out = {k: arr[v] for k, v in ROUND1_EVENT.items()}
    for i in (1, 2):
        out.update({f"mu{i}_{k}": arr[f"mu{i}_{v}"] for k, v in ROUND1_MUON.items()})
    mass = ak.to_numpy(ak.fill_none(arr["dimuon_mass_agreed"], -999.0))
    for name, windows in ROUND1_REGIONS.items():
        out[name] = np.logical_or.reduce([(mass > lo) & (mass < hi) for lo, hi in windows])
    sel = arr["Jet_isSelectedAgreed"]
    for k, v in ROUND1_JET.items():
        if f"Jet_{v}" in arr.fields:
            out[f"Jet_{k}"] = arr[f"Jet_{v}"][sel]
        elif is_mc or k != "hadronFlavour":  # only hadronFlavour is legitimately absent (data)
            raise RuntimeError(f"stage-1 column Jet_{v} missing; rerun stage-1 with the Round-1 sync code")
    if is_mc:
        w = {k: arr[v] for k, v in ROUND1_WEIGHTS.items()}
        out.update(w)
        out["weight_central"] = (w["weight_gen"] * w["weight_norm"] * w["xsec_pb"] * w["lumi_pb"]
                                 * w["weight_pu"] * w["weight_muID"] * w["weight_muIso"] * w["weight_muTrig"])
    return ak.zip(out, depth_limit=1)


def round1_corrections_txt(path):
    """Correction files, JERC tags, WPs and lumi used for 2024 (from the configs at conversion time)."""
    c = getParametersForYr("./configs/parameters/", "2024", switches_path=SWITCHES_YAML)
    jec = dict(c["jec_parameters"])
    lines = [
        "# Purdue Round-1 sync: corrections used (2024, configs at the conversion commit; stage-1 commit in sync_info)",
        f"muon ScaRe, beam-spot constrained pT (bsConstrainedChi2<30), VXBS payload: {c['roccor_file']}",
        f"muon ScaRe, NanoAOD pT fallback (bsConstrainedChi2>=30), regular payload: {c.get('roccor_file_nonBS', c['roccor_file'])}",
        "  order: pT choice -> ScaRe -> FSR recovery",
        "  MC smearing as official muonscarekit MuonScaRe.py: random number from the payload's RandomSmearing(event, lumi, phi);",
        "  smeared pT with ratio to the scale-corrected pT > 2 or < 0.1 (or < 0) is not smeared (protects the regular",
        "  2024 payload's cb_params n~1.002 forward bins, which otherwise give pT up to 1e250 / inf)",
        f"muon ID/Iso/trigger SFs: {c['muSFFileList']['json_file']}",
        f"  pT>=30 GeV: ID {c['muSFFileList']['id']}, Iso {c['muSFFileList']['iso']}, trigger {c['muSFFileList']['trig']}",
        f"  pT<30 GeV:  ID {c['muSFFileList'].get('id_lowpt')}, Iso {c['muSFFileList'].get('iso_lowpt')}, trigger {c['muSFFileList'].get('trig_lowpt')}",
        "  SFs evaluated at the ScaRe+FSR-corrected pT/eta; ID/iso WP chosen with the corrected pT",
        f"JEC: {jec.get('jerc_load_path')}  MC tag {jec.get('jec_tags')}  data tags {dict(jec.get('jec_data_tags', {}))}",
        f"JER: {jec.get('jersmear_load_path')}  tag {jec.get('jer_tags')}  (HE 2.5<|eta|<3: JER for gen-matched jets only, pT>50 GeV)",
        f"2.0<|eta|<2.5 L2L3Residual freeze (data): {'applied' if c['switches'].get('do_jec_residual_freeze') else 'NOT applied'}"
        " (jets with L1L2L3 pT < 30 GeV get L2L3Residual at 30 GeV)",
        f"jet veto map: {c['jet_veto_maps']}  key {c['jet_veto_maps_tag']}  (Round-1 jets inside the map are dropped)",
        f"pileup: {c['pu_file_mc']}",
        f"b-tag: UParTAK4 btagUParTAK4B, loose {c['btag_loose_wp_UParT']}, medium {c['btag_medium_wp_UParT']} ({c['btag_sf_json']})",
        f"MET filters: {list(c['event_flags'])}",
        f"golden JSON / lumi / cross sections: {DATASET_YAML}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def sync_info(unit, provenance_dir, stage1_dir, blind, round_="round0"):
    sample, das, sample_id = UNITS[unit]
    r1 = round_ == "round1"
    git = get_git_state(provenance_dir)  # diff patch kept out of the upload dirs
    return json.dumps({
        "group": GROUP,
        "sample": sample,
        "sample_id": sample_id,
        "das_dataset": das,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "stage1_dir": stage1_dir,  # holds the stage-1 run's own git_info_*/switches_used_* provenance
        "switches_yaml": SWITCHES_YAML,
        "dataset_yaml": DATASET_YAML,
        "met_collection": "PuppiMET (NanoAOD, no re-correction)",
        "btag": "UParTAK4 (btagUParTAK4B), loose WP 0.0246, medium WP 0.1272 (BTV Summer24)",
        # stage-1 window follows muon_sel_corrected_kinematics (on for 2024), whatever the output round
        "preselection": "2 muons pT>20, |eta|<2.4 after ScaRe+FSR, 60<m_mumu(ScaRe+FSR)<160 on the default, IP or agreed pair; golden JSON on data",
        "iteration": ("Round 1 (HMuMu WG 2026-11-05): only Round-1 baseline events, slim corrected-quantity tree; "
                      "corrections in Purdue_round1_corrections.txt") if r1 else
                     "Round 0, 2nd iteration (update_after_Hmm_15Oct); agreed selection in *_agreed branches",
        "blinding": (("data only: events removed if any stored pair has 115<=m_mumu<=135 (ScaRe+FSR-corrected)" if r1 else
                      "data only: events removed if any stored pair has 115<=m_mumu(raw)<=135") if blind else "NOT BLINDED")
                    if sample.startswith("data") else "none (MC)",
        "agreed_selection": "Round 1 (2026-11-05): muons pT>20 and |eta|<2.4 after ScaRe+FSR, mediumId + FSR-corr pfRelIso04_all<0.25 if pT>=30 else tightId + iso<0.15, exactly 2, corrected pT 26/20; electron veto pT>10 |eta|<2.4 mvaIso_WP90 (no ECAL gap); "
                            "jets pT>25 (JEC+JER) |eta|<4.7 tight ID, outside jet veto map, HE pT>50, dR(j,mu)>0.4; "
                            "b-veto N_M=0 and N_L<2 on agreed jets |eta|<2.5; VBF mjj>400 |deta|>2.5 on the leading (and max-mjj) pair, no extra jet-pT cut; "
                            "trigger match, SFs and jet-muon dR on ScaRe+FSR muons; 2.0<|eta|<2.5 L2L3Residual freeze on data",
    })


def shard_stem(parquet_path):
    """part_<input-file stem>_<entrystart>_<entrystop>.parquet -> input-file stem (runner_adapter._build_shard_id)."""
    return os.path.basename(parquet_path)[len("part_"):].rsplit("_", 2)[0]


def unit_files(unit, stage1, prestage):
    """Parquet shards of one output unit; data_H is split by primary dataset (Muon0/Muon1) via its input files."""
    sample, das, _ = UNITS[unit]
    files = sorted(glob.glob(f"{stage1}/f1_0/{sample}/*/*.parquet"))
    if sample != "data_H":
        return files
    primary = das.split("/")[1]  # Muon0 / Muon1
    stems = {os.path.splitext(os.path.basename(f))[0].replace(".", "_"): f
             for f in prestage[sample]["files"]}
    mine = {st for st, f in stems.items() if f"/{primary}/" in f}
    unknown = [f for f in files if shard_stem(f) not in stems]
    if unknown:
        raise RuntimeError(f"{len(unknown)} data_H shards not matched to an input file, e.g. {unknown[0]}")
    return [f for f in files if shard_stem(f) in mine]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="<save_path>/stage1_output/<year> of the sync stage-1 run")
    ap.add_argument("--out", required=True, help="output base directory")
    ap.add_argument("--units", nargs="+", default=list(UNITS), choices=list(UNITS), help="output units (default all)")
    ap.add_argument("--prestage", default="prestage_output/processor_samples_2024_NanoAODv15_sync2024.json",
                    help="prestage JSON of the run (to split data_H into Muon0/Muon1)")
    # ~1 kB/event: 2M events reach uproot's ~2 GB per-file limit
    ap.add_argument("--max-events", type=int, default=1_000_000, help="events per output ROOT file (dirs) / per part before hadd (files)")
    ap.add_argument("--layout", choices=["files", "dirs"], default="files",
                    help="files: one <GROUP>-<unit>.root per dataset (2nd iteration); dirs: 1st-iteration split directories")
    ap.add_argument("--no-blind", action="store_true", help="do not blind data (local checks only, never upload)")
    ap.add_argument("--round", choices=["round0", "round1"], default="round0",
                    help="round0: full flag ntuple (2nd iteration); round1: Round-1 baseline events, slim tree")
    args = ap.parse_args()
    prestage = json.load(open(args.prestage))
    r1 = args.round == "round1"
    blind_cols = BLIND_COLUMNS_R1 if r1 else BLIND_COLUMNS
    if r1:
        os.makedirs(args.out, exist_ok=True)
        round1_corrections_txt(os.path.join(args.out, f"{GROUP}_round1_corrections.txt"))

    for unit in args.units:
        sample, _, sample_id = UNITS[unit]
        files = unit_files(unit, args.stage1, prestage)
        if not files:
            print(f"[{unit}] no parquet under {args.stage1}/f1_0/{sample}; skipping")
            continue
        if args.layout == "dirs":
            outdir = os.path.join(args.out, f"{GROUP}-{unit}")
            provenance = os.path.join(args.out, "_provenance")
        else:
            outdir = args.out
            provenance = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_provenance")
        os.makedirs(outdir, exist_ok=True)
        blind = sample.startswith("data") and not args.no_blind
        if blind and not all(c in ak.from_parquet(files[0]).fields for c in blind_cols):
            raise RuntimeError(f"[{unit}] blinding columns {blind_cols} missing; stage-1 is not 2nd-iteration code")
        info = sync_info(unit, provenance, os.path.abspath(args.stage1), blind, args.round)
        buf, nbuf, ifile, ntot, nblind = [], 0, 0, 0, 0
        # files layout: uproot cannot write files beyond ~2 GB, so write parts and hadd them into one file
        partdir = os.path.join(outdir, f".parts_{GROUP}-{unit}") if args.layout == "files" else outdir
        os.makedirs(partdir, exist_ok=True)
        parts = []

        def flush():
            nonlocal buf, nbuf, ifile
            if not buf:
                return
            arr = ak.concatenate(buf)
            arr = ak.with_field(arr, np.full(len(arr), sample_id, dtype=np.int32), "sample_id")
            path = os.path.join(partdir, f"{GROUP}-{unit}_{ifile}.root")
            with uproot.recreate(path) as fout:
                fout.mktree("Events", to_tree_dict(arr),  # NanoAOD-style nJet + Jet_<var>
                            counter_name=lambda counted: "n" + counted,
                            field_name=lambda outer, inner: f"{outer}_{inner}")
                if args.layout == "dirs" or ifile == 0:  # one sync_info key after hadd
                    fout["sync_info"] = info
            parts.append(path)
            print(f"[{unit}] wrote {path} ({len(arr)} events)")
            buf, nbuf, ifile = [], 0, ifile + 1

        for f in files:
            a = ak.from_parquet(f)
            if blind and len(a):
                drop = blind_mask(a, blind_cols)
                nblind += int(drop.sum())
                a = a[~drop]
            if r1 and len(a):
                a = round1_select(a, is_mc=not sample.startswith("data"))
            if len(a) == 0:
                continue
            buf.append(a)
            nbuf += len(a)
            ntot += len(a)
            if nbuf >= args.max_events:
                flush()
        flush()
        if args.layout == "files" and parts:
            final = os.path.join(outdir, f"{GROUP}-{unit}.root")
            subprocess.run(["hadd", "-f", final, *parts], check=True, stdout=subprocess.DEVNULL)
            nmerged = uproot.open(final)["Events"].num_entries
            if nmerged != ntot:
                raise RuntimeError(f"[{unit}] hadd output has {nmerged} entries, expected {ntot}; parts kept in {partdir}")
            shutil.rmtree(partdir)  # only our own temporary parts
            print(f"[{unit}] merged {len(parts)} parts -> {final}")
        print(f"[{unit}] {ntot} events from {len(files)} parquet files -> {outdir}"
              + (f" ({nblind} blinded events removed)" if blind else ""))


if __name__ == "__main__":
    main()
