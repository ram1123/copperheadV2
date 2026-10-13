#!/usr/bin/env python3
"""
Convert stage-1 sync-mode parquet (switch do_sync_ntuple, configs/switches/switches_official_sync.yaml)
into flat ROOT TTrees ("Events") for the HMuMu Run-3 synchronisation exercise.

One output directory per dataset (data_H split into Muon0/Muon1), named <GROUP>-<DATASETNAME> as requested;
files are split every --max-events events. Jet_* jagged columns become a ROOT jagged collection with
counter nJet. Run metadata (sample, DAS datasets, git commit, switches, MET/b-tag definitions) is
stored in the file as TObjString "sync_info" (JSON). See docs/sync_2024_ntuples.md for the content.

How to run (repo root, default pixi env; input is a finished stage-1 run of the sync switches):
    ./run_in_pixi.sh default python scripts/make_sync_ntuple.py --stage1 <save_path>/stage1_output/2024 \
        --out <outdir> [--units <unit> ...] [--max-events 2000000]

Example:
    ./run_in_pixi.sh default python scripts/make_sync_ntuple.py \
        --stage1 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_sync2024/stage1_output/2024 \
        --out /depot/cms/hmm/shar1172/hmm_sync/run3_sync_2024/Purdue
"""
import argparse
import glob
import json
import os

import awkward as ak
import numpy as np
import uproot

from modules.git_utils import get_git_state

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


def sync_info(unit, outdir, stage1_dir):
    sample, das, sample_id = UNITS[unit]
    git = get_git_state(os.path.join(os.path.dirname(outdir), "_provenance"))  # diff patch kept out of the upload dirs
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
        "preselection": "2 muons raw (beam-spot constrained) pT>20, |eta|<2.4, 50<m_mumu(raw)<200; golden JSON on data",
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
    ap.add_argument("--max-events", type=int, default=2_000_000, help="events per output ROOT file")
    args = ap.parse_args()
    prestage = json.load(open(args.prestage))

    for unit in args.units:
        sample, _, sample_id = UNITS[unit]
        files = unit_files(unit, args.stage1, prestage)
        if not files:
            print(f"[{unit}] no parquet under {args.stage1}/f1_0/{sample}; skipping")
            continue
        outdir = os.path.join(args.out, f"{GROUP}-{unit}")
        os.makedirs(outdir, exist_ok=True)
        info = sync_info(unit, outdir, os.path.abspath(args.stage1))
        buf, nbuf, ifile, ntot = [], 0, 0, 0

        def flush():
            nonlocal buf, nbuf, ifile
            if not buf:
                return
            arr = ak.concatenate(buf)
            arr = ak.with_field(arr, np.full(len(arr), sample_id, dtype=np.int32), "sample_id")
            path = os.path.join(outdir, f"{GROUP}-{unit}_{ifile}.root")
            with uproot.recreate(path) as fout:
                fout.mktree("Events", to_tree_dict(arr),  # NanoAOD-style nJet + Jet_<var>
                            counter_name=lambda counted: "n" + counted,
                            field_name=lambda outer, inner: f"{outer}_{inner}")
                fout["sync_info"] = info
            print(f"[{unit}] wrote {path} ({len(arr)} events)")
            buf, nbuf, ifile = [], 0, ifile + 1

        for f in files:
            a = ak.from_parquet(f)
            if len(a) == 0:
                continue
            buf.append(a)
            nbuf += len(a)
            ntot += len(a)
            if nbuf >= args.max_events:
                flush()
        flush()
        print(f"[{unit}] {ntot} events from {len(files)} parquet files -> {outdir}")


if __name__ == "__main__":
    main()
