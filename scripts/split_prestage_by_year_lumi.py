"""
Split the Summer24 MC shared by 2024/2025/2026 (identical RunIII2024Summer24NanoAODv15
files referenced under all three year blocks in configs/datasets/dataset_nanoAODv15_run3.yaml)
into disjoint per-year buckets at file level, proportional to each year's integrated
luminosity (PC guideline: 2025+2026 are one year sharing Summer24 MC with 2024; each MC
event must be used once, not re-weighted into every year).

Inputs:
  * --mc-source: a full-fraction prestage JSON (run_prestage.py output) holding the complete
    MC file list with per-file num_entries/steps. Only its MC samples are used.
  * --data-source (template over {year}): each year's own prestage JSON; only its DATA
    samples are copied through unchanged.

Output: one JSON per year (--output, default processor_samples_<year>_NanoAODv15_LumiSplit.json)
= that year's data + its MC bucket. Stage-1 reads it with `run_stage1.py --prestage-tag LumiSplit`
(PRESTAGE_TAG=LumiSplit / snakemake prestage_tag). Outputs never overwrite an input.

Per-file sumGenWgts/nGenEvts/sumLHEPdfWgts are re-read from each file's Runs tree (reusing
run_prestage.py's helpers) and summed per bucket, so each bucket is normalized from its own
files. metadata["split_provenance"] marks the sample so stage-1 saves it under f1_0.

How to run (repo root; needs a valid VOMS proxy -- it reads every MC file's Runs tree over XRootD --
and the full per-year prestage JSONs from run_prestage.py):
    ./run_in_pixi.sh default python scripts/split_prestage_by_year_lumi.py [--dry-run] [--years ...]

Example:
    # preview the per-year file split (no file reads, nothing written)
    ./run_in_pixi.sh default python scripts/split_prestage_by_year_lumi.py --dry-run
    # write prestage_output/processor_samples_{2024,2025,2026}_NanoAODv15_LumiSplit.json
    ./run_in_pixi.sh default python scripts/split_prestage_by_year_lumi.py --n-workers 20
    # then run stage-1 on them
    snakemake -s workflow/Snakefile all_stage1 -j 2 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60 --config use_gateway=True prestage_tag=LumiSplit run_tag=<new_run_tag>
"""
import argparse
import copy
import json
import multiprocessing
import os
import time

import yaml

from run_prestage import (
    _accumulate_pdf_sumw,
    _minnlo_genweight_metadata_for_file,
    _runs_tree_metadata_for_file,
)
from modules.git_utils import get_git_state
from modules.utils import logger

UPROOT_OPTIONS = {"timeout": 4 * 2400}


def load_lumis(lumi_yaml, years):
    with open(lumi_yaml) as f:
        lumis = yaml.safe_load(f)["integrated_lumis"]
    return {y: float(lumis[y]) for y in years}


def per_file_metadata(fnames, sample_name, n_workers):
    fn = _minnlo_genweight_metadata_for_file if "MiNNLO" in sample_name else _runs_tree_metadata_for_file
    worker_args = [(fname, UPROOT_OPTIONS) for fname in fnames]
    results = {}
    with multiprocessing.Pool(processes=min(len(fnames), n_workers)) as pool:
        for fname, (sum_gen_wgts, n_gen_evts, pdf_wgts) in zip(
            fnames, pool.imap(fn, worker_args)
        ):
            results[fname] = (sum_gen_wgts, n_gen_evts, pdf_wgts)
    return results


def partition_files(files_dict, fracs):
    """Deterministic, contiguous, event-count-weighted split of one sample's files into len(fracs) buckets.

    A file goes to the bucket its cumulative-event midpoint falls in; cut points are then nudged so
    every bucket gets >= 1 file whenever there are at least as many files as buckets.
    """
    items = sorted(files_dict.items(), key=lambda kv: kv[0])
    n_files, n_buckets = len(items), len(fracs)
    total_events = sum(v["num_entries"] for _, v in items)

    cum_targets, acc = [], 0.0
    for f in fracs:
        acc += f
        cum_targets.append(acc * total_events)

    assign, running = [], 0
    for _, fdict in items:
        mid = running + fdict["num_entries"] / 2.0
        assign.append(next((i for i, t in enumerate(cum_targets) if mid < t), n_buckets - 1))
        running += fdict["num_entries"]

    # starts[b] = index of the first file of bucket b (assignments are monotonic)
    starts = [next((i for i, a in enumerate(assign) if a >= b), n_files) for b in range(n_buckets)]
    if n_files >= n_buckets:
        forced = False
        for b in range(1, n_buckets):
            lo, hi = starts[b - 1] + 1, n_files - (n_buckets - b)
            new = min(max(starts[b], lo), hi)
            forced |= new != starts[b]
            starts[b] = new
        if forced:
            logger.warning("[split] lopsided file-size distribution; forced >= 1 file per bucket")
    else:
        logger.warning(f"[split] only {n_files} file(s) for {n_buckets} buckets; some buckets stay empty")

    starts.append(n_files)
    return [dict(items[starts[b]:starts[b + 1]]) for b in range(n_buckets)]


def build_bucket_metadata(orig_metadata, bucket_files, per_file, bucket_label, split_provenance):
    n_gen_evts = sum(per_file[f][1] for f in bucket_files)
    sum_gen_wgts = sum(per_file[f][0] for f in bucket_files)
    pdf_sumw = None
    for f in bucket_files:
        pdf_sumw = _accumulate_pdf_sumw(pdf_sumw, per_file[f][2])

    meta = copy.deepcopy(orig_metadata)
    meta["nGenEvts"] = n_gen_evts
    meta["sumGenWgts"] = sum_gen_wgts
    meta["sumLHEPdfWgts"] = pdf_sumw
    meta["n_files_used"] = len(bucket_files)
    meta["fraction"] = (
        n_gen_evts / orig_metadata["nGenEvts"] if orig_metadata["nGenEvts"] else 0.0
    )
    meta["split_provenance"] = {**split_provenance, "bucket": bucket_label}
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", default=["2024", "2025", "2026"])
    parser.add_argument("--mc-source", default="prestage_output/processor_samples_2024_NanoAODv15.json")
    parser.add_argument("--data-source", default="prestage_output/processor_samples_{year}_NanoAODv15.json",
                        help="Per-year prestage JSON template ({year}); only data samples are taken from it")
    parser.add_argument("--output", default="prestage_output/processor_samples_{year}_NanoAODv15_LumiSplit.json",
                        help="Per-year output template ({year})")
    parser.add_argument("--lumi-yaml", default="configs/parameters/lumi.yaml")
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--dry-run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    years = args.years

    data_sources = {y: args.data_source.format(year=y) for y in years}
    outputs = {y: args.output.format(year=y) for y in years}
    inputs = {os.path.realpath(p) for p in [args.mc_source, *data_sources.values()]}
    clash = [p for p in outputs.values() if os.path.realpath(p) in inputs]
    if clash:
        raise SystemExit(f"[split] refusing to overwrite input file(s): {clash}")

    lumis = load_lumis(args.lumi_yaml, years)
    total_lumi = sum(lumis.values())
    fracs = {y: lumis[y] / total_lumi for y in years}
    logger.info(f"[split] lumis: {lumis}")
    logger.info("[split] fractions: " + ", ".join(f"{y}={f:.4f}" for y, f in fracs.items()))

    with open(args.mc_source) as f:
        mc_samples = {k: v for k, v in json.load(f).items() if v["metadata"]["is_mc"]}

    outs = {y: {} for y in years}
    for y in years:
        with open(data_sources[y]) as f:
            year_samples = json.load(f)
        for name, sample in year_samples.items():
            if not sample["metadata"]["is_mc"]:
                outs[y][name] = sample
            elif name in mc_samples and set(sample["files"]) != set(mc_samples[name]["files"]):
                logger.warning(f"[split] {y}/{name}: MC file list differs from --mc-source; using --mc-source's")
            elif name not in mc_samples:
                logger.warning(f"[split] {y}/{name}: MC sample not in --mc-source; dropped from the output")
        logger.info(f"[split] {y}: {len(outs[y])} data samples from {data_sources[y]}")

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    git_state = get_git_state(f"prestage_output/_provenance/split_{'_'.join(years)}_{run_timestamp}")
    split_provenance = {
        "source_file": args.mc_source,
        "lumis": lumis,
        "fractions": fracs,
        "timestamp": run_timestamp,
        "commit": git_state["commit"],
        "dirty": git_state["dirty"],
    }

    for sample_name, sample in mc_samples.items():
        files_dict = sample["files"]
        buckets = partition_files(files_dict, [fracs[y] for y in years])
        logger.info(
            f"[split] {sample_name}: {len(files_dict)} files -> "
            + " / ".join(f"{len(b)} ({y})" for y, b in zip(years, buckets))
        )
        if args.dry_run:
            continue

        per_file = per_file_metadata(list(files_dict.keys()), sample_name, args.n_workers)
        for y, bucket_files in zip(years, buckets):
            if not bucket_files:
                continue
            meta = build_bucket_metadata(sample["metadata"], bucket_files, per_file, y, split_provenance)
            outs[y][sample_name] = {"files": bucket_files, "metadata": meta}

    if args.dry_run:
        logger.info("[split] dry run, nothing written")
        return

    for y in years:
        os.makedirs(os.path.dirname(outputs[y]) or ".", exist_ok=True)
        with open(outputs[y], "w") as f:
            json.dump(outs[y], f, indent=2, sort_keys=True)
        logger.info(f"[split] wrote {outputs[y]} ({len(outs[y])} samples)")


if __name__ == "__main__":
    main()
