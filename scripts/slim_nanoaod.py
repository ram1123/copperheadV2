#!/usr/bin/env python3
"""
Reduce a NanoAOD ROOT file for CI / testing.

What it does:
- keeps only the first N entries from the Events tree
- copies other objects/trees as well
- writes output with strong compression for compact size

Default compression:
- LZMA level 9  -> usually very small output, slower to write/read

Example:
    python slim_nanoaod.py input.root output.root --nevts 1000
    python slim_nanoaod.py input.root nano_test_1000.root --nevts 1000 --compression LZMA --level 9
"""

import argparse
import os
import sys
import ROOT


def copy_tree_with_limit(infile, outdir, tree_name, n_keep=None):
    """
    Copy a TTree. If tree_name == 'Events', keep only first n_keep entries.
    Otherwise copy full tree.
    """
    obj = infile.Get(tree_name)
    if not obj:
        print(f"[warning] Tree '{tree_name}' not found, skipping.")
        return

    if not obj.InheritsFrom("TTree"):
        print(f"[warning] Object '{tree_name}' is not a TTree, skipping in tree copier.")
        return

    outdir.cd()

    if tree_name == "Events" and n_keep is not None:
        n_entries = obj.GetEntries()
        n_copy = min(n_keep, n_entries)
        print(f"[info] Copying {n_copy}/{n_entries} entries from '{tree_name}'")
        cloned = obj.CloneTree(0)

        for i in range(n_copy):
            if i % 100000 == 0 and i > 0:
                print(f"[info]   processed {i} entries")
            obj.GetEntry(i)
            cloned.Fill()

        cloned.Write()
    else:
        n_entries = obj.GetEntries()
        print(f"[info] Copying full tree '{tree_name}' with {n_entries} entries")
        cloned = obj.CloneTree(-1, "fast")
        cloned.Write()


def copy_other_objects(infile, outfile, skip_names=None):
    """
    Copy all non-TTree objects from the input ROOT file.
    """
    if skip_names is None:
        skip_names = set()

    keys = infile.GetListOfKeys()
    for key in keys:
        name = key.GetName()
        if name in skip_names:
            continue

        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            continue

        outfile.cd()
        print(f"[info] Copying object '{name}' ({obj.ClassName()})")
        obj.Write()


def main():
    parser = argparse.ArgumentParser(description="Slim NanoAOD ROOT file")
    parser.add_argument("input", help="Input NanoAOD ROOT file")
    parser.add_argument("output", help="Output slim ROOT file")
    parser.add_argument("--nevts", type=int, default=1000, help="Number of Events entries to keep")
    parser.add_argument(
        "--compression",
        choices=["LZMA", "ZLIB", "ZSTD"],
        default="LZMA",
        help="ROOT compression algorithm",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=9,
        help="Compression level (default: 9)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] Input file does not exist: {args.input}")
        sys.exit(1)

    ROOT.gROOT.SetBatch(True)

    infile = ROOT.TFile.Open(args.input, "READ")
    if not infile or infile.IsZombie():
        print(f"[error] Failed to open input file: {args.input}")
        sys.exit(1)

    # Set compression algorithm
    algo_map = {
        "LZMA": ROOT.ROOT.kLZMA,
        "ZLIB": ROOT.ROOT.kZLIB,
        "ZSTD": ROOT.ROOT.kZSTD,
    }
    comp_algo = algo_map[args.compression]

    outfile = ROOT.TFile(args.output, "RECREATE", "", args.level)
    outfile.SetCompressionAlgorithm(comp_algo)
    outfile.SetCompressionLevel(args.level)

    print("[info] Input :", args.input)
    print("[info] Output:", args.output)
    print(f"[info] Compression: {args.compression} level {args.level}")
    print(f"[info] Keeping first {args.nevts} events from 'Events' tree")

    # Copy main trees commonly present in NanoAOD
    tree_names = ["Events", "Runs", "LuminosityBlocks", "MetaData", "ParameterSets"]
    for tree_name in tree_names:
        copy_tree_with_limit(
            infile,
            outfile,
            tree_name,
            n_keep=args.nevts if tree_name == "Events" else None,
        )

    # Copy everything else except the trees already handled
    copy_other_objects(infile, outfile, skip_names=set(tree_names))

    outfile.Write()
    outfile.Close()
    infile.Close()

    in_size = os.path.getsize(args.input) / (1024 * 1024)
    out_size = os.path.getsize(args.output) / (1024 * 1024)

    print(f"[done] Input size : {in_size:.2f} MB")
    print(f"[done] Output size: {out_size:.2f} MB")
    if in_size > 0:
        print(f"[done] Reduction  : x{in_size / out_size:.2f}")


if __name__ == "__main__":
    main()