"""
get_neg_weight_fraction.py
Compute negative weight fraction and effective luminosity
from the Runs tree across all NanoAOD files for one year.

Uses the standard CMS formula:
    N_eff = sumw² / sumw2

This is correct for amcatnloFXFX where weights are NOT unit (±1)
but large numbers (~100-1000 per event).

Usage:
    python get_neg_weight_fraction.py /path/to/files/*.root --xsec 3.05
    python get_neg_weight_fraction.py --filelist files.txt --xsec 3.05 --lumi 8.1
"""
import argparse
import glob
import math
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning  # suppress edm dictionary warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="ROOT files or glob patterns")
    parser.add_argument("--filelist", help="Text file with one path per line")
    parser.add_argument("--xsec", type=float, default=None,
                        help="Cross section in pb (from XSDB)")
    parser.add_argument("--lumi", type=float, default=None,
                        help="Data luminosity in fb⁻¹ (to compute MC/data ratio)")
    args = parser.parse_args()

    # Collect input files
    input_files = []
    if args.filelist:
        with open(args.filelist) as f:
            input_files = [line.strip() for line in f if line.strip()]
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            # treat as literal path (e.g. xrootd URL)
            input_files.append(pattern)

    if not input_files:
        raise RuntimeError("No input files found.")

    print(f"Processing {len(input_files)} file(s)...")

    chain = ROOT.TChain("Runs")
    for f in input_files:
        chain.Add(f)

    total_sumw  = 0.0
    total_sumw2 = 0.0
    total_count = 0

    for entry in chain:
        total_sumw  += entry.genEventSumw
        total_sumw2 += entry.genEventSumw2
        total_count += entry.genEventCount

    # --- Standard CMS formula (works for any weight magnitude) ---
    # N_eff = sumw² / sumw2
    # This equals N_total * (1 - 2*f_neg)² for uniform |weight|
    N_eff = total_sumw ** 2 / total_sumw2

    # Back-compute f_neg for reference (assumes uniform |weight|)
    penalty_ratio = N_eff / total_count          # = (1 - 2*f_neg)²
    f_neg = (1.0 - math.sqrt(penalty_ratio)) / 2.0

    print(f"\n{'='*55}")
    print(f"Total generated events  : {total_count:>20,.0f}")
    print(f"Sum of weights  (sumw)  : {total_sumw:>20,.2f}")
    print(f"Sum of weights² (sumw2) : {total_sumw2:>20,.2f}")
    print(f"")
    print(f"N_eff = sumw²/sumw2     : {N_eff:>20,.0f}")
    print(f"N_eff / N_total         : {penalty_ratio:>20.4f}  ({penalty_ratio*100:.2f}%)")
    print(f"Neg. weight fraction    : {f_neg:>20.4f}  ({f_neg*100:.2f}%)")
    print(f"Stat penalty (1/ratio)  : {1/penalty_ratio:>20.2f}x more events needed vs LO")

    if args.xsec:
        # Effective luminosity this sample represents (in fb⁻¹)
        # L_eff = N_eff / (σ [pb] × 1000 [fb⁻¹/pb⁻¹... wait:
        # 1 pb × 1 fb⁻¹ = 1e-36 × 1e39 cm² = 1e3 → need × 1000 to get events? No.
        # 1 pb = 1e-36 cm²; 1 fb⁻¹ = 1e39 cm⁻²; product = 1e3 events... no that's wrong
        # 1 pb × 1 fb⁻¹ = (1e-3 fb) × (1 fb⁻¹) = 1e-3 events → need ×1000 if σ in pb, L in fb⁻¹
        # Actually: 1 pb × 1 fb⁻¹ = 1e-3 × 1e15 barn × 1e-15 barn⁻¹ = 1 event? Let me be careful.
        # 1 pb = 1e-12 barn; 1 fb = 1e-15 barn; 1 fb⁻¹ = 1e15 barn⁻¹ = 1e15 / (1e-12 pb) = 1e3 pb⁻¹
        # So σ [pb] × L [fb⁻¹] × 1000 = events.  Hence L [fb⁻¹] = N_eff / (σ [pb] × 1000)
        L_eff = N_eff / (args.xsec * 1e3)   # fb⁻¹

        print(f"\nWith σ = {args.xsec} pb:")
        print(f"  Effective luminosity  = {L_eff:,.1f} fb⁻¹")

        if args.lumi:
            ratio = L_eff / args.lumi
            print(f"  Data luminosity       = {args.lumi:.1f} fb⁻¹")
            print(f"  MC / data ratio       = {ratio:.1f}x")

    print('='*55)


if __name__ == "__main__":
    main()
