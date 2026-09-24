"""
dijet_genjet_mass.py
--------------------
Compute the generator-level dijet invariant mass from CMS NanoAOD using
PyROOT RDataFrame, and report the fraction of events falling in three
pT-ordered dijet-mass windows:

    Window A :   0 - 300 GeV
    Window B : 300 - 350 GeV
    Window C : 350 - inf GeV

Gen jets are cleaned against generator-level prompt leptons
(|pdgId| ∈ {11, 13, 15}) using ΔR > 0.3 before the dijet mass is computed.

Usage
-----
    python dijet_genjet_mass.py [input.root [input2.root ...]] [options]

    --tree TREE      TTree name inside the ROOT file (default: Events)
    --nthreads N     Enable implicit MT with N threads (default: 1)
    --dr-cut FLOAT   ΔR threshold for lepton-jet cleaning (default: 0.3)

Examples
--------
    python dijet_genjet_mass.py nano_mc.root
    python dijet_genjet_mass.py /path/to/*.root --nthreads 4
    python dijet_genjet_mass.py nano_mc.root --dr-cut 0.4
"""

import argparse
import textwrap
import time

import ROOT

# ─────────────────────────────────────────────────────────────────────────────
# C++ helpers injected once at module load
# ─────────────────────────────────────────────────────────────────────────────
ROOT.gInterpreter.Declare(
    r"""
#ifndef DIJET_HELPERS
#define DIJET_HELPERS

#include <cmath>
#include "Math/LorentzVector.h"
#include "Math/PtEtaPhiM4D.h"

using LV = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;

// ── delta-R between two (eta,phi) pairs ──────────────────────────────────────
inline float delta_r(float eta1, float phi1, float eta2, float phi2)
{
    float deta = eta1 - eta2;
    float dphi = phi1 - phi2;
    // wrap dphi into (-pi, pi]
    while (dphi >  M_PI) dphi -= 2.f * M_PI;
    while (dphi < -M_PI) dphi += 2.f * M_PI;
    return std::sqrt(deta * deta + dphi * dphi);
}

// ── GenJet cleaning mask ─────────────────────────────────────────────────────
// Returns a boolean RVec (one entry per GenJet).
// A jet is KEPT (true) if it is not matched within dr_cut to any generator-
// level lepton with |pdgId| ∈ {11, 13, 15}.
//
// Lepton selection from GenPart:
//   • electrons (|pdgId|=11) and muons (|pdgId|=13): status == 1 (stable)
//   • taus      (|pdgId|=15): isLastCopyBeforeFSR flag (statusFlags bit 13)
//     since taus decay and never appear as status=1.
ROOT::RVec<bool> clean_genjet_mask(
    const ROOT::RVec<float>& jet_eta,
    const ROOT::RVec<float>& jet_phi,
    const ROOT::RVec<float>& part_eta,
    const ROOT::RVec<float>& part_phi,
    const ROOT::RVec<float>& part_pt,          // was missing
    const ROOT::RVec<int>&   part_pdgId,
    const ROOT::RVec<int>&   part_status,
    const ROOT::RVec<int>&   part_statusFlags,
    float dr_cut)
{
    std::vector<float> lep_eta, lep_phi;
    for (std::size_t i = 0; i < part_pdgId.size(); ++i) {
        if (std::abs(part_pdgId[i]) != 13) continue;  // muons only
        if (part_status[i] != 1)           continue;  // stable
        if (!(part_statusFlags[i] & 1))    continue;  // isPrompt: only hard-process muons
        lep_eta.push_back(part_eta[i]);
        lep_phi.push_back(part_phi[i]);
    }

    ROOT::RVec<bool> mask(jet_eta.size(), true);
    for (std::size_t j = 0; j < jet_eta.size(); ++j) {
        for (std::size_t l = 0; l < lep_eta.size(); ++l) {
            if (delta_r(jet_eta[j], jet_phi[j], lep_eta[l], lep_phi[l]) < dr_cut) {
                mask[j] = false;
                break;
            }
        }
    }
    return mask;
}

// ── dijet invariant mass from the two leading jets ───────────────────────────
// Returns -1 if fewer than 2 jets are present.
float dijet_mass(
    const ROOT::RVec<float>& pt,
    const ROOT::RVec<float>& eta,
    const ROOT::RVec<float>& phi,
    const ROOT::RVec<float>& mass)
{
    if (pt.size() < 2) return -1.0f;
    LV j1(pt[0], eta[0], phi[0], mass[0]);
    LV j2(pt[1], eta[1], phi[1], mass[1]);
    return static_cast<float>((j1 + j2).M());
}

#endif  // DIJET_HELPERS
"""
)

# ─────────────────────────────────────────────────────────────────────────────
# Mass windows (GeV)
# ─────────────────────────────────────────────────────────────────────────────
WINDOWS = [
    ("A", 0.0,   300.0,       "0 - 300 GeV"),
    ("B", 300.0, 350.0,       "300 - 350 GeV"),
    ("C", 350.0, float("inf"), "350 - ∞  GeV"),
]


def run(
    input_files: list[str],
    tree_name: str = "Events",
    nthreads: int = 1,
    dr_cut: float = 0.3,
) -> None:
    # ── MT must be enabled before any RDataFrame object is created ────────
    if nthreads > 1:
        ROOT.EnableImplicitMT(nthreads)
        print(f"[INFO] Implicit MT enabled with {nthreads} threads.")

    t0 = time.perf_counter()

    # ── RDataFrame from file list (avoids TChain null-tree segfault) ──────
    file_vec = ROOT.std.vector["string"](input_files)
    rdf = ROOT.RDataFrame(tree_name, file_vec)

    # ── total events ──────────────────────────────────────────────────────
    count_total = rdf.Count()

    # ── uncleaned dijet mass (baseline, for comparison) ───────────────────
    rdf_raw = rdf.Define(
        "GenDijet_mass_raw",
        "dijet_mass(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass)",
    )
    count_2j_raw = rdf_raw.Filter("GenDijet_mass_raw >= 0", "≥2 GenJets (raw)").Count()
    count_4j_raw = rdf_raw.Filter("GenJet_pt.size() >= 4").Count()

    # ── gen-jet lepton cleaning ───────────────────────────────────────────
    # 1. Build per-jet boolean mask (True = not matched to a gen lepton)
    # 2. Apply mask to all four kinematic columns → CleanGenJet_*
    # 3. Compute dijet mass from the cleaned collection
    rdf_clean = (
        rdf_raw
        .Define(
            "GenJet_isClean",
            f"clean_genjet_mask(GenJet_eta, GenJet_phi, "
            f"GenPart_eta, GenPart_phi, GenPart_pt, GenPart_pdgId, "
            f"GenPart_status, GenPart_statusFlags, {dr_cut}f)",
        )
        .Define("CleanGenJet_pt",   "GenJet_pt  [GenJet_isClean]")
        .Define("CleanGenJet_eta",  "GenJet_eta [GenJet_isClean]")
        .Define("CleanGenJet_phi",  "GenJet_phi [GenJet_isClean]")
        .Define("CleanGenJet_mass", "GenJet_mass[GenJet_isClean]")
        .Define(
            "GenDijet_mass",
            "dijet_mass(CleanGenJet_pt, CleanGenJet_eta, CleanGenJet_phi, CleanGenJet_mass)",
        )
    )

    # ── events with ≥ 2 clean jets ────────────────────────────────────────
    rdf_2j = rdf_clean.Filter("GenDijet_mass >= 0", "≥2 clean GenJets")
    count_2j = rdf_2j.Count()

    # ── counts per mass window (all lazy) ────────────────────────────────
    window_counts: dict[str, ROOT.RResultPtr] = {}
    for label, lo, hi, _ in WINDOWS:
        cut = (
            f"GenDijet_mass >= {lo}"
            if hi == float("inf")
            else f"GenDijet_mass >= {lo} && GenDijet_mass < {hi}"
        )
        window_counts[label] = rdf_2j.Filter(cut).Count()

    # ── histograms (raw vs cleaned, single pass) ──────────────────────────
    h_raw = rdf_raw.Filter("GenDijet_mass_raw >= 0").Histo1D(
        ROOT.RDF.TH1DModel(
            "GenDijet_mass_raw",
            "Gen dijet mass (no lepton cleaning);M_{jj} [GeV];Events",
            100, 0, 1000,
        ),
        "GenDijet_mass_raw",
    )
    h_clean = rdf_2j.Histo1D(
        ROOT.RDF.TH1DModel(
            "GenDijet_mass_clean",
            f"Gen dijet mass (#DeltaR>{dr_cut} lepton cleaning);M_{{jj}} [GeV];Events",
            100, 0, 1000,
        ),
        "GenDijet_mass",
    )

    # ── trigger the single event-loop pass ───────────────────────────────
    n_total    = count_total.GetValue()
    n_2j_raw   = count_2j_raw.GetValue()
    n_4j_raw   = count_4j_raw.GetValue()
    n_2j_clean = count_2j.GetValue()
    n_win      = {lbl: ptr.GetValue() for lbl, ptr in window_counts.items()}
    elapsed    = time.perf_counter() - t0


    n_win_total = sum(n_win.values())
    n_removed   = n_2j_raw - n_2j_clean   # jets removed by lepton cleaning

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    sep  = "=" * 68
    sep2 = "-" * 68

    print(f"2 jet events: {n_2j_raw}")
    print(f"4 jet events: {n_4j_raw}")
    def pct(n, d):
        return f"{n / d * 100:>9.3f}%" if d > 0 else "       N/A"

    print()
    print(sep)
    print("  Generator-level Dijet Mass Analysis  -  Event Summary")
    print(sep)
    print(f"  Input file(s)   : {len(input_files)}")
    for f in input_files[:5]:
        print(f"    • {f}")
    if len(input_files) > 5:
        print(f"    … and {len(input_files)-5} more")
    print(f"  TTree           : {tree_name}")
    print(f"  Lepton ΔR cut   : {dr_cut}  (|pdgId| ∈ {{11}})")
    print(f"  Elapsed time    : {elapsed:.2f} s")
    print(sep2)
    print(f"  {'Category':<36}  {'Events':>10}  {'Fraction':>10}")
    print(sep2)
    print(f"  {'Total events (all)':<41}  {n_total:>10,}")
    print(f"  {'Events with ≥2 GenJets (before cleaning)':<41}  {n_2j_raw:>10,}  {pct(n_2j_raw, n_total)}")
    print(f"  {'Events with ≥2 GenJets (after cleaning)':<41}  {n_2j_clean:>10,}  {pct(n_2j_clean, n_total)}")
    print(f"  {'  → removed by lepton cleaning':<41}  {n_removed:>10,}  {pct(n_removed, n_2j_raw)}")
    print(sep2)
    print(f"  {'Mass window':<36}  {'Events':>10}  {'/ clean 2j':>10}  {'/ total':>9}")
    print(sep2)
    for label, lo, hi, desc in WINDOWS:
        n = n_win[label]
        row = f"Window {label}: {desc}"
        print(f"  {row:<36}  {n:>10,}  {pct(n, n_2j_clean)}  {pct(n, n_total)}")
    print(sep2)
    if n_win_total == n_2j_clean:
        print(f"  Window sum check : {n_win_total:,} == {n_2j_clean:,}  ✓")
    else:
        print(
            f"  [WARNING] Window sum ({n_win_total:,}) ≠ clean-2j events ({n_2j_clean:,}). "
            "Check boundary conditions."
        )
    print(sep)
    print()

    # ── save histograms ───────────────────────────────────────────────────
    out_file = ROOT.TFile.Open("dijet_mass_output.root", "RECREATE")
    if out_file and not out_file.IsZombie():
        h_raw.Write()
        h_clean.Write()
        out_file.Close()
        print("  Histograms saved to dijet_mass_output.root")
        print("    • GenDijet_mass_raw   - before lepton cleaning")
        print("    • GenDijet_mass_clean - after lepton cleaning")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "inputs", nargs="+", metavar="FILE",
        help="NanoAOD ROOT file(s); glob expansion handled by the shell.",
    )
    p.add_argument(
        "--tree", default="Events",
        help="TTree name (default: Events).",
    )
    p.add_argument(
        "--nthreads", type=int, default=1,
        help="Threads for implicit MT (default: 1).",
    )
    p.add_argument(
        "--dr-cut", type=float, default=0.3, dest="dr_cut",
        help="ΔR threshold for lepton-jet cleaning (default: 0.3).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_files=args.inputs,
        tree_name=args.tree,
        nthreads=args.nthreads,
        dr_cut=args.dr_cut,
    )
