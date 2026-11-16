#!/usr/bin/env python3
"""
Generator-matching efficiency of the VBF tag jets and ggH contamination, from stage-1 output written with
save_alt_vbf_pairs (configs/switches/switches_vbf_pair_study.yaml). Reuses the reading, truth and category
helpers of scripts/vbf_jet_selection/compare_pair_methods.py, so definitions are identical.

VBF quarks: the two outgoing quarks of the truth (events with exactly 2; qg->qqq events are counted separately).
Two truths are compared side by side, each with two matching chains:
  truth lhe : outgoing LHE partons (every generator)
  truth genq: GenPart hard-process quarks after the shower (fromHardProcess & isLastCopy); Pythia only
  chain gen : quark -> GenJet (dR<0.4) of a selected reco jet (Jet_genJetIdx)
  chain reco: quark -> selected reco jet directly (dR<0.4)
Outputs (CSV + printed tables) per year and VBF sample:
  1. matching efficiency vs reco jet multiplicity (0, 1, >=2 selected jets): fraction of events in which 2 / 1 / 0
     of the VBF quarks are matched to a selected jet (any jet; for >=2 jets this is the pair-selection ceiling)
  2. pair matching for the chosen pair method(s) (njets >= 2, b-veto; and in the VBF category): both / one / no
     selected-pair jets matched to a VBF quark
  3. ggH contamination ggH/(ggH+VBF) (and VH) in the cut-based VBF category, inclusive and per jet multiplicity
Weights: wgt_nominal (each year normalised to its own luminosity); events in the chosen dimuon-mass region.

How to run (repo root, default pixi env; reads parquet only, no proxy / Dask):
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/gen_matching_efficiency.py \
        --input <stage1_output dir containing <year>/f1_0> [--years 2024 2023BPix] [--methods leadpt maxmjj] \
        [--region h-peak] [--label <out label>]

Example:
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/gen_matching_efficiency.py \
        --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterEvents_Nov16_VBFPairStudy_Signal_LumiSplit/stage1_output \
        --years 2024 --methods leadpt leadpt_maxmjj maxmjj maxdeta maxmjj_deta25 maxmjj_opphemi maxmjj_bothpt30 maxmjj_bothpt35 maxqvg maxqvg_deta25 --label Nov16_genmatch

        # maxdeta maxmjj_deta25 maxmjj_opphemi maxmjj_bothpt30 maxmjj_bothpt35 maxqvg maxqvg_deta25
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pair_methods import TRUTH_SOURCES, method_columns, read_sample, regions, truth  # noqa: E402

VBF_PREFIX = "vbf_"
GGH_SAMPLE = "ggh_powhegPS"
VH_SAMPLES = ["zh_powheg", "wph_powheg", "wmh_powheg"]
NJ_BINS = {"0j": lambda nj: nj == 0, "1j": lambda nj: nj == 1, ">=2j": lambda nj: nj >= 2}


def wfrac(w, num, den):
    d = w[den].sum()
    return w[num & den].sum() / d if d != 0 else np.nan


def matching_tables(year, sample, df, methods, reg):
    """Rows for (1) jet-multiplicity matching efficiency and (2) pair matching, per truth and chain."""
    eff_rows, pair_rows = [], []
    w = df.wgt_nominal.fillna(0).to_numpy()
    nj = df.njets_nominal.fillna(0).to_numpy()
    mcols = {m: method_columns(df, m) for m in methods if f"vbf_{m}_mjj_nominal" in df}
    for src in TRUTH_SOURCES:
        if f"vbf_{src}1_pdgId" not in df or df[f"vbf_{src}1_pdgId"].isna().all():
            continue  # e.g. genq for Herwig (no hard-process flags)
        T = truth(df, src)
        qq = T["qq"] & reg
        for chain in ("gen", "reco"):
            m = T[f"match_{chain}"]
            nmatch_any = (m[:, 0] >= 0).astype(int) + (m[:, 1] >= 0).astype(int)
            for bname, fbin in NJ_BINS.items():
                den = qq & fbin(nj)
                eff_rows.append(dict(year=year, sample=sample, truth=src, chain=chain, njets=bname,
                                     n_raw=int(den.sum()), frac_of_qq=wfrac(w, den, qq),
                                     matched2=wfrac(w, nmatch_any == 2, den), matched1=wfrac(w, nmatch_any == 1, den),
                                     matched0=wfrac(w, nmatch_any == 0, den)))
            for meth, mc in mcols.items():
                ia, ib = mc.ia.to_numpy(), mc.ib.to_numpy()
                npair = sum(((m[:, k] >= 0) & ((m[:, k] == ia) | (m[:, k] == ib))).astype(int) for k in range(2))
                for sel, smask in (("bveto_nj2", mc.bveto_nj2.to_numpy()), ("vbfcat", mc.vbf.to_numpy())):
                    den = qq & smask
                    pair_rows.append(dict(year=year, sample=sample, truth=src, chain=chain, method=meth, selection=sel,
                                          n_raw=int(den.sum()), both=wfrac(w, npair == 2, den),
                                          one=wfrac(w, npair == 1, den), none=wfrac(w, npair == 0, den),
                                          ceiling=wfrac(w, nmatch_any == 2, den)))
        eff_rows.append(dict(year=year, sample=sample, truth=src, chain="-", njets="all", n_raw=int(reg.sum()),
                             frac_of_qq=np.nan, matched2=wfrac(w, T["qq"], reg), matched1=wfrac(w, T["nq"] == 3, reg),
                             matched0=np.nan, note="matched2 = frac qq, matched1 = frac 3-quark (qg) events"))
    return eff_rows, pair_rows


def contamination_rows(year, inp, vbf_sample, methods, region_name):
    """ggH/(ggH+VBF) and VH share in the cut-based VBF category, inclusive and per njets bin."""
    frames = {}
    for s in [vbf_sample, GGH_SAMPLE] + VH_SAMPLES:
        df = read_sample(inp, s)
        if df is not None:
            frames[s] = df
    if vbf_sample not in frames or GGH_SAMPLE not in frames:
        return []
    rows = []
    for meth in methods:
        y = {}
        for s, df in frames.items():
            if f"vbf_{meth}_mjj_nominal" not in df:
                continue
            mc = method_columns(df, meth)
            sel = mc.vbf.to_numpy() & regions(df)[region_name]
            w = df.wgt_nominal.fillna(0).to_numpy()
            nj = df.njets_nominal.fillna(0).to_numpy()
            y[s] = {b: w[sel & f(nj)].sum() for b, f in {"all": lambda x: x >= 0, "nj2": lambda x: x == 2,
                                                          "nj3": lambda x: x == 3, "nj4p": lambda x: x >= 4}.items()}
        for b in ("all", "nj2", "nj3", "nj4p"):
            v, g = y[vbf_sample][b], y[GGH_SAMPLE][b]
            vh = sum(y[s][b] for s in VH_SAMPLES if s in y)
            rows.append(dict(year=year, method=meth, region=region_name, njets=b, VBF=v, ggH=g, VH=vh,
                             ggH_frac=g / (g + v) if g + v else np.nan,
                             VH_frac=vh / (v + g + vh) if v + g + vh else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="stage1_output directory containing <year>/f1_0")
    ap.add_argument("--years", nargs="*", default=None, help="default: every year directory found")
    ap.add_argument("--methods", nargs="*", default=["leadpt"])
    ap.add_argument("--region", default="h-peak", choices=["h-peak", "h-sidebands", "z-peak"])
    ap.add_argument("--label", default="gen_matching")
    a = ap.parse_args()
    years = a.years or sorted(os.path.basename(d) for d in glob.glob(f"{a.input}/*") if os.path.isdir(f"{d}/f1_0"))
    outdir = f"validation/vbf_jet_selection/{a.label}"
    os.makedirs(outdir, exist_ok=True)
    eff, pair, cont = [], [], []
    for year in years:
        inp = f"{a.input}/{year}"
        vbf_samples = sorted(os.path.basename(d) for d in glob.glob(f"{inp}/f1_0/{VBF_PREFIX}*"))
        for s in vbf_samples:
            df = read_sample(inp, s)
            if df is None:
                continue
            reg = regions(df)[a.region]
            e, p = matching_tables(year, s, df, a.methods, reg)
            eff += e
            pair += p
            print(f"[{year}] {s}: {len(df)} events")
        nominal_vbf = next((s for s in ("vbf_powheg", "vbf_powheg_dipole") if s in vbf_samples), None)
        if nominal_vbf:
            cont += contamination_rows(year, inp, nominal_vbf, a.methods, a.region)
    E, P, C = pd.DataFrame(eff), pd.DataFrame(pair), pd.DataFrame(cont)
    E.to_csv(f"{outdir}/matching_eff_vs_njets.csv", index=False)
    P.to_csv(f"{outdir}/pair_matching.csv", index=False)
    C.to_csv(f"{outdir}/ggh_contamination.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.float_format", "{:.3f}".format):
        if len(E):
            print("\n=== 1. VBF quarks matched to selected jets, by reco jet multiplicity (qq events)")
            print(E[E.njets != "all"].pivot_table(index=["year", "sample", "njets"], columns=["truth", "chain"],
                                                  values=["frac_of_qq", "matched2", "matched1", "matched0"]).to_string())
        if len(P):
            print("\n=== 2. Pair matching (both / one / none of the selected pair matched), GenJet chain")
            print(P[P.chain == "gen"].pivot_table(index=["year", "sample", "method", "selection"], columns="truth",
                                                  values=["both", "one", "none", "ceiling"]).to_string())
        if len(C):
            print("\n=== 3. ggH contamination in the VBF category")
            print(C.to_string(index=False))
    print(f"\nwritten: {outdir}/matching_eff_vs_njets.csv, pair_matching.csv, ggh_contamination.csv")


if __name__ == "__main__":
    main()
