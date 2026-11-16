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
  Plots (validation/vbf_jet_selection/<label>/plots_<region>/): matching_vs_njets_<year>, truth_comparison_<year>
  (LHE vs GenPart, GenJet vs reco chain), pair_matching_<year>, ggh_contamination_<year>, summary_vs_year (>1 year).
Weights: wgt_nominal (each year normalised to its own luminosity); events in the chosen dimuon-mass region.

How to run (repo root, default pixi env; reads parquet only, no proxy / Dask):
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/gen_matching_efficiency.py \
        --input <stage1_output dir containing <year>/f1_0> [--years 2024 2023BPix] [--methods leadpt maxmjj] \
        [--region h-peak] [--label <out label>] [--samples vbf_powheg ...] [--skip TuneCP5] [--replot | --no-plots]

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
YEAR_ORDER = ["2022preEE", "2022postEE", "2023", "2023BPix", "2024", "2025", "2026"]


def year_sorted(years):
    return sorted(set(map(str, years)), key=lambda y: (YEAR_ORDER.index(y) if y in YEAR_ORDER else 99, y))


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


# ----------------------------------------------------------------------------- plots
MATCH_COLORS = {"matched2": "#2a9d8f", "matched1": "#e9c46a", "matched0": "#adb5bd"}
PAIR_COLORS = {"both": "#2a9d8f", "one": "#e9c46a", "none": "#e76f51"}
NJ_ORDER = ["0j", "1j", ">=2j"]
NJ_LABEL = {"0j": "0 jets", "1j": "1 jet", ">=2j": r"$\geq$2 jets"}
SAMPLE_LABEL = {"vbf_powheg": "Powheg+Pythia8", "vbf_powheg_dipole": "Powheg+Pythia8 (dipole)",
                "vbf_powheg_herwig": "Powheg+Herwig7", "vbf_aMCatNLO": "aMC@NLO+Pythia8"}
TRUTH_LABEL = {"genq": "GenPart quarks", "lhe": "LHE partons"}


def _best_truth(df):
    """genq where available (Pythia), else lhe (Herwig)."""
    return "genq" if "genq" in set(df.truth) else "lhe"


def make_plots(E, P, C, outdir, region, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    pdir = f"{outdir}/plots_{region}"
    os.makedirs(pdir, exist_ok=True)

    def label(ax, year, loc=0):
        # data=False already prints "Simulation"; Run 3 is 13.6 TeV
        hep.cms.label("Preliminary", data=False, rlabel=f"{year} (13.6 TeV)", ax=ax, fontsize=15, loc=loc)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(f"{pdir}/{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    E = E[E.njets != "all"] if len(E) else E
    for year in year_sorted(E.year) if len(E) else []:
        Ey = E[E.year == year]
        samples = sorted(set(Ey["sample"]), key=lambda x: (x not in ("vbf_powheg", "vbf_powheg_dipole"), x))
        # 1. matched VBF quarks vs reco jet multiplicity (stacked), one panel per sample
        fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 6.5), sharey=True, squeeze=False)
        for ax, smp in zip(axes[0], samples):
            d = Ey[(Ey["sample"] == smp) & (Ey.chain == "gen")]
            d = d[d.truth == _best_truth(d)].set_index("njets").reindex(NJ_ORDER)
            bottom = np.zeros(len(NJ_ORDER))
            for col, lab in (("matched2", "2 quarks matched"), ("matched1", "1 quark matched"), ("matched0", "0 matched")):
                v = d[col].fillna(0).to_numpy()
                ax.bar(range(len(NJ_ORDER)), v, bottom=bottom, color=MATCH_COLORS[col], label=lab, width=0.7)
                for i, (b, x) in enumerate(zip(bottom, v)):
                    if x > 0.04:
                        ax.text(i, b + x / 2, f"{x:.2f}", ha="center", va="center", fontsize=12)
                bottom += v
            for i, f in enumerate(d.frac_of_qq.fillna(0)):
                ax.text(i, 1.02, f"{100 * f:.0f}% of events", ha="center", fontsize=11)
            ax.set_xticks(range(len(NJ_ORDER)), [NJ_LABEL[n] for n in NJ_ORDER])
            ax.set_ylim(0, 1.18)
            ax.set_title(f"{SAMPLE_LABEL.get(smp, smp)}\n({TRUTH_LABEL[_best_truth(Ey[Ey['sample'] == smp])]}, via GenJet)",
                         fontsize=13, pad=8)
        axes[0][0].set_ylabel("Fraction of events", fontsize=20)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=13, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
        fig.text(0.01, 1.02, f"CMS Simulation Preliminary   {year} (13.6 TeV)   VBF quarks matched to selected jets",
                 fontsize=15, fontweight="bold", va="bottom")
        save(fig, f"matching_vs_njets_{year}")

        # 2. LHE vs GenPart truth, GenJet vs direct reco chain: matched2 in the >=2-jet bin
        d = Ey[Ey.njets == ">=2j"]
        fig, ax = plt.subplots(figsize=(1.8 + 2.6 * len(samples), 6))
        combos = [("lhe", "gen"), ("lhe", "reco"), ("genq", "gen"), ("genq", "reco")]
        cols = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"]
        wbar = 0.2
        for k, ((t, c), col) in enumerate(zip(combos, cols)):
            vals = [d[(d["sample"] == smp) & (d.truth == t) & (d.chain == c)].matched2.mean() for smp in samples]
            xs = np.arange(len(samples)) + (k - 1.5) * wbar
            ax.bar(xs, np.nan_to_num(vals), width=wbar, color=col,
                   label=f"{TRUTH_LABEL[t]}, {'via GenJet' if c == 'gen' else 'reco jet direct'}")
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, rotation=90)
        ax.set_xticks(range(len(samples)), [SAMPLE_LABEL.get(s_, s_) for s_ in samples], fontsize=13)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel(r"Both VBF quarks matched ($\geq$2 jets)", fontsize=18)
        ax.legend(fontsize=12, ncol=2, loc="upper left")
        label(ax, year)
        save(fig, f"truth_comparison_{year}")

    # 3. pair matching per method, nominal VBF sample (stacked horizontal bars + ceiling)
    for year in year_sorted(P.year) if len(P) else []:
        Py = P[(P.year == year) & (P.chain == "gen")]
        smp = next((x for x in ("vbf_powheg", "vbf_powheg_dipole") if x in set(Py["sample"])), None)
        if smp is None:
            continue
        d = Py[Py["sample"] == smp]
        d = d[d.truth == _best_truth(d)]
        meths = [m for m in methods if m in set(d.method)]
        fig, axes = plt.subplots(1, 2, figsize=(15, 2.5 + 0.7 * len(meths)), sharey=True)
        for ax, (sel, title) in zip(axes, (("bveto_nj2", r"b-veto, $\geq$2 jets"), ("vbfcat", "VBF category"))):
            dd = d[d.selection == sel].set_index("method").reindex(meths)
            left = np.zeros(len(meths))
            for col, lab in (("both", "both jets matched"), ("one", "one matched"), ("none", "none")):
                v = dd[col].fillna(0).to_numpy()
                ax.barh(range(len(meths)), v, left=left, color=PAIR_COLORS[col], label=lab, height=0.7)
                for i, (l_, x) in enumerate(zip(left, v)):
                    if x > 0.06:
                        ax.text(l_ + x / 2, i, f"{x:.3f}", ha="center", va="center", fontsize=11)
                left += v
            ax.scatter(dd.ceiling, range(len(meths)), marker="|", s=300, color="k", zorder=3, label="ceiling")
            ax.set_xlim(0, 1)
            ax.set_title(title, fontsize=14, pad=8)
            ax.set_xlabel("Fraction of events", fontsize=16)
        axes[0].set_yticks(range(len(meths)), meths, fontsize=12)
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=12, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        fig.text(0.01, 1.0, f"CMS Simulation Preliminary   {year} (13.6 TeV)   "
                 f"{SAMPLE_LABEL.get(smp, smp)}: chosen pair vs VBF quarks ({TRUTH_LABEL[_best_truth(d)]}, via GenJet)",
                 fontsize=14, fontweight="bold", va="bottom")
        save(fig, f"pair_matching_{year}")

    # 4. ggH contamination in the VBF category vs jet multiplicity, per method
    for year in year_sorted(C.year) if len(C) else []:
        d = C[C.year == year]
        bins = ["all", "nj2", "nj3", "nj4p"]
        blab = {"all": "inclusive", "nj2": "2 jets", "nj3": "3 jets", "nj4p": r"$\geq$4 jets"}
        meths = [m for m in methods if m in set(d.method)]
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(bins) * max(1, len(meths)) * 0.5 + 4), 6))
        wbar = 0.8 / max(1, len(meths))
        cmap = plt.get_cmap("tab10")
        for k, m in enumerate(meths):
            dm = d[d.method == m].set_index("njets").reindex(bins)
            xs = np.arange(len(bins)) + (k - (len(meths) - 1) / 2) * wbar
            ax.bar(xs, dm.ggH_frac.fillna(0), width=wbar, color=cmap(k % 10), label=m)
            if len(meths) <= 4:
                for x, v in zip(xs, dm.ggH_frac):
                    if np.isfinite(v):
                        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_xticks(range(len(bins)), [blab[b] for b in bins])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("ggH / (ggH + VBF), VBF category", fontsize=17)
        ax.legend(fontsize=11, ncol=2, loc="upper left")
        label(ax, year)
        save(fig, f"ggh_contamination_{year}")

    # 5. year summary: ceiling, leadpt pair matching (VBF category) and ggH fraction for the nominal sample
    if len(E) and len(set(E.year)) > 1:
        years = year_sorted(E.year)
        rows = []
        for y in years:
            e = E[(E.year == y) & (E.njets == ">=2j") & (E.chain == "gen") & E["sample"].isin(["vbf_powheg", "vbf_powheg_dipole"])]
            e = e[e.truth == _best_truth(e)] if len(e) else e
            p = P[(P.year == y) & (P.chain == "gen") & (P.method == methods[0]) & (P.selection == "vbfcat")
                  & P["sample"].isin(["vbf_powheg", "vbf_powheg_dipole"])] if len(P) else P
            p = p[p.truth == _best_truth(p)] if len(p) else p
            c = C[(C.year == y) & (C.method == methods[0]) & (C.njets == "all")] if len(C) else C
            rows.append((y, e.matched2.mean() if len(e) else np.nan, p.both.mean() if len(p) else np.nan,
                         c.ggH_frac.mean() if len(c) else np.nan))
        r = np.array([x[1:] for x in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, (lab, mk) in enumerate(((r"ceiling ($\geq$2 jets)", "o"), (f"{methods[0]}: both matched (VBF cat.)", "s"),
                                       (f"{methods[0]}: ggH/(ggH+VBF) (VBF cat.)", "^"))):
            ax.plot(range(len(years)), r[:, j], marker=mk, ms=9, lw=1.5, label=lab)
        ax.set_xticks(range(len(years)), years, rotation=20)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Fraction")
        ax.legend(fontsize=12, loc="lower left")
        hep.cms.label("Preliminary", data=False, rlabel="Run 3 (13.6 TeV)", ax=ax, fontsize=15, loc=0)
        save(fig, "summary_vs_year")
    print(f"plots: {pdir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="stage1_output directory containing <year>/f1_0")
    ap.add_argument("--years", nargs="*", default=None, help="default: every year directory found")
    ap.add_argument("--methods", nargs="*", default=["leadpt"])
    ap.add_argument("--region", default="h-peak", choices=["h-peak", "h-sidebands", "z-peak"])
    ap.add_argument("--label", default="gen_matching")
    # samples come from the stage-1 output directories (not the dataset yaml): filter them here
    ap.add_argument("--samples", nargs="*", default=None, help="only these VBF samples (default: all vbf_* found)")
    ap.add_argument("--skip", nargs="*", default=["TuneCP5"],
                    help="drop VBF samples whose name contains any of these strings (default: TuneCP5; '--skip' = none)")
    ap.add_argument("--no-plots", action="store_true", help="tables only")
    ap.add_argument("--replot", action="store_true", help="redo the plots from the CSVs of --label (no parquet reading)")
    a = ap.parse_args()
    years = a.years or year_sorted(os.path.basename(d) for d in glob.glob(f"{a.input}/*") if os.path.isdir(f"{d}/f1_0"))
    outdir = f"validation/vbf_jet_selection/{a.label}"
    os.makedirs(outdir, exist_ok=True)
    if a.replot:
        rd = lambda n: pd.read_csv(f"{outdir}/{n}") if os.path.exists(f"{outdir}/{n}") else pd.DataFrame()  # noqa: E731
        make_plots(rd("matching_eff_vs_njets.csv"), rd("pair_matching.csv"), rd("ggh_contamination.csv"), outdir, a.region,
                   a.methods)
        return
    eff, pair, cont = [], [], []
    for year in years:
        inp = f"{a.input}/{year}"
        vbf_samples = sorted(os.path.basename(d) for d in glob.glob(f"{inp}/f1_0/{VBF_PREFIX}*"))
        vbf_samples = [s for s in vbf_samples if (a.samples is None or s in a.samples)
                       and not any(p in s for p in a.skip)]
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
    if not a.no_plots:
        make_plots(E, P, C, outdir, a.region, a.methods)


if __name__ == "__main__":
    main()
