#!/usr/bin/env python3
"""
Phase-0 VBF tag-jet pair study from EXISTING stage-1 output (no reprocessing).

Compares the current VBF pair (two leading-pT jets -> jj_mass_nominal) with the max-m_jj pair over all
selected jets (jj_mass_maxAllPairs_nominal). A different pair is possible only for njets >= 3, and the
two masses differ iff max-mjj picks another pair (ties aside), so the fraction of changed pairs is exact
from these two columns. The dEta/pT of the max-mjj pair is NOT stored, so the VBF category with the
max-mjj pair can only be approximated with an m_jj-only cut (flagged as such in the table).

Steps:
  skim  : read ~15 columns per sample (pyarrow, column projection) into a cache .parquet per sample.
          Data events with 115 <= m_mumu < 135 are dropped at read time (h-peak stays blinded).
  table : per sample x region x selection fractions (njets>=3, pair changed, m_jj-only VBF promotion).
  plots : Data/MC of lead-pT m_jj vs max-m_jj in z-peak and h-sidebands (default 2024 grouping,
          DY = dyTo2Mu_M-50_aMCatNLO, no DYVBF stitching).

How to run (repo root, default pixi env; no proxy / Dask needed):
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/phase0_maxmjj_vs_lead.py \
        --steps skim,table,plots [--workers 8] [--label Nov14_2024_phase0]

Example:
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/phase0_maxmjj_vs_lead.py --steps skim,table,plots --workers 8
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/phase0_maxmjj_vs_lead.py --steps table,plots   # reuse cache
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STAGE1 = (
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/"
    "Run3_nanoAODv15_FilterEvents_Nov14_tightPassLepVeto_OfficialRecomendation_CommonVBFBaseline_"
    "FatJet_Systematics_LumiSplit/stage1_output/2024/compacted"
)
CACHE = "/work/projects/hmm/shar1172/vbf_jet_selection/phase0_cache"

COLS = [
    "dimuon_mass", "njets_nominal", "jj_mass_nominal", "jj_mass_maxAllPairs_nominal", "jj_dEta_nominal",
    "jet1_pt_nominal", "jet2_pt_nominal", "jet1_eta_nominal", "jet2_eta_nominal",
    "nBtagLoose_nominal", "nBtagMedium_nominal", "wgt_nominal",
]
MC_COLS = ["jet1_hasMatchedGenJet_nominal", "jet2_hasMatchedGenJet_nominal", "gjj_mass"]

DATA = [f"data_{e}" for e in "CDEFGHI"]
# Data/MC stack grouping = configs/samples/samples.yaml for 2024 (DYVBF off)
GROUPS = {
    "DY": ["dyTo2Mu_M-50_aMCatNLO"],
    "TT": ["ttjets_dl", "ttjets_sl"],
    "ST": ["st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "EWK": ["ewk_mmjj_mll_105_160"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz_2l2nu", "zz_2l2q", "zz_4l"],
    "VVV": ["www", "wwz", "wzz", "zzz"],
}
TABLE_SAMPLES = [
    "vbf_powheg", "vbf_powheg_herwig", "vbf_aMCatNLO", 
    # "vbf_aMCatNLO_TuneCP5Up", "vbf_aMCatNLO_TuneCP5Down",
    "ggh_powhegPS", "ggh_amcatnlo", "dyTo2Mu_M-50_aMCatNLO", "dyTo2Mu_M-105To160", "dy_VBF_filter",
    "ewk_mmjj_mll_105_160", "ttjets_dl", "ttjets_sl", "zh_powheg", "wph_powheg", "wmh_powheg", "data",
]
SIGNALS = ["vbf_powheg", "ggh_powhegPS"]
ALL_SKIM = sorted(set(sum(GROUPS.values(), []) + TABLE_SAMPLES[:-1] + DATA))


def _read_file(args):
    path, is_data = args
    names = set(pq.read_schema(path).names)
    cols = [c for c in COLS + ([] if is_data else MC_COLS) if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    if is_data:
        df = df[~((df.dimuon_mass >= 115) & (df.dimuon_mass < 135))]  # blinded h-peak
        if "wgt_nominal" not in df:
            df["wgt_nominal"] = 1.0
    df = df[(df.dimuon_mass >= 70) & (df.dimuon_mass < 150)]
    return df.astype({c: "float32" for c in df.columns if df[c].dtype == "float64"})


def skim(workers):
    os.makedirs(CACHE, exist_ok=True)
    for s in ALL_SKIM:
        out = f"{CACHE}/{s}.parquet"
        if os.path.exists(out):
            continue
        files = sorted(glob.glob(f"{STAGE1}/{s}/*/*.parquet"))
        if not files:
            print(f"[skim] no files for {s}, skipping")
            continue
        with ProcessPoolExecutor(workers) as ex:
            dfs = list(ex.map(_read_file, [(f, s.startswith("data")) for f in files], chunksize=4))
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(out)
        print(f"[skim] {s}: {len(files)} files, {len(df)} events")


def load(s):
    if s == "data":
        return pd.concat([pd.read_parquet(f"{CACHE}/{d}.parquet") for d in DATA], ignore_index=True)
    return pd.read_parquet(f"{CACHE}/{s}.parquet")


def add_flags(df):
    nj = df.njets_nominal.fillna(0)
    bveto = ~((df.nBtagLoose_nominal >= 2) | ((df.nBtagMedium_nominal >= 1) & (nj >= 2)))
    mlead = df.jj_mass_nominal.fillna(-1)
    mmax = df.jj_mass_maxAllPairs_nominal.fillna(-1)
    df["nj2"] = (nj >= 2) & bveto
    df["nj3"] = (nj >= 3) & bveto
    # float32 storage: relative tolerance
    df["changed"] = (nj >= 3) & (mmax > mlead * (1 + 1e-4) + 1e-3)
    df["vbf_lead"] = df.nj2 & (mlead > 400) & (df.jj_dEta_nominal > 2.5) & (df.jet1_pt_nominal > 35)
    # max-mjj pair dEta/pT not stored -> mjj-only proxy
    df["vbfmjj_lead"] = df.nj2 & (mlead > 400)
    df["vbfmjj_max"] = df.nj2 & (mmax > 400)
    m = df.dimuon_mass
    df["z-peak"] = (m >= 70) & (m < 110)
    df["h-sidebands"] = ((m >= 110) & (m < 115)) | ((m >= 135) & (m < 150))
    df["h-peak"] = (m >= 115) & (m < 135)
    return df


def frac(num, den, w):
    d = w[den].sum()
    return w[num & den].sum() / d if d != 0 else np.nan


def table(outdir):
    rows = []
    for s in TABLE_SAMPLES:
        try:
            df = add_flags(load(s))
        except FileNotFoundError:
            continue
        w = df.wgt_nominal.fillna(0).values
        one = np.ones(len(df))
        regions = ["z-peak", "h-sidebands"] + ([] if s == "data" else ["h-peak"])
        for reg in regions:
            r = df[reg].values
            for sel, base in [("bveto_nj2", df.nj2.values), ("vbfcat_lead", df.vbf_lead.values)]:
                den = r & base
                row = dict(sample=s, region=reg, selection=sel, n_raw=int(den.sum()), yield_w=w[den].sum())
                for tag, ww in [("w", w), ("raw", one)]:
                    row[f"f_nj3_{tag}"] = frac(df.nj3.values, den, ww)
                    row[f"f_changed_{tag}"] = frac(df.changed.values, den, ww)
                    row[f"f_changed_given_nj3_{tag}"] = frac(df.changed.values, den & df.nj3.values, ww)
                if sel == "bveto_nj2":
                    a, b = (r & df.vbfmjj_lead.values), (r & df.vbfmjj_max.values)
                    row["mjj400_yield_lead"] = w[a].sum()
                    row["mjj400_yield_max"] = w[b].sum()
                    row["mjj400_gain_w"] = w[b].sum() / w[a].sum() - 1 if w[a].sum() else np.nan
                rows.append(row)
        print(f"[table] {s} done")
    out = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    out.to_csv(f"{outdir}/phase0_pair_change_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        cols = ["sample", "region", "selection", "n_raw", "yield_w", "f_nj3_w", "f_changed_w",
                "f_changed_given_nj3_w", "mjj400_gain_w"]
        print(out[cols].to_string(index=False))
    return out


def plots(outroot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")

    data = add_flags(load("data"))
    mc = {g: add_flags(pd.concat([load(s) for s in ss if os.path.exists(f"{CACHE}/{s}.parquet")],
                                 ignore_index=True)) for g, ss in GROUPS.items()}
    sig = {s: add_flags(load(s)) for s in SIGNALS}
    bins = np.array([0, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 3000, 4500])
    variables = [("jj_mass_nominal", "lead-$p_T$ pair $m_{jj}$ [GeV]"),
                 ("jj_mass_maxAllPairs_nominal", "max-$m_{jj}$ pair $m_{jj}$ [GeV]")]
    summary = []
    for sel in ["nj2", "nj3"]:
        for reg in ["z-peak", "h-sidebands"]:
            odir = f"{outroot}/bveto_{sel}_{reg}"
            os.makedirs(odir, exist_ok=True)
            for var, xl in variables:
                hd, _ = np.histogram(data.loc[data[sel] & data[reg], var], bins)
                stack = []
                for g, df in mc.items():
                    m = df[sel] & df[reg]
                    stack.append(np.histogram(df.loc[m, var], bins, weights=df.loc[m, "wgt_nominal"])[0])
                tot = np.sum(stack, axis=0)
                bw = np.diff(bins)
                summary.append(dict(selection=sel, region=reg, var=var, data=hd.sum(), mc=tot.sum(),
                                    data_over_mc=hd.sum() / tot.sum()))
                fig, (ax, rx) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
                hep.histplot([h / bw for h in stack], bins, stack=True, histtype="fill", label=list(mc), ax=ax)
                hep.histplot(hd / bw, bins, yerr=np.sqrt(hd) / bw, histtype="errorbar", color="k", label="Data", ax=ax)
                for (s, df), col in zip(sig.items(), ["red", "black"]):
                    m = df[sel] & df[reg]
                    hs = np.histogram(df.loc[m, var], bins, weights=df.loc[m, "wgt_nominal"])[0]
                    hep.histplot(hs * 100 / bw, bins, label=f"{s} x100", ax=ax, lw=2, color=col, ls="--")
                ax.set_yscale("log")
                ax.set_ylabel("Events / GeV")
                ax.legend(ncol=2, fontsize=13, loc="upper right")
                hep.cms.label("Preliminary", data=True, lumi=109.82, year="2024", ax=ax, fontsize=16)
                ax.text(0.03, 0.04, f"{reg}, b-veto, njets$\\geq${sel[-1]}", transform=ax.transAxes, fontsize=14)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx.errorbar(0.5 * (bins[1:] + bins[:-1]), hd / tot, yerr=np.sqrt(hd) / tot, fmt="ko")
                rx.axhline(1, color="gray", ls="--")
                rx.set_ylim(0.4, 1.4)
                rx.set_ylabel("Data/MC")
                rx.set_xlabel(xl)
                for ext in ("png", "pdf"):
                    fig.savefig(f"{odir}/{var}.{ext}", bbox_inches="tight")
                plt.close(fig)
    pd.DataFrame(summary).to_csv(f"{outroot}/phase0_datamc_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default="skim,table,plots")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--label", default="Nov14_2024_phase0")
    a = p.parse_args()
    outroot = f"validation/vbf_jet_selection/{a.label}"
    steps = a.steps.split(",")
    if "skim" in steps:
        skim(a.workers)
    if "table" in steps:
        table(outroot)
    if "plots" in steps:
        plots(outroot)
