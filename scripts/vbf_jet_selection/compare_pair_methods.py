#!/usr/bin/env python3
"""
Compare VBF tag-jet pair selection methods on stage-1 output written with save_alt_vbf_pairs
(configs/switches/switches_vbf_pair_study.yaml). Stage-1 (pick_vbf_pairs -> pair_dict["pairs_by_method"]) stores,
per method, the chosen pair as vbf_<method>_{jet1,jet2}_{pt,eta,phi,idx,btagPNetQvG,hasMatchedGenJet}_nominal and
vbf_<method>_{mjj,deta}_nominal. Per event it stores two truths, each with the selected jet matched to every parton
(dR<0.4) via its GenJet (..._jetidx_gen) and directly (..._jetidx_reco):
  vbf_lhe<i>_*  : outgoing LHE partons (every generator)
  vbf_genq<i>_* : GenPart hard-process quarks after the shower (fromHardProcess & isLastCopy); Pythia only,
                  Herwig NanoAOD has no hard-process flags.
Methods: leadpt (two leading-pT jets), maxmjj, maxdeta, maxmjj_deta25, maxmjj_opphemi (opposite eta hemispheres),
leadpt_maxmjj (leading jet + max-mjj partner), maxmjj_bothpt30/35, maxqvg (highest PNet QvG sum), maxqvg_deta25.
jet1 is always the higher-pT jet of the pair.

For each method the cut-based VBF category (b-veto & m_jj>400 & dEta_jj>2.5 [rounded 1e-4] & jet1 pT>35, as
modules/selection.applyRegionCatCuts) is applied to that method's pair; ggH = b-veto & !VBF. Outputs:
  1. Data/MC of jet1/jet2 pT and eta, m_jj, dEta_jj in z-peak and h-sidebands, for b-veto njets>=2 and the VBF
     category. Data h-peak (115-135) is dropped at read time.
  2. Truth matching on signal, per truth (column "truth" = lhe / genq): VBF quarks = the two outgoing quarks
     (events with exactly 2; 3-quark qg events reported separately). Pair purity / partial / ceiling (both quarks
     matched to distinct selected jets) and method-independent efficiencies (sel_eff, correct_eff), inclusive and
     vs njets, quark |eta|, PV_npvsGood and true m_qq, for the GenJet chain and the direct chain.
  3. ggH/(ggH+VBF) in the VBF category (h-peak MC).
  4. Yields per process group in the VBF and ggH categories with MC-stat errors and S/sqrt(B).
DY: z-peak from inclusive M-50; h-sidebands/h-peak from M-105To160 (gjj_mass<=300) + VBF filter (>300).

How to run (repo root, default pixi env; reads parquet only, no proxy / Dask):
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/compare_pair_methods.py \
        --input <stage1_output>/2024 --label <label> [--data-eras C D] [--steps cache,truth,yields,plots]

Example:
    ./run_in_pixi.sh default python scripts/vbf_jet_selection/compare_pair_methods.py \
        --input /work/projects/hmm/shar1172/vbf_jet_selection/test_stage1/A7/stage1_output/2024 \
        --label test_A7 --steps cache,truth,yields
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

METHODS = ["leadpt", "maxmjj", "maxdeta", "maxmjj_deta25", "maxmjj_opphemi", "leadpt_maxmjj", "maxmjj_bothpt30",
           "maxmjj_bothpt35", "maxqvg", "maxqvg_deta25"]
NL = 4
ETA_CUT_DECIMALS = 4  # = modules/selection.py: dEta rounded before the 2.5 cut
DY_GJJ_STITCH = 300.0  # Run-3 VBF-filter stitching on gjj_mass, as modules/selection.py (do_vbf_filter_study)

GROUPS = {
    "DY": ["dyTo2Mu_M-50_aMCatNLO", "dyTo2Mu_M-105To160", "dy_VBF_filter"],
    "TT": ["ttjets_dl", "ttjets_sl"],
    "ST": ["st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "EWK": ["ewk_mmjj_mll_105_160"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz_2l2nu", "zz_2l2q", "zz_4l"],
    "VVV": ["www", "wwz", "wzz", "zzz"],
}
SIG = {"VBF": ["vbf_powheg"], "ggH": ["ggh_powhegPS"], "VH": ["zh_powheg", "wph_powheg", "wmh_powheg"]}
TRUTH_SAMPLES = ["vbf_powheg", "vbf_powheg_herwig", "vbf_aMCatNLO", 
                # "vbf_aMCatNLO_TuneCP5Up", "vbf_aMCatNLO_TuneCP5Down"
                ]
EVT_COLS = ["dimuon_mass", "njets_nominal", "nBtagLoose_nominal", "nBtagMedium_nominal", "wgt_nominal",
            "PV_npvsGood", "jet1_pt_nominal", "jj_mass_nominal", "jj_dEta_nominal", "gjj_mass"]
PAIR_F = ["jet1_pt", "jet1_eta", "jet2_pt", "jet2_eta", "mjj", "deta", "jet1_idx", "jet2_idx",
          "jet1_hasMatchedGenJet", "jet2_hasMatchedGenJet"]
LHE_F = ["pt", "eta", "phi", "pdgId", "jetidx_reco", "jetidx_gen"]
TRUTH_SOURCES = ["lhe", "genq"]  # LHE partons; GenPart fromHardProcess & isLastCopy quarks


# ----------------------------------------------------------------------------- input
def sample_files(inp, s):
    for pat in (f"{inp}/compacted/{s}/*/*.parquet", f"{inp}/f1_0/{s}/*/*.parquet", f"{inp}/{s}/*/*.parquet"):
        f = sorted(glob.glob(pat))
        if f:
            return f
    return []


def read_sample(inp, s):
    files = sample_files(inp, s)
    if not files:
        return None
    names = set(pq.read_schema(files[0]).names)
    if "vbf_leadpt_mjj_nominal" not in names:
        raise RuntimeError(f"{s}: no vbf_<method>_* columns (stage-1 not run with save_alt_vbf_pairs)")
    cols = [c for c in EVT_COLS if c in names]
    cols += [f"vbf_{m}_{f}_nominal" for m in METHODS for f in PAIR_F if f"vbf_{m}_{f}_nominal" in names]
    cols += [f"vbf_{t}{i}_{f}" for t in TRUTH_SOURCES for i in range(1, NL + 1) for f in LHE_F if f"vbf_{t}{i}_{f}" in names]
    dfs = []
    for f in files:
        d = pq.read_table(f, columns=cols).to_pandas()
        if s.startswith("data"):
            d = d[~((d.dimuon_mass >= 115) & (d.dimuon_mass < 135))]  # blinded h-peak
            d["wgt_nominal"] = 1.0
        dfs.append(d[(d.dimuon_mass >= 70) & (d.dimuon_mass < 150)])
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------------- per-method columns
def method_columns(df, method):
    c = lambda f: df[f"vbf_{method}_{f}_nominal"].to_numpy(dtype=float)  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["j1_pt"], out["j1_eta"], out["j2_pt"], out["j2_eta"] = c("jet1_pt"), c("jet1_eta"), c("jet2_pt"), c("jet2_eta")
    out["mjj"], out["deta"] = c("mjj"), c("deta")
    out["has"] = np.isfinite(out.mjj.to_numpy())
    out["ia"] = np.nan_to_num(c("jet1_idx"), nan=-1).astype(int)
    out["ib"] = np.nan_to_num(c("jet2_idx"), nan=-1).astype(int)
    if f"vbf_{method}_jet1_hasMatchedGenJet_nominal" in df:
        g1 = df[f"vbf_{method}_jet1_hasMatchedGenJet_nominal"].fillna(True).astype(bool).to_numpy()
        g2 = df[f"vbf_{method}_jet2_hasMatchedGenJet_nominal"].fillna(True).astype(bool).to_numpy()
        out["pu"] = (~g1 | ~g2) & out.has.to_numpy()  # >=1 tag jet without GenJet (PU proxy)
    nj = df.njets_nominal.fillna(0).to_numpy()
    bveto = ~((df.nBtagLoose_nominal >= 2) | ((df.nBtagMedium_nominal >= 1) & (nj >= 2))).to_numpy()
    out["bveto_nj2"] = bveto & (nj >= 2)
    deta_r = np.rint(out.deta.to_numpy() * 10.0**ETA_CUT_DECIMALS) / 10.0**ETA_CUT_DECIMALS
    out["vbf"] = out.bveto_nj2 & out.has & (out.mjj > 400) & (deta_r > 2.5) & (out.j1_pt > 35)
    out["ggh"] = bveto & ~out.vbf
    return out


def regions(df, sample=""):
    """Mass windows; for DY samples also the stitching role (see GROUPS["DY"])."""
    m = df.dimuon_mass.to_numpy()
    R = {"z-peak": (m >= 70) & (m < 110), "h-sidebands": ((m >= 110) & (m < 115)) | ((m >= 135) & (m < 150)),
         "h-peak": (m >= 115) & (m < 135)}
    if sample in GROUPS["DY"]:
        gjj = df.gjj_mass.fillna(-1).to_numpy() if "gjj_mass" in df else np.full(len(df), -1.0)
        use = {"dyTo2Mu_M-50_aMCatNLO": {"z-peak": True, "h-sidebands": False, "h-peak": False},
               "dyTo2Mu_M-105To160": {"z-peak": False, "h-sidebands": gjj <= DY_GJJ_STITCH, "h-peak": gjj <= DY_GJJ_STITCH},
               "dy_VBF_filter": {"z-peak": False, "h-sidebands": gjj > DY_GJJ_STITCH, "h-peak": gjj > DY_GJJ_STITCH}}[sample]
        R = {k: v & use[k] for k, v in R.items()}
    return R


def closure_lead(sample, df, mc):
    """'leadpt' must reproduce the stage-1 jet1/jj_mass and the applyRegionCatCuts VBF flag."""
    has = mc.has.to_numpy()
    for mine, ref in (("j1_pt", "jet1_pt_nominal"), ("mjj", "jj_mass_nominal")):
        if ref in df:
            bad = ~np.isclose(mc[mine].to_numpy()[has], df[ref].to_numpy(dtype=float)[has], rtol=1e-5, atol=1e-3)
            if bad.any():
                raise AssertionError(f"[closure] {sample}: leadpt {mine} != {ref} for {bad.sum()} events")
    nj = df.njets_nominal.fillna(0)
    bveto = ~((df.nBtagLoose_nominal >= 2) | ((df.nBtagMedium_nominal >= 1) & (nj >= 2)))
    deta_r = np.rint(df.jj_dEta_nominal * 1e4) / 1e4
    ref_vbf = (bveto & (df.jj_mass_nominal > 400) & (deta_r > 2.5) & (df.jet1_pt_nominal > 35)).fillna(False)
    nbad = int((ref_vbf.to_numpy() != mc.vbf.to_numpy()).sum())
    if nbad:
        raise AssertionError(f"[closure] {sample}: leadpt VBF flag differs from stage-1 cut for {nbad} events")
    print(f"[closure] {sample}: leadpt == stage-1 jet1/jj_mass/VBF flag ({len(df)} events)")


# ----------------------------------------------------------------------------- truth
def truth(df, src="lhe"):
    """Tagging quarks = first two outgoing quarks of truth `src` (pT order); their matched selected-jet indices."""
    L = {f: np.stack([df[f"vbf_{src}{i}_{f}"].to_numpy(dtype=float) for i in range(1, NL + 1)], 1) for f in LHE_F}
    isq = np.abs(L["pdgId"]) <= 5  # NaN (missing parton) -> False
    nq = isq.sum(1)
    order = np.argsort(~isq, axis=1, kind="stable")[:, :2]
    r = np.arange(len(df))[:, None]
    T = {"qq": nq == 2, "nq": nq}
    q = {f: L[f][r, order] for f in LHE_F}
    T["q_eta"] = q["eta"]
    px, py, pz = q["pt"] * np.cos(q["phi"]), q["pt"] * np.sin(q["phi"]), q["pt"] * np.sinh(q["eta"])
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    T["mqq"] = np.sqrt(np.clip(e.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2, 0, None))
    for chain in ("gen", "reco"):
        m = np.nan_to_num(q[f"jetidx_{chain}"], nan=-1).astype(int)
        m[(m[:, 0] == m[:, 1]) & (m[:, 1] >= 0), 1] = -1  # both quarks on one jet: count it once
        T[f"match_{chain}"] = m
    return T


def truth_bins(df, T):
    nj = df.njets_nominal.fillna(0).to_numpy()
    npv = df.PV_npvsGood.to_numpy() if "PV_npvsGood" in df else np.zeros(len(df))
    fwd = np.max(np.abs(T["q_eta"]), axis=1)
    bins = {"inclusive": np.ones(len(df), bool), "nj2": nj == 2, "nj3": nj == 3, "nj4p": nj >= 4,
            "qmaxeta_central": fwd <= 2.5, "qmaxeta_HE": (fwd > 2.5) & (fwd <= 3.0), "qmaxeta_HF": fwd > 3.0}
    for lo, hi in ((0, 30), (30, 45), (45, 60), (60, 200)):
        bins[f"npv_{lo}_{hi}"] = (npv >= lo) & (npv < hi)
    for lo, hi in ((0, 400), (400, 800), (800, 1500), (1500, 99999)):
        bins[f"mqq_{lo}_{hi}"] = (T["mqq"] >= lo) & (T["mqq"] < hi)
    return bins


def purity_rows(sample, df, T, mcols, bins, src="lhe"):
    rows = []
    w = df.wgt_nominal.fillna(0).to_numpy()
    for chain in ("gen", "reco"):
        m = T[f"match_{chain}"]
        ceiling = (m[:, 0] >= 0) & (m[:, 1] >= 0)
        for meth, mc in mcols.items():
            ia, ib = mc.ia.to_numpy(), mc.ib.to_numpy()
            nmatch = sum(((m[:, k] >= 0) & ((m[:, k] == ia) | (m[:, k] == ib))).astype(int) for k in range(2))
            common = mc.bveto_nj2.to_numpy() & T["qq"]  # method-independent denominator
            for sel_name, sel in (("bveto_nj2", mc.bveto_nj2.to_numpy()), ("vbfcat", mc.vbf.to_numpy())):
                base = sel & T["qq"]
                for bname, bmask in bins.items():
                    b = base & bmask
                    W, Wc = w[b].sum(), w[common & bmask].sum()
                    if W <= 0 or Wc <= 0:
                        continue
                    rows.append(dict(sample=sample, truth=src, chain=chain, method=meth, selection=sel_name, bin=bname,
                                     n_raw=int(b.sum()), n_eff=W ** 2 / (w[b] ** 2).sum(),
                                     purity=w[b & (nmatch == 2)].sum() / W, partial=w[b & (nmatch == 1)].sum() / W,
                                     none=w[b & (nmatch == 0)].sum() / W, ceiling=w[b & ceiling].sum() / W,
                                     sel_eff=W / Wc, correct_eff=w[b & (nmatch == 2)].sum() / Wc))
    return rows


# ----------------------------------------------------------------------------- steps
def build_cache(inp, cache, samples, data_eras):
    os.makedirs(cache, exist_ok=True)
    for s in samples:
        out = f"{cache}/{s}.parquet"
        if os.path.exists(out):
            continue
        df = read_sample(inp, s)
        if df is None:
            print(f"[cache] {s}: no files, skipped")
            continue
        df.to_parquet(out)
        print(f"[cache] {s}: {len(df)} events")
    if not os.path.exists(f"{cache}/data.parquet"):
        eras = [d for d in (read_sample(inp, f"data_{e}") for e in data_eras) if d is not None]
        if eras:
            pd.concat(eras, ignore_index=True).to_parquet(f"{cache}/data.parquet")


def load(cache, s):
    p = f"{cache}/{s}.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None


def step_truth(cache, outdir):
    rows, mult = [], []
    for s in TRUTH_SAMPLES:
        df = load(cache, s)
        if df is None:
            continue
        mcols = {m: method_columns(df, m) for m in METHODS if f"vbf_{m}_mjj_nominal" in df}
        w = df.wgt_nominal.fillna(0).to_numpy()
        for src in TRUTH_SOURCES:
            if f"vbf_{src}1_pdgId" not in df or df[f"vbf_{src}1_pdgId"].isna().all():
                continue  # e.g. genq for Herwig (no hard-process flags)
            T = truth(df, src)
            rows += purity_rows(s, df, T, mcols, truth_bins(df, T), src)
            mult.append(dict(sample=s, truth=src, frac_qq=w[T["qq"]].sum() / w.sum(), frac_3q=w[T["nq"] == 3].sum() / w.sum()))
        print(f"[truth] {s} done")
    t = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    t.to_csv(f"{outdir}/truth_purity.csv", index=False)
    pd.DataFrame(mult).to_csv(f"{outdir}/truth_quark_multiplicity.csv", index=False)
    if len(t):
        v = t[(t.chain == "gen") & (t.bin == "inclusive")]
        for col in ("purity", "correct_eff", "sel_eff"):
            print(f"--- {col} (GenJet chain, inclusive)")
            print(v.pivot_table(index="method", columns=["truth", "sample", "selection"], values=col).round(3).to_string())
        print(v[v.method == "leadpt"][["truth", "sample", "selection", "ceiling"]].to_string(index=False))
    print(pd.DataFrame(mult).to_string(index=False))


def step_yields(cache, outdir, lumi_scale):
    rows = []
    for g, ss in {**GROUPS, **SIG}.items():
        for s in ss:
            df = load(cache, s)
            if df is None:
                continue
            R, w = regions(df, s), df.wgt_nominal.fillna(0).to_numpy()
            closure_lead(s, df, method_columns(df, "leadpt"))
            for m in METHODS:
                mc = method_columns(df, m)
                for cat in ("vbf", "ggh"):
                    c = mc[cat].to_numpy()
                    for reg in ("h-peak", "h-sidebands", "z-peak"):
                        sel = c & R[reg]
                        W = w[sel].sum()
                        rows.append(dict(group=g, sample=s, method=m, category=cat, region=reg, yield_full_lumi=W,
                                         sumw2=(w[sel] ** 2).sum(), n_raw=int(sel.sum()),
                                         pu_jet_frac=w[sel & mc.pu.to_numpy()].sum() / W if "pu" in mc and W else np.nan))
    data = load(cache, "data")
    if data is not None:
        R = regions(data)
        closure_lead("data", data, method_columns(data, "leadpt"))
        for m in METHODS:
            mc = method_columns(data, m)
            for cat in ("vbf", "ggh"):
                for reg in ("h-sidebands", "z-peak"):
                    n = float((mc[cat].to_numpy() & R[reg]).sum())
                    rows.append(dict(group="Data", sample="data", method=m, category=cat, region=reg,
                                     yield_full_lumi=n, sumw2=n, n_raw=int(n)))
    y = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    y.to_csv(f"{outdir}/yields_per_sample.csv", index=False)
    keys = ["method", "category", "region", "group"]
    g = y.groupby(keys).yield_full_lumi.sum().unstack("group")
    g2 = y.groupby(keys).sumw2.sum().unstack("group")
    bkg = [c for c in GROUPS if c in g]
    g["B"] = g[bkg].sum(axis=1)
    g["B_err"] = np.sqrt(g2[bkg].sum(axis=1))
    if "VBF" in g:
        g["VBF_err"] = np.sqrt(g2.VBF)
        g["S_over_sqrtB"] = g.VBF / np.sqrt(g.B)
        # MC-stat error: dZ^2 = (dS/sqrtB)^2 + (S dB / 2B^1.5)^2
        g["S_over_sqrtB_err"] = np.sqrt((g.VBF_err / np.sqrt(g.B)) ** 2 + (g.VBF * g.B_err / (2 * g.B ** 1.5)) ** 2)
        if "ggH" in g:
            g["ggH_frac"] = g.ggH / (g.ggH + g.VBF)
    if "Data" in g:  # MC normalised to full 2024 lumi; data = chosen eras
        g["Data_over_MC"] = g.Data / (lumi_scale * (g.B + g.get("VBF", 0) + g.get("ggH", 0) + g.get("VH", 0)))
    g.to_csv(f"{outdir}/yields_summary.csv")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
        print(g.xs("h-peak", level="region").to_string())
    return g


def step_plots(cache, outroot, lumi_scale, lumi_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    mcs = {g: [s for s in ss if os.path.exists(f"{cache}/{s}.parquet")] for g, ss in GROUPS.items()}
    mcs = {g: ss for g, ss in mcs.items() if ss}
    if not mcs:
        print("[plots] no background samples in cache, skipping Data/MC plots")
        return
    sig = {k: ss[0] for k, ss in SIG.items() if k in ("VBF", "ggH") and os.path.exists(f"{cache}/{ss[0]}.parquet")}
    V = {"j1_pt": (np.array([25, 35, 50, 75, 100, 150, 200, 300, 500]), "jet1 $p_T$ [GeV]"),
         "j2_pt": (np.array([25, 35, 50, 75, 100, 150, 250]), "jet2 $p_T$ [GeV]"),
         "j1_eta": (np.linspace(-4.7, 4.7, 48), "jet1 $\\eta$"), "j2_eta": (np.linspace(-4.7, 4.7, 48), "jet2 $\\eta$"),
         "mjj": (np.array([0, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 3000, 4500]), "$m_{jj}$ [GeV]"),
         "deta": (np.linspace(0, 9, 37), "$|\\Delta\\eta_{jj}|$")}
    frames = {}
    for name in ["data"] + sum(mcs.values(), []) + list(sig.values()):
        df = load(cache, name)
        if df is not None:
            frames[name] = (df, {m: method_columns(df, m) for m in METHODS}, regions(df, name))
    summary = []
    for m in METHODS:
        for sel in ("bveto_nj2", "vbf"):
            for reg in ("z-peak", "h-sidebands"):
                odir = f"{outroot}/{m}/{sel}_{reg}"
                os.makedirs(odir, exist_ok=True)
                for var, (bins, xl) in V.items():
                    bw = np.diff(bins)

                    def h(name, scale=1.0):
                        df, mc, R = frames[name]
                        msk = mc[m][sel].to_numpy() & R[reg]
                        x = mc[m][var].to_numpy()[msk]
                        ww = df.wgt_nominal.fillna(0).to_numpy()[msk] * scale
                        return np.histogram(np.clip(x, bins[0], bins[-1] - 1e-6), bins, weights=ww)[0]

                    hd = h("data") if "data" in frames else np.zeros(len(bw))
                    stack = [np.sum([h(s, lumi_scale) for s in ss], axis=0) for ss in mcs.values()]
                    tot = np.sum(stack, axis=0)
                    summary.append(dict(method=m, selection=sel, region=reg, var=var, data=hd.sum(), mc=tot.sum()))
                    fig, (ax, rx) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
                    hep.histplot([s / bw for s in stack], bins, stack=True, histtype="fill", label=list(mcs), ax=ax)
                    hep.histplot(hd / bw, bins, yerr=np.sqrt(hd) / bw, histtype="errorbar", color="k", label="Data", ax=ax)
                    for (k, sname), col in zip(sig.items(), ["red", "black"]):
                        hep.histplot(h(sname, lumi_scale) * 100 / bw, bins, label=f"{k} x100", color=col, ls="--", lw=2, ax=ax)
                    ax.set_yscale("log")
                    ax.set_ylabel("Events / bin width")
                    ax.legend(ncol=2, fontsize=12, loc="upper right")
                    hep.cms.label("Preliminary", data=True, lumi=lumi_label, year="2024", ax=ax, fontsize=16)
                    ax.text(0.03, 0.04, f"pair: {m} | {sel} | {reg}", transform=ax.transAxes, fontsize=13)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rx.errorbar(0.5 * (bins[1:] + bins[:-1]), hd / tot, yerr=np.sqrt(hd) / tot, fmt="ko")
                    rx.axhline(1, color="gray", ls="--")
                    rx.set_ylim(0.4, 1.6)
                    rx.set_ylabel("Data/MC")
                    rx.set_xlabel(xl)
                    for ext in ("png", "pdf"):
                        fig.savefig(f"{odir}/{var}.{ext}", bbox_inches="tight")
                    plt.close(fig)
        print(f"[plots] {m} done")
    pd.DataFrame(summary).to_csv(f"{outroot}/datamc_summary.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="<stage1_output>/<year> directory")
    p.add_argument("--label", required=True)
    p.add_argument("--cache-root", default="/work/projects/hmm/shar1172/vbf_jet_selection/pairstudy_cache")
    p.add_argument("--data-eras", nargs="*", default=["C", "D"])
    p.add_argument("--steps", default="cache,truth,yields,plots")
    a = p.parse_args()
    import yaml
    ycfg = yaml.safe_load(open("configs/datasets/dataset_nanoAODv15_run3.yaml"))["years"]["2024"]["Data"]
    lumi = sum(ycfg[f"data_{e}"]["lumi_pb"] for e in a.data_eras)
    lumi_scale = lumi / ycfg["total_lumi_pb"]
    cache = f"{a.cache_root}/{a.label}"
    outroot = f"validation/vbf_jet_selection/{a.label}"
    steps = a.steps.split(",")
    samples = sorted(set(sum(GROUPS.values(), []) + sum(SIG.values(), []) + TRUTH_SAMPLES + ["ggh_amcatnlo"]))
    if "cache" in steps:
        build_cache(a.input, cache, samples, a.data_eras)
    print(f"data eras {a.data_eras}: {lumi / 1000:.2f} fb-1, MC scale {lumi_scale:.4f}")
    if "truth" in steps:
        step_truth(cache, outroot)
    if "yields" in steps:
        step_yields(cache, outroot, lumi_scale)
    if "plots" in steps:
        step_plots(cache, outroot, lumi_scale, round(lumi / 1000, 2))
