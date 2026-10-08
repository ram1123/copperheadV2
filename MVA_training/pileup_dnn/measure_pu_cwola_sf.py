#!/usr/bin/env python3
"""
Data/MC scale factors for the CWoLa PU/HS forward-jet tagger (train_pu_cwola.py), measured
in the same Z(mumu) + 1-jet selection, binned in jet pT and |eta|.

Two measurements per (region, pT, |eta|) bin:
  1. Raw tagger efficiency in the HS-enriched and PU-enriched mixtures, data vs MC
     (model-free, but each mixture is a blend of HS and PU jets).
  2. Per-class efficiencies and yields from a binned Poisson fit of the dphi(Z, jet)
     distribution with MC gen-matched (HS) and gen-unmatched (PU) templates, done
     separately for jets passing and failing the tagger WP. The tagger never sees dphi,
     so the fit variable is independent of the cut being measured.
     Outputs:
       tagger SFs     sf_{hs,pu}_{pass,fail} = eff_data / eff_MC (fail: (1-eff) ratio)
       PU-rate SF     pu_rate_sf_rel = (N_PU data/MC) / (N_HS data/MC)
     pu_rate_sf_rel is the per-jet weight for gen-unmatched MC jets when no tagger cut is
     applied (VBF acceptance recovery); it is relative to the HS yield so an overall MC
     normalization offset cancels.

Data events used in training (event % 10 >= 4) are excluded by default.

Example command:
--------------------
B=/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJets_Sep22_tightPassLepVeto_DefaultjetPt25GeV/stage1_output/2024/compacted
python MVA_training/pileup_dnn/measure_pu_cwola_sf.py \
  --model-dir validation/pu_cwola/run2024_baseline \
  --data "$B/data_*/*/*.parquet" \
  --mc "$B/dyTo2Mu_M-50_aMCatNLO/*/*.parquet" "$B/ttjets_*/*/*.parquet" \
       "$B/ww_*/*/*.parquet" "$B/wz_*/*/*.parquet" "$B/zz_*/*/*.parquet" \
  -o validation/pu_cwola/run2024_baseline/sf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
import train_pu_cwola as cw  # noqa: E402
import train_pu_dnn as base  # noqa: E402
from modules.git_utils import get_git_commit, get_git_state  # noqa: E402

DEFAULT_PT_EDGES = [25.0, 30.0, 35.0, 40.0, 50.0]
DEFAULT_ETA_EDGES = {
    "HE": [2.5, 2.65, 2.853, 3.0],
    "HF": [3.0, 3.314, 3.839, 4.363, 5.191],
}
HELDOUT_FRACTION = 0.4  # event % 10 < 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure data/MC SFs for the CWoLa PU/HS tagger in Z+1jet.")
    p.add_argument("--model-dir", required=True, help="train_pu_cwola.py output directory.")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--mc", nargs="+", required=True, help="All MC contributing to Z+1jet (DY, top, diboson).")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--regions", nargs="+", default=None, help="Default: regions found in --model-dir.")
    p.add_argument("--pt-edges", nargs="+", type=float, default=DEFAULT_PT_EDGES)
    p.add_argument("--dphi-bins", type=int, default=16)
    p.add_argument("--all-data-events", action="store_true", help="Also use data events seen in training.")
    p.add_argument("--min-data-jets", type=int, default=100, help="Skip fit in bins with fewer data jets.")
    p.add_argument("--n-toys", type=int, default=2000)
    p.add_argument("--max-files-data", type=int, default=None)
    p.add_argument("--max-files-mc", type=int, default=None)
    p.add_argument("--plot-format", nargs="+", default=["png", "pdf"])
    p.add_argument("--plot-all-fits", action="store_true", help="Also draw fits for every (pT, |eta|) bin.")
    return p.parse_args()


# ------------------------------------------------------------------ tagger
def load_tagger(model_dir: Path, region: str):
    rdir = model_dir / region
    summary = json.loads((rdir / f"summary_{region}.json").read_text())
    scaler = base.Scaler(**json.loads((rdir / "scaler.json").read_text()))
    model = torch.jit.load(str(rdir / "model_torchscript.pt"), map_location="cpu").eval()
    return model, scaler, summary


def tag(df: pd.DataFrame, model, scaler: base.Scaler, summary: dict) -> tuple[np.ndarray, np.ndarray]:
    if len(df) == 0:
        return np.zeros(0, np.float32), np.zeros(0, bool)
    with torch.no_grad():
        s = torch.sigmoid(model(torch.from_numpy(scaler.transform(df)))).numpy()
    return s, base.pass_from_threshold(s, summary["threshold"], summary["direction"])


# ------------------------------------------------------------------ fit
def fit_two_templates(data_h: np.ndarray, t_hs: np.ndarray, t_pu: np.ndarray) -> dict:
    """Binned Poisson fit data = mu_hs * t_hs + mu_pu * t_pu; analytic Hessian for the covariance."""
    t_hs = np.clip(t_hs, 1e-9, None)
    t_pu = np.clip(t_pu, 1e-9, None)

    def nll(mu):
        pred = np.clip(mu[0] * t_hs + mu[1] * t_pu, 1e-12, None)
        return float(np.sum(pred - data_h * np.log(pred)))

    res = minimize(nll, x0=[1.0, 1.0], method="L-BFGS-B", bounds=[(0.0, 50.0), (0.0, 50.0)])
    mu = res.x
    pred = np.clip(mu[0] * t_hs + mu[1] * t_pu, 1e-12, None)
    tt = np.stack([t_hs, t_pu])
    hess = (tt[:, None, :] * tt[None, :, :] * (data_h / pred**2)).sum(axis=-1)
    cov = np.linalg.pinv(hess)
    nz = data_h > 0
    chi2 = float(np.sum((data_h[nz] - pred[nz]) ** 2 / data_h[nz]))
    return {"mu": mu, "cov": cov, "chi2": chi2, "ndf": int(nz.sum()) - 2, "ok": bool(res.success), "pred": pred}


def derived_quantities(mu_p: np.ndarray, mu_f: np.ndarray, mc: dict) -> dict:
    """mu_* shape (..., 2) = (mu_hs, mu_pu) for pass/fail fits; mc = template integrals."""
    n = {
        "hs_pass": mu_p[..., 0] * mc["hs_pass"], "pu_pass": mu_p[..., 1] * mc["pu_pass"],
        "hs_fail": mu_f[..., 0] * mc["hs_fail"], "pu_fail": mu_f[..., 1] * mc["pu_fail"],
    }
    out = {}
    for cls in ("hs", "pu"):
        tot_d = n[f"{cls}_pass"] + n[f"{cls}_fail"]
        tot_m = mc[f"{cls}_pass"] + mc[f"{cls}_fail"]
        e_d = np.divide(n[f"{cls}_pass"], tot_d, out=np.full_like(tot_d, np.nan), where=tot_d > 0)
        e_m = mc[f"{cls}_pass"] / tot_m if tot_m > 0 else np.nan
        out[f"eff_{cls}_data"] = e_d
        out[f"eff_{cls}_mc"] = np.full_like(e_d, e_m)
        out[f"sf_{cls}_pass"] = e_d / e_m
        out[f"sf_{cls}_fail"] = (1 - e_d) / (1 - e_m)
        out[f"{cls}_yield_sf"] = tot_d / tot_m if tot_m > 0 else np.full_like(tot_d, np.nan)
    out["pu_rate_sf_rel"] = out["pu_yield_sf"] / out["hs_yield_sf"]
    return out


def with_toys(fp: dict, ff: dict, mc: dict, n_toys: int, rng: np.random.Generator) -> dict:
    central = derived_quantities(fp["mu"], ff["mu"], mc)
    tp = np.clip(rng.multivariate_normal(fp["mu"], fp["cov"], size=n_toys), 0, None)
    tf = np.clip(rng.multivariate_normal(ff["mu"], ff["cov"], size=n_toys), 0, None)
    toys = derived_quantities(tp, tf, mc)
    out = {}
    for k, v in central.items():
        out[k] = float(v)
        out[f"{k}_err"] = float(np.nanstd(toys[k]))
    return out


# ------------------------------------------------------------------ per-bin measurement
def binomial_eff(passed: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    sw = w.sum()
    if sw <= 0:
        return float("nan"), float("nan")
    e = w[passed].sum() / sw
    # weighted binomial variance (effective-entries approximation)
    var = np.sum(w**2 * (passed.astype(float) - e) ** 2) / sw**2
    return float(e), float(np.sqrt(max(var, 0)))


def measure_bin(d: pd.DataFrame, m: pd.DataFrame, dphi_edges: np.ndarray, data_scale: float,
                args, rng) -> tuple[dict, dict | None]:
    row = {"n_data": int(len(d)), "sumw_mc": float(m["w_phys"].sum())}
    for name, code in (("hsmix", cw.MIX_HS), ("pumix", cw.MIX_PU)):
        dd, mm = d[d["mix"] == code], m[m["mix"] == code]
        e_d, err_d = binomial_eff(dd["pass"].to_numpy(), np.ones(len(dd)))
        e_m, err_m = binomial_eff(mm["pass"].to_numpy(), mm["w_phys"].to_numpy(np.float64))
        row[f"raw_eff_{name}_data"], row[f"raw_eff_{name}_data_err"] = e_d, err_d
        row[f"raw_eff_{name}_mc"], row[f"raw_eff_{name}_mc_err"] = e_m, err_m
        row[f"raw_sf_{name}"] = e_d / e_m if e_m else float("nan")
        row[f"raw_sf_{name}_err"] = row[f"raw_sf_{name}"] * np.hypot(err_d / e_d, err_m / e_m) if e_d and e_m else float("nan")
    if len(d) < args.min_data_jets:
        row["fit_status"] = "skipped_low_stats"
        return row, None

    hist = lambda df, w=None: np.histogram(df["dphiZJet"], bins=dphi_edges, weights=w)[0].astype(np.float64)  # noqa: E731
    fits, mc_int = {}, {}
    for state, pmask_d, pmask_m in (("pass", d["pass"], m["pass"]), ("fail", ~d["pass"], ~m["pass"])):
        dd, mm = d[pmask_d.to_numpy()], m[pmask_m.to_numpy()]
        g = mm["y_gen"].to_numpy() == 1
        w = mm["w_phys"].to_numpy(np.float64) * data_scale
        t_hs, t_pu = hist(mm[g], w[g]), hist(mm[~g], w[~g])
        mc_int[f"hs_{state}"], mc_int[f"pu_{state}"] = float(t_hs.sum()), float(t_pu.sum())
        fits[state] = fit_two_templates(hist(dd), t_hs, t_pu)
        fits[state].update({"data": hist(dd), "t_hs": t_hs, "t_pu": t_pu})
        row[f"fit_{state}_mu_hs"], row[f"fit_{state}_mu_pu"] = map(float, fits[state]["mu"])
        row[f"fit_{state}_chi2_ndf"] = fits[state]["chi2"] / max(fits[state]["ndf"], 1)
        row[f"fit_{state}_ok"] = fits[state]["ok"]
    row.update(with_toys(fits["pass"], fits["fail"], mc_int, args.n_toys, rng))
    row["fit_status"] = "ok" if fits["pass"]["ok"] and fits["fail"]["ok"] else "fit_warning"
    return row, fits


def plot_fit(fits: dict, dphi_edges: np.ndarray, title: str, outbase: Path, formats: list[str]) -> None:
    c = 0.5 * (dphi_edges[1:] + dphi_edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, state in zip(axes, ("pass", "fail")):
        f = fits[state]
        hs, pu = f["mu"][0] * f["t_hs"], f["mu"][1] * f["t_pu"]
        ax.hist([c, c], bins=dphi_edges, weights=[pu, hs], stacked=True, color=["#d62728", "#1f77b4"], alpha=0.6,
                label=[f"PU (mu={f['mu'][1]:.2f})", f"HS (mu={f['mu'][0]:.2f})"])
        ax.errorbar(c, f["data"], yerr=np.sqrt(f["data"]), fmt="ko", ms=3, label="Data")
        ax.set_xlabel(r"$|\Delta\phi(Z, jet)|$")
        ax.set_ylabel("Jets")
        ax.set_title(f"{title} | tagger {state} | chi2/ndf={f['chi2'] / max(f['ndf'], 1):.2f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    base.save_plot(fig, outbase, formats)


def plot_sf_vs_pt(df: pd.DataFrame, region: str, outbase: Path, formats: list[str]) -> None:
    rows = df[(df["level"] == "pt_eta") & (df["fit_status"] != "skipped_low_stats")]
    qty = [("pu_rate_sf_rel", "PU-jet rate SF (rel. to HS)"), ("sf_pu_pass", "Tagger SF: PU jets passing"),
           ("sf_hs_pass", "Tagger SF: HS jets passing"), ("raw_sf_pumix", "Raw eff SF: PU-enriched mixture")]
    fig, axes = plt.subplots(1, len(qty), figsize=(4.4 * len(qty), 4.0))
    for ax, (col, label) in zip(axes, qty):
        for (lo, hi), grp in rows.groupby(["eta_low", "eta_high"]):
            x = 0.5 * (grp["pt_low"] + grp["pt_high"])
            ax.errorbar(x, grp[col], yerr=grp.get(f"{col}_err"), xerr=0.5 * (grp["pt_high"] - grp["pt_low"]),
                        fmt="o", ms=4, capsize=2, label=f"{lo:.2f}<|eta|<{hi:.2f}")
        ax.axhline(1, color="gray", lw=1)
        ax.set_xlabel("jet pT [GeV]")
        ax.set_title(f"{region}: {label}", fontsize=9)
        ax.set_ylim(0, 2)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    base.save_plot(fig, outbase, formats)


def region_rows(region: str, data: pd.DataFrame, mc: pd.DataFrame, args, out: Path, rng) -> pd.DataFrame:
    pt_edges = np.asarray(args.pt_edges)
    e_edges = np.asarray(DEFAULT_ETA_EDGES["HE" if region.startswith("HE") else "HF"])
    dphi_edges = np.linspace(0, np.pi, args.dphi_bins + 1)
    data_scale = 1.0 if args.all_data_events else HELDOUT_FRACTION
    fitdir = out / "fits"
    fitdir.mkdir(parents=True, exist_ok=True)

    bins = [("region", pt_edges[0], pt_edges[-1], e_edges[0], e_edges[-1])]
    bins += [("eta", pt_edges[0], pt_edges[-1], lo, hi) for lo, hi in zip(e_edges[:-1], e_edges[1:])]
    bins += [("pt_eta", pl, ph, lo, hi) for lo, hi in zip(e_edges[:-1], e_edges[1:])
             for pl, ph in zip(pt_edges[:-1], pt_edges[1:])]
    rows = []
    for level, pl, ph, el, eh in bins:
        sel_d = data["pt"].between(pl, ph, inclusive="left") & data["aeta"].between(el, eh, inclusive="left")
        sel_m = mc["pt"].between(pl, ph, inclusive="left") & mc["aeta"].between(el, eh, inclusive="left")
        row, fits = measure_bin(data[sel_d], mc[sel_m], dphi_edges, data_scale, args, rng)
        row.update({"region": region, "level": level, "pt_low": pl, "pt_high": ph, "eta_low": el, "eta_high": eh})
        rows.append(row)
        if fits is not None and (level != "pt_eta" or args.plot_all_fits):
            tag_ = f"{region}_pt{pl:g}-{ph:g}_eta{el:g}-{eh:g}"
            # no dots in the stem: save_plot uses with_suffix()
            plot_fit(fits, dphi_edges, tag_, fitdir / f"dphi_fit_{tag_.replace('.', 'p')}", args.plot_format)
    return pd.DataFrame(rows)


def lookup_tables(df: pd.DataFrame, pt_edges: list[float]) -> dict:
    """Per-region (pT x |eta|) arrays for per-jet weights; NaN where the fit was skipped."""
    tables = {}
    for region, grp in df[df["level"] == "pt_eta"].groupby("region"):
        e_edges = sorted(set(grp["eta_low"]) | set(grp["eta_high"]))
        entry = {"pt_edges": list(pt_edges), "abseta_edges": e_edges}
        for col in ("pu_rate_sf_rel", "sf_hs_pass", "sf_hs_fail", "sf_pu_pass", "sf_pu_fail"):
            for suffix in ("", "_err"):
                name = col + suffix
                arr = np.full((len(e_edges) - 1, len(pt_edges) - 1), np.nan)
                if name in grp.columns:
                    for _, r in grp.iterrows():
                        arr[e_edges.index(r["eta_low"]), pt_edges.index(r["pt_low"])] = r[name]
                entry[name] = [[None if not np.isfinite(v) else float(v) for v in rowv] for rowv in arr]
        tables[region] = entry
    return tables


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "git_state.json").write_text(json.dumps({"git_commit": get_git_commit(), "git_state": get_git_state(out)}, indent=2))
    model_dir = Path(args.model_dir)
    sel = json.loads((model_dir / "selection.json").read_text())
    regions = args.regions or [p.name for p in sorted(model_dir.iterdir()) if (p / "model_torchscript.pt").exists()]
    rng = np.random.default_rng(12345)

    data, data_paths = cw.load_zjet(args.data, sel, args.max_files_data)
    mc, mc_paths = cw.load_zjet(args.mc, sel, args.max_files_mc)
    if not args.all_data_events:
        data = data[data["is_heldout"]].copy()
    print(f"Z+1jet forward jets: data={len(data)} ({'all' if args.all_data_events else 'held-out'}), MC={len(mc)}")
    (out / "inputs.json").write_text(json.dumps({
        "model_dir": str(model_dir), "selection": sel, "data": args.data, "mc": args.mc,
        "n_data_files": len(data_paths), "n_mc_files": len(mc_paths), "all_data_events": args.all_data_events,
    }, indent=2))

    frames = []
    for region in regions:
        model, scaler, summary = load_tagger(model_dir, region)
        d = data[cw.region_mask(data, region)].copy()
        m = mc[cw.region_mask(mc, region)].copy()
        d["score"], d["pass"] = tag(d, model, scaler, summary)
        m["score"], m["pass"] = tag(m, model, scaler, summary)
        df = region_rows(region, d, m, args, out, rng)
        df.to_csv(out / f"sf_{region}.csv", index=False)
        plot_sf_vs_pt(df, region, out / f"sf_vs_pt_{region}", args.plot_format)
        frames.append(df)
        print(df[df["level"] != "pt_eta"][["region", "level", "eta_low", "eta_high", "n_data", "raw_sf_hsmix",
                                            "raw_sf_pumix", "pu_rate_sf_rel", "pu_rate_sf_rel_err",
                                            "sf_pu_pass", "sf_hs_pass"]].to_string(index=False))

    allsf = pd.concat(frames, ignore_index=True)
    allsf.to_csv(out / "sf_all.csv", index=False)
    (out / "pu_cwola_sf.json").write_text(json.dumps({
        "description": (
            "Z+1jet CWoLa tagger SFs. pu_rate_sf_rel: per-jet weight for gen-unmatched MC jets "
            "(no tagger cut). sf_{hs,pu}_{pass,fail}: per-jet weights when the tagger WP is applied, "
            "chosen by gen-match (hs/pu) and pass/fail. Arrays are [abseta_bin][pt_bin]."
        ),
        "model_dir": str(model_dir),
        "regions": lookup_tables(allsf, list(args.pt_edges)),
    }, indent=2))
    print(f"Done. Wrote SFs for {len(regions)} region(s) to {out}")


if __name__ == "__main__":
    main()
