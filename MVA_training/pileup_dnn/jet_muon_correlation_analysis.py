"""
Jet-Muon Correlation Analysis
==============================
Analyzes correlations between muons & real/fake jets, and
between real/fake jets in mixed-jet events.

Real jets  : hasMatchedGenJet > 0.5
Fake jets  : hasMatchedGenJet <= 0.5

Usage
-----
    python jet_muon_correlation_analysis.py --input /path/to/*.parquet [--output ./plots]
    python MVA_training/pileup_dnn/jet_muon_correlation_analysis.py \
        --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_June02_tightPassLepVeto_NoJER/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part00*.parquet \
        --output validation/muon_fakejet_corr_v4


Structure
---------
1. Muon ↔ Fake jets  (Δη, ΔΦ, ΔR, pT ratio)
   Muon ↔ Real jets
2. Mixed-jet events (has both real AND fake jets):
   Real  vs Real
   Fake  vs Fake
   Fake  vs Real
"""

import argparse
import glob
import itertools
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

# ── Optional coffea/awkward imports ────────────────────────────────────────────
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, BaseSchema
print(f"coffea {coffea.__version__} loaded")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ──────────────────────────────────────────────────────────────────────
GEN_JET_CUT = 0.5          # hasMatchedGenJet threshold: real if > cut

MU_VARS  = ["mu1_pt", "mu1_eta", "mu1_phi"]   # extend as needed
# Jet column bases — _nominal suffix resolved automatically at runtime
JET_VARS = ["jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal",
            "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal"]

# (mu_var_base, jet_var_base, latex_title)
CORR_PAIRS_MU_JET = [
    ("mu1_pt",  "jet1_pt_nominal",  r"$p_T^{\mu_1}$ vs $p_T^{j_1}$"),
    ("mu1_eta", "jet1_eta_nominal", r"$\eta^{\mu_1}$ vs $\eta^{j_1}$"),
    ("mu1_pt",  "jet2_pt_nominal",  r"$p_T^{\mu_1}$ vs $p_T^{j_2}$"),
]

# pT axis ranges applied to all 2D histograms and 1D pT plots
MU_PT_RANGE  = (0, 200)   # GeV
JET_PT_RANGE = (0, 200)   # GeV

PLOT_STYLE = dict(alpha=0.6, bins=50, histtype="step", linewidth=1.8)

# Jet composition / ID variables  (base name without jet index or _nominal suffix)
# Each entry: (base_name, x_axis_label, x_range, is_integer)
#   is_integer=True  -> use integer bins (1 bin per integer value)
#   is_integer=False -> use 60 uniform bins
JET_ID_VARS = [
    # Energy fractions — spike near 0, need log-y to see shape
    ("muEF",               "Muon Energy Fraction",               (0.0,  0.5),  False),
    ("chEmEF",             "Charged EM Energy Fraction",         (0.0,  0.6),  False),
    ("chHEF",              "Charged Hadron Energy Fraction",     (0.0,  1.0),  False),
    ("neEmEF",             "Neutral EM Energy Fraction",         (0.0,  1.0),  False),
    ("neHEF",              "Neutral Hadron Energy Fraction",     (0.0,  1.0),  False),
    # Constituent counts — integer-valued
    ("nConstituents",      "N constituents",                     (0,   60),    True),
    ("nElectrons",         "N electrons",                        (0,    6),    True),
    ("nMuons",             "N muons",                            (0,    6),    True),
    ("chMultiplicity",     "Charged multiplicity",               (0,   50),    True),
    ("neMultiplicity",     "Neutral multiplicity",               (0,   30),    True),
    # Muon subtraction
    ("muonSubtrFactor",    "Muon subtraction factor",            (0.0,  1.0),  False),
    ("muonSubtrDeltaEta",  r"$\Delta\eta$ (muon subtr.)",      (-0.3, 0.3),  False),
    ("muonSubtrDeltaPhi",  r"$\Delta\phi$ (muon subtr.)",      (-0.3, 0.3),  False),
]

COLORS = {
    "real": "#2196F3",
    "fake": "#F44336",
    "real_real": "#1565C0",
    "fake_fake": "#B71C1C",
    "fake_real": "#FF9800",
}


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_parquet(paths: list[str]) -> pd.DataFrame:
    """Load one or more parquet files into a single DataFrame."""
    frames = []
    for p in paths:
        tmp = pd.read_parquet(p)
        print(f"  {os.path.basename(p):40s}  {len(tmp):>8,} events")
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df):,} events from {len(paths)} file(s)")
    return df


def resolve_input_paths(raw_args: list[str]) -> list[str]:
    """
    Accept any mix of:
      - explicit file paths
      - shell-unexpanded globs  (quoted by user, e.g. '*.parquet')
      - already shell-expanded paths (no * present)
    Always re-globs every argument so quoted patterns work too.
    """
    files = []
    for pat in raw_args:
        if "*" in pat or "?" in pat:
            matched = sorted(glob.glob(pat, recursive=True))
            if not matched:
                print(f"  Warning: glob matched nothing: {pat}")
            files.extend(matched)
        else:
            if not os.path.isfile(pat):
                print(f"  Warning: file not found: {pat}")
            else:
                files.append(pat)
    files = sorted(set(files))
    if not files:
        sys.exit("No parquet files found. Check --input paths.")
    print(f"Found {len(files)} parquet file(s) to load.")
    return files


# ══════════════════════════════════════════════════════════════════════════════
# Classification helpers
# ══════════════════════════════════════════════════════════════════════════════

def classify_jets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean columns:
      jet1_real, jet1_fake, jet2_real, jet2_fake
    and an event-level flag:
      has_real, has_fake, mixed  (event has both real AND fake jets)
    """
    for idx in [1, 2, 3, 4]:
        # Try both naming conventions: with and without _nominal suffix
        col = next((c for c in [
            f"jet{idx}_hasMatchedGenJet_nominal",
            f"jet{idx}_hasMatchedGenJet",
        ] if c in df.columns), None)

        if col:
            df[f"jet{idx}_real"] = df[col] > GEN_JET_CUT
            df[f"jet{idx}_fake"] = df[col] <= GEN_JET_CUT
            # Treat NaN as fake (unmatched)
            nan_mask = df[col].isna()
            df.loc[nan_mask, f"jet{idx}_real"] = False
            df.loc[nan_mask, f"jet{idx}_fake"] = True
        else:
            if idx <= 2:
                print(f"  Warning: no hasMatchedGenJet column found for jet{idx}")
            df[f"jet{idx}_real"] = False
            df[f"jet{idx}_fake"] = False

    # Event has at least one real/fake jet (using jet1 as proxy if only one jet)
    df["has_real"] = df["jet1_real"]
    df["has_fake"] = df["jet1_fake"]

    # Mixed = event has BOTH a real AND a fake jet (check across jet1..jet4)
    real_cols = [c for c in [f"jet{i}_real" for i in range(1,5)] if c in df.columns]
    fake_cols = [c for c in [f"jet{i}_fake" for i in range(1,5)] if c in df.columns]
    if real_cols and fake_cols:
        any_real = df[real_cols].any(axis=1)
        any_fake = df[fake_cols].any(axis=1)
        df["mixed"] = any_real & any_fake
    else:
        df["mixed"] = False

    # ── Event-level jet composition (requires >=2 jets) ───────────────────────
    # has_2jets: jet2 exists with pt > 0
    j2_pt_col = next((c for c in ["jet2_pt_nominal", "jet2_pt"] if c in df.columns), None)
    if j2_pt_col:
        df["has_2jets"] = df[j2_pt_col].notna() & (df[j2_pt_col] > 0)
    else:
        df["has_2jets"] = False

    # Event composition based on jet1 AND jet2 status
    # both_real  : jet1 real  AND jet2 real
    # both_fake  : jet1 fake  AND jet2 fake
    # mixed_comp : one real, one fake (jet1 real+jet2 fake OR jet1 fake+jet2 real)
    if "jet2_real" in df.columns:
        df["evtcomp_both_real"]  = df["has_2jets"] & df["jet1_real"] & df["jet2_real"]
        df["evtcomp_both_fake"]  = df["has_2jets"] & df["jet1_fake"] & df["jet2_fake"]
        df["evtcomp_mixed"]      = df["has_2jets"] & (
            (df["jet1_real"] & df["jet2_fake"]) |
            (df["jet1_fake"] & df["jet2_real"])
        )
    else:
        for c in ["evtcomp_both_real","evtcomp_both_fake","evtcomp_mixed"]:
            df[c] = False

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Column-name resolution  (handles _nominal suffix transparently)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_col(df: pd.DataFrame, base: str) -> str | None:
    """
    Return the actual column name for `base`, trying:
      base                   (e.g. "jet1_eta")
      base + "_nominal"      (e.g. "jet1_eta_nominal")
    Returns None if neither is present.
    """
    for candidate in [base, base + "_nominal"]:
        if candidate in df.columns:
            return candidate
    return None


def get(df: pd.DataFrame, base: str) -> np.ndarray | None:
    """Return column values (float64) for base name (with or without _nominal)."""
    col = resolve_col(df, base)
    return df[col].values.astype(float) if col is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Angular / kinematic helpers
# ══════════════════════════════════════════════════════════════════════════════

def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    dphi = phi1 - phi2
    return np.where(dphi > np.pi, dphi - 2*np.pi,
           np.where(dphi < -np.pi, dphi + 2*np.pi, dphi))


def delta_r(eta1, phi1, eta2, phi2) -> np.ndarray:
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# ══════════════════════════════════════════════════════════════════════════════
# Correlation metrics
# ══════════════════════════════════════════════════════════════════════════════

def correlation_summary(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute Pearson + Spearman correlation for a pair of arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {}
    pearson_r, pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {
        "label": label,
        "n": len(x),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": f"{pearson_p:.2e}",
        "spearman_r": round(spearman_r, 4),
        "spearman_p": f"{spearman_p:.2e}",
    }



def save_fig(fig, outdir: str, stem: str):
    """Save figure as both PNG (screen) and PDF (report-ready, vector)."""
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    **({} if ext == "png" else {"backend": "pdf"}))
    print(f"  Saved {stem}.png / .pdf")


def save_panel(ax, outdir: str, stem: str):
    """
    Extract a single Axes panel into its own figure and save as PDF only.
    The combined PNG from save_fig() is kept as the overview; individual
    PDFs are for dropping straight into a report.
    """
    import io
    # Grab the bounding box of this axes in figure-fraction coords
    fig_src = ax.get_figure()
    renderer = fig_src.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig_src.transFigure.inverted())

    fig_out, ax_out = plt.subplots(figsize=(6, 5))
    # Copy via a tight savefig of just this axes
    # We use the bbox_inches trick to clip to this axes
    path = os.path.join(outdir, f"{stem}.pdf")
    fig_src.savefig(path, bbox_inches=ax.get_tightbbox(renderer)
                    .transformed(fig_src.dpi_scale_trans.inverted()),
                    backend="pdf")
    plt.close(fig_out)
    print(f"  Saved {stem}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Muon ↔ Jet correlations
# ══════════════════════════════════════════════════════════════════════════════

def plot_mu_jet_correlations(df: pd.DataFrame, outdir: str) -> list[dict]:
    """
    For each kinematic variable pair, overlay distributions for:
      • muon ↔ real jets
      • muon ↔ fake jets
    Plus ΔR distributions.
    """
    results = []
    fig_rows = len(CORR_PAIRS_MU_JET) + 1   # +1 for ΔR
    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 4 * fig_rows))
    fig.suptitle("Muon ↔ Jet Correlations", fontsize=15, fontweight="bold", y=1.01)

    df_real = df[df["jet1_real"]].copy()
    df_fake = df[df["jet1_fake"]].copy()

    # ── Kinematic scatter / 2D histogram ──────────────────────────────────────
    for row, (mu_var, jet_var_base, title) in enumerate(CORR_PAIRS_MU_JET):
        jet_var = resolve_col(df, jet_var_base)
        if mu_var not in df.columns or jet_var is None:
            for col in range(2):
                axes[row, col].text(0.5, 0.5, f"Column not found\n{jet_var_base}",
                                    ha="center", va="center", transform=axes[row, col].transAxes)
            continue

        ax_real = axes[row, 0]
        ax_fake = axes[row, 1]

        for ax, sub_df, jtype, color in [
            (ax_real, df_real, "Real",  COLORS["real"]),
            (ax_fake, df_fake, "Fake",  COLORS["fake"]),
        ]:
            x = sub_df[mu_var].dropna().values
            y = sub_df[jet_var].dropna().values
            n = min(len(x), len(y))
            if n < 5:
                ax.text(0.5, 0.5, "No events", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            # Determine axis ranges based on whether variable is pT
            xrange = MU_PT_RANGE  if "pt" in mu_var.lower()  else None
            yrange = JET_PT_RANGE if "pt" in jet_var.lower() else None
            hist2d_kw = dict(bins=50,
                             cmap="Blues" if jtype == "Real" else "Reds",
                             norm=matplotlib.colors.LogNorm())
            if xrange: hist2d_kw["range"] = [xrange, yrange or [y.min(), y.max()]]
            if yrange and not xrange:
                hist2d_kw["range"] = [[x.min(), x.max()], yrange]
            ax.hist2d(x[:n], y[:n], **hist2d_kw)
            if xrange: ax.set_xlim(*xrange)
            if yrange: ax.set_ylim(*yrange)
            ax.set_xlabel(mu_var, fontsize=10)
            ax.set_ylabel(jet_var, fontsize=10)
            ax.set_title(f"{jtype} jets — {title}", fontsize=10)
            ax.grid(True, alpha=0.3)

            res = correlation_summary(sub_df[mu_var].values,
                                      sub_df[jet_var].values,
                                      f"{jtype}: {mu_var} vs {jet_var_base}")
            if res:
                results.append(res)
                ax.annotate(
                    f"Pearson r = {res['pearson_r']}\nSpearman r = {res['spearman_r']}",
                    xy=(0.04, 0.92), xycoords="axes fraction",
                    fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )

    # ── ΔR row ────────────────────────────────────────────────────────────────
    dr_row = len(CORR_PAIRS_MU_JET)
    ax_dr_real = axes[dr_row, 0]
    ax_dr_fake = axes[dr_row, 1]

    j1_eta_col = resolve_col(df, "jet1_eta_nominal")
    j1_phi_col = resolve_col(df, "jet1_phi_nominal")

    for ax, sub_df, jtype, color in [
        (ax_dr_real, df_real, "Real",  COLORS["real"]),
        (ax_dr_fake, df_fake, "Fake",  COLORS["fake"]),
    ]:
        if all(c is not None for c in [j1_eta_col, j1_phi_col]) and \
           all(c in sub_df.columns for c in ["mu1_eta","mu1_phi"]):
            dr = delta_r(sub_df["mu1_eta"].values,      sub_df["mu1_phi"].values,
                         sub_df[j1_eta_col].values,     sub_df[j1_phi_col].values)
            dr = dr[np.isfinite(dr)]
            if len(dr) > 0:
                ax.hist(dr, bins=60, color=color, histtype="step", linewidth=1.8, alpha=0.8)
                ax.set_xlabel(r"$\Delta R(\mu_1, j_1)$", fontsize=10)
                ax.set_ylabel("Events", fontsize=10)
                ax.set_title(f"ΔR — μ₁ vs {jtype} jet  (N={len(dr):,})", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axvline(0.4, color="gray", ls="--", lw=1, label="ΔR=0.4")
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "eta/phi columns not found", ha="center", va="center",
                    transform=ax.transAxes)

    plt.tight_layout()
    save_fig(fig, outdir, "1_mu_jet_correlations")
    # Individual panel PDFs
    panel_labels = [
        (f"{mu_var.replace('/','_')}_vs_{jet_var_base.replace('/','_')}", mu_var, jet_var_base)
        for mu_var, jet_var_base, _ in CORR_PAIRS_MU_JET
    ]
    for row, (mu_var, jet_var_base, _) in enumerate(CORR_PAIRS_MU_JET):
        slug = f"{mu_var}_vs_{jet_var_base}".replace("/","_")
        save_panel(axes[row, 0], outdir, f"1a_{slug}_real")
        save_panel(axes[row, 1], outdir, f"1b_{slug}_fake")
    dr_row = len(CORR_PAIRS_MU_JET)
    save_panel(axes[dr_row, 0], outdir, "1c_deltaR_mu_jet1_real")
    save_panel(axes[dr_row, 1], outdir, "1d_deltaR_mu_jet1_fake")
    plt.close(fig)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Mixed-event jet–jet correlations
# ══════════════════════════════════════════════════════════════════════════════

def plot_mixed_event_correlations(df: pd.DataFrame, outdir: str) -> list[dict]:
    """
    Events with 2 jets:
      real  vs real   (both jets real)
      fake  vs fake   (both jets fake)
      fake  vs real   (mixed)
    """
    results = []

    # Resolve actual column names (handles _nominal suffix)
    j1_eta = resolve_col(df, "jet1_eta_nominal")
    j1_phi = resolve_col(df, "jet1_phi_nominal")
    j2_pt  = resolve_col(df, "jet2_pt_nominal")
    j2_eta = resolve_col(df, "jet2_eta_nominal")
    j2_phi = resolve_col(df, "jet2_phi_nominal")
    j1_pt  = resolve_col(df, "jet1_pt_nominal")

    if not all([j2_pt, j2_eta, j2_phi, "jet1_real" in df.columns, "jet2_real" in df.columns]):
        missing = [n for n,v in [("jet2_pt_nominal",j2_pt),("jet2_eta_nominal",j2_eta),
                                  ("jet2_phi_nominal",j2_phi)] if v is None]
        missing += [c for c in ["jet1_real","jet2_real"] if c not in df.columns]
        print(f"  Skipping mixed-event section (missing: {missing})")
        return results

    # Restrict to events that actually have a jet2 (pt > 0 / not NaN)
    has_jet2 = df[j2_pt].notna() & (df[j2_pt] > 0)
    dfj = df[has_jet2].copy()

    # Define subsets
    rr = dfj[ dfj["jet1_real"] &  dfj["jet2_real"]].copy()
    ff = dfj[~dfj["jet1_real"] & ~dfj["jet2_real"]].copy()
    fr = dfj[dfj["mixed"]].copy()

    print(f"  Jet-2 events: total={len(dfj):,}  RR={len(rr):,}  FF={len(ff):,}  FR={len(fr):,}")

    subsets = [
        (rr, "real vs real",  COLORS["real_real"]),
        (ff, "fake vs fake",  COLORS["fake_fake"]),
        (fr, "fake vs real",  COLORS["fake_real"]),
    ]

    jet_corr_vars = [
        (j1_pt,  j2_pt,  r"$p_T^{j_1}$ vs $p_T^{j_2}$"),
        (j1_eta, j2_eta, r"$\eta^{j_1}$ vs $\eta^{j_2}$"),
    ]

    # ── 2D histograms ─────────────────────────────────────────────────────────
    n_rows = len(jet_corr_vars) + 1   # +1 for ΔR
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    fig.suptitle("Mixed-Event Jet–Jet Correlations", fontsize=15,
                 fontweight="bold", y=1.01)

    cmaps = ["Blues", "Reds", "Oranges"]

    for row, (v1, v2, title) in enumerate(jet_corr_vars):
        for col, (sub_df, label, color) in enumerate(subsets):
            ax = axes[row, col]
            if len(sub_df) == 0:
                ax.text(0.5, 0.5, "No events", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            x = sub_df[v1].dropna().values
            y = sub_df[v2].dropna().values
            n = min(len(x), len(y))
            xrange = JET_PT_RANGE if "pt" in (v1 or "").lower() else None
            yrange = JET_PT_RANGE if "pt" in (v2 or "").lower() else None
            hist2d_kw = dict(bins=40, cmap=cmaps[col], norm=matplotlib.colors.LogNorm())
            if xrange or yrange:
                hist2d_kw["range"] = [xrange or [x.min(), x.max()],
                                      yrange or [y.min(), y.max()]]
            ax.hist2d(x[:n], y[:n], **hist2d_kw)
            if xrange: ax.set_xlim(*xrange)
            if yrange: ax.set_ylim(*yrange)
            ax.set_xlabel(v1, fontsize=9)
            ax.set_ylabel(v2, fontsize=9)
            ax.set_title(f"{label}\n{title}", fontsize=9)
            ax.grid(True, alpha=0.3)

            res = correlation_summary(sub_df[v1].values, sub_df[v2].values,
                                      f"{label}: {v1} vs {v2}")
            if res:
                results.append(res)
                ax.annotate(
                    f"Pearson r = {res['pearson_r']}\nSpearman r = {res['spearman_r']}\nN = {res['n']:,}",
                    xy=(0.04, 0.92), xycoords="axes fraction",
                    fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )

    # ── ΔR row ────────────────────────────────────────────────────────────────
    dr_row = len(jet_corr_vars)
    for col, (sub_df, label, color) in enumerate(subsets):
        ax = axes[dr_row, col]
        if len(sub_df) < 5:
            ax.text(0.5, 0.5, "No events", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        if all(c is not None for c in [j1_eta, j1_phi, j2_eta, j2_phi]):
            dr = delta_r(sub_df[j1_eta].values, sub_df[j1_phi].values,
                         sub_df[j2_eta].values, sub_df[j2_phi].values)
            dr = dr[np.isfinite(dr)]
            ax.hist(dr, bins=60, color=color, histtype="step", linewidth=2, alpha=0.8)
            ax.set_xlabel(r"$\Delta R(j_1, j_2)$", fontsize=9)
            ax.set_ylabel("Events", fontsize=9)
            ax.set_title(f"ΔR — {label}  (N={len(dr):,})", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(0.4, color="gray", ls="--", lw=1, label="ΔR=0.4")
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, outdir, "2_mixed_event_jet_correlations")
    # Individual panel PDFs — jet_corr_vars rows + ΔR row, 3 subset columns
    subset_slugs = ["real_real", "fake_fake", "fake_real"]
    var_slugs    = ["pt_j1_vs_pt_j2", "eta_j1_vs_eta_j2", "deltaR_j1_j2"]
    for row_i, vslug in enumerate(var_slugs):
        for col_i, sslug in enumerate(subset_slugs):
            save_panel(axes[row_i, col_i], outdir, f"2_{vslug}_{sslug}")
    plt.close(fig)

    # ── ΔR overlay (all three subsets) ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for sub_df, label, color in subsets:
        if len(sub_df) < 5:
            continue
        dr = delta_r(sub_df[j1_eta].values, sub_df[j1_phi].values,
                     sub_df[j2_eta].values, sub_df[j2_phi].values)
        dr = dr[np.isfinite(dr)]
        ax2.hist(dr, bins=60, color=color, histtype="step", linewidth=2,
                 density=True, label=f"{label} (N={len(dr):,})")
    ax2.axvline(0.4, color="gray", ls="--", lw=1, label="ΔR=0.4")
    ax2.set_xlabel(r"$\Delta R(j_1, j_2)$", fontsize=12)
    ax2.set_ylabel("Normalised events", fontsize=12)
    ax2.set_title("ΔR(j₁,j₂) — real/real vs fake/fake vs mixed (normalised)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, outdir, "2b_deltaR_overlay")
    plt.close(fig2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Summary comparison panel (muon pT / eta)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mu_comparison(df: pd.DataFrame, outdir: str):
    """Overlay muon distributions split by event jet-type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Muon Distributions by Jet Type", fontsize=13, fontweight="bold")

    df_real = df[df["jet1_real"]]
    df_fake = df[df["jet1_fake"]]

    for ax, var, xlabel in [
        (axes[0], "mu1_pt",  r"$p_T^{\mu_1}$ [GeV]"),
        (axes[1], "mu1_eta", r"$\eta^{\mu_1}$"),
    ]:
        if var not in df.columns:
            continue
        for sub_df, label, color in [
            (df_real, "Real jet events", COLORS["real"]),
            (df_fake, "Fake jet events", COLORS["fake"]),
        ]:
            vals = sub_df[var].dropna().values
            pt_range = MU_PT_RANGE if var == "mu1_pt" else None
            ax.hist(vals, bins=60, color=color, histtype="step",
                    linewidth=2, density=True, label=f"{label} (N={len(vals):,})",
                    range=pt_range if pt_range else (vals.min(), vals.max()))
            if pt_range: ax.set_xlim(*pt_range)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Normalised events", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, outdir, "3_muon_comparison")
    save_panel(axes[0], outdir, "3a_mu1_pt_by_jettype")
    save_panel(axes[1], outdir, "3b_mu1_eta_by_jettype")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Min ΔR between muons and jets
# ══════════════════════════════════════════════════════════════════════════════

def compute_min_dr_mu_jets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event compute the minimum ΔR between any muon (mu1, mu2)
    and any jet (jet1..jet4) that is present (pt > 0).
    Stores: minDR_mu_jet, minDR_mu_realjet, minDR_mu_fakejet
    """
    mu_cols  = [("mu1_eta", "mu1_phi"), ("mu2_eta", "mu2_phi")]
    jet_idxs = [1, 2, 3, 4]

    # Build arrays: shape (N,) for each jet
    for mu_eta_col, mu_phi_col in mu_cols:
        if mu_eta_col not in df.columns:
            continue
        mu_eta = df[mu_eta_col].values.astype(float)
        mu_phi = df[mu_phi_col].values.astype(float)

        for jidx in jet_idxs:
            j_eta_col = resolve_col(df, f"jet{jidx}_eta_nominal")
            j_phi_col = resolve_col(df, f"jet{jidx}_phi_nominal")
            j_pt_col  = resolve_col(df, f"jet{jidx}_pt_nominal")
            if j_eta_col is None or j_phi_col is None:
                continue
            j_eta = df[j_eta_col].values.astype(float)
            j_phi = df[j_phi_col].values.astype(float)
            j_pt  = df[j_pt_col].values.astype(float) if j_pt_col else np.ones(len(df))
            present = np.isfinite(j_eta) & np.isfinite(j_phi) & (j_pt > 0)
            dr = delta_r(mu_eta, mu_phi, j_eta, j_phi)
            dr = np.where(present, dr, np.inf)
            col_name = f"_dr_{mu_eta_col[:3]}_{jidx}"   # e.g. _dr_mu1_1
            df[col_name] = dr

    # Collect all per-(muon,jet) DR columns
    dr_cols      = [c for c in df.columns if c.startswith("_dr_")]
    real_jr_cols = []
    fake_jr_cols = []
    for jidx in jet_idxs:
        real_col = f"jet{jidx}_real"
        fake_col = f"jet{jidx}_fake"
        for mu_eta_col, _ in mu_cols:
            col = f"_dr_{mu_eta_col[:3]}_{jidx}"
            if col not in df.columns:
                continue
            if real_col in df.columns:
                real_jr_cols.append((col, real_col))
            if fake_col in df.columns:
                fake_jr_cols.append((col, fake_col))

    if dr_cols:
        df["minDR_mu_jet"] = df[dr_cols].min(axis=1)

    # min DR to a real jet
    if real_jr_cols:
        real_dr_arr = np.column_stack([
            np.where(df[rc].values, df[dc].values, np.inf)
            for dc, rc in real_jr_cols
        ])
        df["minDR_mu_realjet"] = real_dr_arr.min(axis=1)

    # min DR to a fake jet
    if fake_jr_cols:
        fake_dr_arr = np.column_stack([
            np.where(df[fc].values, df[dc].values, np.inf)
            for dc, fc in fake_jr_cols
        ])
        df["minDR_mu_fakejet"] = fake_dr_arr.min(axis=1)

    # Replace inf with NaN for cleaner plotting
    for c in ["minDR_mu_jet", "minDR_mu_realjet", "minDR_mu_fakejet"]:
        if c in df.columns:
            df[c] = df[c].replace(np.inf, np.nan)

    # Drop temp columns
    df.drop(columns=dr_cols, inplace=True, errors="ignore")
    return df


def plot_min_dr(df: pd.DataFrame, outdir: str):
    """
    Single plot with three distributions by event jet composition (>=2 jets):
      1. both jets real
      2. both jets fake
      3. one real + one fake
    Each curve shows min ΔR(μ, any jet) for that event class.
    """
    col = "minDR_mu_jet"
    if col not in df.columns:
        print("  Skipping min-ΔR section (minDR_mu_jet not computed)")
        return

    compositions = [
        ("evtcomp_both_real",  "both jets real",      COLORS["real_real"]),
        ("evtcomp_both_fake",  "both jets fake",      COLORS["fake_fake"]),
        ("evtcomp_mixed",      "one real + one fake", COLORS["fake_real"]),
    ]
    comp_cols_present = [c for c, _, _ in compositions if c in df.columns]
    if not comp_cols_present:
        print("  Skipping min-ΔR section (evtcomp columns not found)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(r"Min $\Delta R(\mu, \mathrm{jet})$ by event jet composition",
                 fontsize=14, fontweight="bold")

    for comp_col, label, color in compositions:
        if comp_col not in df.columns:
            continue
        vals = df.loc[df[comp_col], col].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        ax.hist(vals, bins=60, range=(0, 6), histtype="step",
                linewidth=2, density=True, color=color,
                label=f"{label} (N={len(vals):,})")

    ax.axvline(0.4, color="gray", ls=":", lw=1.5, label=r"$\Delta R=0.4$")
    ax.set_xlabel(r"min $\Delta R(\mu, \mathrm{jet})$", fontsize=12)
    ax.set_ylabel("Normalised events", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, outdir, "4_min_dR_mu_jet")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — ΔΦ(MET, muons) and ΔΦ(MET, jets)
# ══════════════════════════════════════════════════════════════════════════════

def compute_met_dphi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute min/max |ΔΦ| between PuppiMET and muons/jets.
    Jets are treated per-event-composition: both_real, both_fake, mixed.
    Only events with >=2 jets are used for jet ΔΦ.
    """
    met_phi_col = next((c for c in ["PuppiMET_phi", "MET_phi", "met_phi"]
                        if c in df.columns), None)
    if met_phi_col is None:
        print("  Warning: no MET phi column found — skipping MET ΔΦ computation")
        return df

    met_phi = df[met_phi_col].values.astype(float)

    # ── Muons: min/max over mu1, mu2 ─────────────────────────────────────────
    mu_dphi_list = []
    for mu_phi_col in ["mu1_phi", "mu2_phi"]:
        if mu_phi_col in df.columns:
            mu_dphi_list.append(np.abs(delta_phi(met_phi, df[mu_phi_col].values.astype(float))))
    if mu_dphi_list:
        mu_arr = np.column_stack(mu_dphi_list)
        df["minDPhi_MET_mu"] = mu_arr.min(axis=1)
        df["maxDPhi_MET_mu"] = mu_arr.max(axis=1)

    # ── Jets: min/max over jet1+jet2 for events with >=2 jets ────────────────
    dphi_per_jet = {}
    for jidx in [1, 2]:
        j_phi_col = resolve_col(df, f"jet{jidx}_phi_nominal")
        j_pt_col  = resolve_col(df, f"jet{jidx}_pt_nominal")
        if j_phi_col is None:
            continue
        j_phi = df[j_phi_col].values.astype(float)
        j_pt  = df[j_pt_col].values.astype(float) if j_pt_col else np.ones(len(df))
        present = np.isfinite(j_phi) & (j_pt > 0)
        dphi_per_jet[jidx] = np.where(present, np.abs(delta_phi(met_phi, j_phi)), np.nan)

    if len(dphi_per_jet) == 2:
        arr2 = np.column_stack([dphi_per_jet[1], dphi_per_jet[2]])
        df["minDPhi_MET_jet"] = np.nanmin(arr2, axis=1)
        df["maxDPhi_MET_jet"] = np.nanmax(arr2, axis=1)

    return df


def plot_met_dphi(df: pd.DataFrame, outdir: str):
    """
    5a — ΔΦ(MET, muons): min & max, three event-composition curves
         (both_real / both_fake / mixed jet events)
    5b — ΔΦ(MET, jets):  min & max, same three event compositions
         Only events with >=2 jets.
    """
    # Event-composition subsets (>=2 jets required for jet plots, optional for muon)
    compositions = [
        ("evtcomp_both_real", "both jets real",  COLORS["real_real"]),
        ("evtcomp_both_fake", "both jets fake",  COLORS["fake_fake"]),
        ("evtcomp_mixed",     "one real+one fake", COLORS["fake_real"]),
    ]
    # Check at least one composition column exists
    comp_cols_present = [c for c, _, _ in compositions if c in df.columns]
    if not comp_cols_present:
        print("  Skipping MET ΔΦ plots (evtcomp columns not found)")
        return

    pi_ticks      = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    pi_ticklabels = ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]

    def _hist(ax, vals, color, label):
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            return
        ax.hist(vals, bins=60, range=(0, np.pi), histtype="step",
                linewidth=2, density=True, color=color,
                label=f"{label} (N={len(vals):,})")

    def _style(ax, title):
        ax.set_xlabel(r"$|\Delta\phi|$", fontsize=11)
        ax.set_ylabel("Normalised events", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, np.pi)
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_ticklabels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 5a: MET–muon ΔΦ ──────────────────────────────────────────────────────
    mu_pairs = [(c, t) for c, t in [("minDPhi_MET_mu", r"min $|\Delta\phi|$(MET, $\mu$)"),
                                     ("maxDPhi_MET_mu", r"max $|\Delta\phi|$(MET, $\mu$)")]
                if c in df.columns]
    if mu_pairs:
        fig, axes = plt.subplots(1, len(mu_pairs), figsize=(7*len(mu_pairs), 5))
        if len(mu_pairs) == 1:
            axes = [axes]
        fig.suptitle(r"$\Delta\phi$(MET, $\mu$) by jet composition", fontsize=14, fontweight="bold")
        for ax, (col, title) in zip(axes, mu_pairs):
            for comp_col, label, color in compositions:
                if comp_col not in df.columns:
                    continue
                vals = df.loc[df[comp_col], col].dropna().values
                _hist(ax, vals, color, label)
            _style(ax, title)
        plt.tight_layout()
        save_fig(fig, outdir, "5a_dphi_MET_muon")
        for ax_i, (col, _) in enumerate(mu_pairs):
            slug = "min" if "min" in col else "max"
            save_panel(axes[ax_i], outdir, f"5a_dphi_MET_muon_{slug}")
        plt.close(fig)

    # ── 5b: MET–jet ΔΦ (>=2 jets events only) ────────────────────────────────
    jet_pairs = [(c, t) for c, t in [("minDPhi_MET_jet", r"min $|\Delta\phi|$(MET, jet)"),
                                      ("maxDPhi_MET_jet", r"max $|\Delta\phi|$(MET, jet)")]
                 if c in df.columns]
    if jet_pairs:
        fig, axes = plt.subplots(1, len(jet_pairs), figsize=(7*len(jet_pairs), 5))
        if len(jet_pairs) == 1:
            axes = [axes]
        fig.suptitle(r"$\Delta\phi$(MET, jet) — events with $\geq$2 jets, by composition",
                     fontsize=14, fontweight="bold")
        for ax, (col, title) in zip(axes, jet_pairs):
            for comp_col, label, color in compositions:
                if comp_col not in df.columns:
                    continue
                # Only >=2 jet events
                mask = df[comp_col] & df.get("has_2jets", pd.Series(True, index=df.index))
                vals = df.loc[mask, col].dropna().values
                _hist(ax, vals, color, label)
            _style(ax, title)
        plt.tight_layout()
        save_fig(fig, outdir, "5b_dphi_MET_jet")
        for ax_i, (col, _) in enumerate(jet_pairs):
            slug = "min" if "min" in col else "max"
            save_panel(axes[ax_i], outdir, f"5b_dphi_MET_jet_{slug}")
        plt.close(fig)



# ==============================================================================
# Section 6 -- Jet composition / ID variables: real vs fake
# ==============================================================================

def _gather_jet_id(df, base):
    """Stack real/fake values for base variable across jet1 and jet2."""
    real_list, fake_list = [], []
    for jidx in [1, 2]:
        col = resolve_col(df, f"jet{jidx}_{base}_nominal")
        if col is None:
            bare = f"jet{jidx}_{base}"
            col = bare if bare in df.columns else None
        if col is None:
            continue
        pt_col = resolve_col(df, f"jet{jidx}_pt_nominal")
        present = (df[pt_col].notna() & (df[pt_col] > 0)) if pt_col else pd.Series(True, index=df.index)
        real_flag = df.get(f"jet{jidx}_real", pd.Series(False, index=df.index))
        fake_flag = df.get(f"jet{jidx}_fake", pd.Series(False, index=df.index))
        real_list.append(df.loc[real_flag & present, col].dropna().values.astype(float))
        fake_list.append(df.loc[fake_flag & present, col].dropna().values.astype(float))
    real_vals = np.concatenate(real_list) if real_list else np.array([])
    fake_vals = np.concatenate(fake_list) if fake_list else np.array([])
    return real_vals[np.isfinite(real_vals)], fake_vals[np.isfinite(fake_vals)]


def _draw_jet_id_panel(ax, real_vals, fake_vals, xlabel, xrange, is_integer, log_y=False):
    """Draw a single jet-ID panel onto ax."""
    if is_integer:
        lo, hi = int(xrange[0]), int(xrange[1])
        bins = np.arange(lo, hi + 2) - 0.5   # centred integer bins
    else:
        bins = np.linspace(xrange[0], xrange[1], 61)

    hist_kw = dict(bins=bins, histtype="step", linewidth=2, density=True)

    if len(real_vals) >= 10:
        ax.hist(real_vals, color=COLORS["real"],
                label=f"Real (N={len(real_vals):,})", **hist_kw)
    if len(fake_vals) >= 10:
        ax.hist(fake_vals, color=COLORS["fake"],
                label=f"Fake (N={len(fake_vals):,})", **hist_kw)

    ax.set_xlim(*xrange)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Norm. jets (log)" if log_y else "Normalised jets", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both" if log_y else "major")
    if log_y:
        ax.set_yscale("log")


def plot_jet_id_vars(df: pd.DataFrame, outdir: str):
    """
    For each variable in JET_ID_VARS overlay real vs fake jets (jet1+jet2 stacked).

    Produces four files per variable:
      6_jet_id_overview_linear.png/.pdf  -- linear-y grid
      6_jet_id_overview_log.png/.pdf     -- log-y grid
      6_<var>_linear.pdf                 -- individual linear panel
      6_<var>_log.pdf                    -- individual log-y panel
    """
    # Filter to variables present in data
    available = []
    for base, xlabel, xrange, is_int in JET_ID_VARS:
        col = resolve_col(df, f"jet1_{base}_nominal")
        if col is None and f"jet1_{base}" in df.columns:
            col = f"jet1_{base}"
        if col is not None:
            available.append((base, xlabel, xrange, is_int))
        else:
            print(f"  Skipping {base} (not found in data)")

    if not available:
        print("  No jet ID variables found -- skipping section 6")
        return

    # Pre-gather all data
    data = {base: _gather_jet_id(df, base) for base, *_ in available}

    n_vars = len(available)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    for log_y in [False, True]:
        suffix = "log" if log_y else "linear"
        title  = f"Jet Composition Variables -- Real vs Fake Jets  ({'log' if log_y else 'linear'} scale)"

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5.5 * n_cols, 4.5 * n_rows))
        axes_flat = axes.flatten() if n_vars > 1 else [axes]
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        for ax_idx, (base, xlabel, xrange, is_int) in enumerate(available):
            ax = axes_flat[ax_idx]
            real_vals, fake_vals = data[base]
            _draw_jet_id_panel(ax, real_vals, fake_vals, xlabel, xrange, is_int, log_y=log_y)
            ax.set_title(base, fontsize=10, fontweight="bold")

        for ax in axes_flat[n_vars:]:
            ax.set_visible(False)

        plt.tight_layout()
        save_fig(fig, outdir, f"6_jet_id_overview_{suffix}")

        # Individual panel PDFs
        for ax_idx, (base, xlabel, xrange, is_int) in enumerate(available):
            save_panel(axes_flat[ax_idx], outdir, f"6_{base}_{suffix}")

        plt.close(fig)

# Print results table
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict]):
    if not results:
        return
    col_w = [max(len(str(r.get(k, ""))) for r in results) for k in results[0]]
    header = list(results[0].keys())
    fmt = "  ".join(f"{{:<{w}}}" for w in
                    [max(len(h), max(len(str(r.get(h,""))) for r in results))
                     for h in header])
    print("\n" + "─"*90)
    print("CORRELATION SUMMARY")
    print("─"*90)
    print(fmt.format(*header))
    print("─"*90)
    for r in results:
        print(fmt.format(*[str(r.get(h,"")) for h in header]))
    print("─"*90 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Jet–Muon correlation analysis (real/fake jets)")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Parquet file(s) or glob pattern")
    parser.add_argument("--output", default="./corr_plots",
                        help="Output directory for plots")
    parser.add_argument("--gen-jet-col", default=None,
                        help="Override hasMatchedGenJet column name")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Resolve all input paths / globs
    print(f"\nResolving input paths …")
    files = resolve_input_paths(args.input)

    print(f"\nLoading {len(files)} file(s) …")
    df = load_parquet(files)

    # Allow custom hasMatchedGenJet column name
    if args.gen_jet_col and args.gen_jet_col in df.columns:
        df["hasMatchedGenJet"] = df[args.gen_jet_col]
        # propagate to per-jet columns if absent
        if "jet1_hasMatchedGenJet" not in df.columns:
            df["jet1_hasMatchedGenJet"] = df[args.gen_jet_col]

    print(f"Columns: {list(df.columns)[:30]} …")
    print(f"Shape: {df.shape}")

    # Classify
    df = classify_jets(df)
    n_real = df["jet1_real"].sum()
    n_fake = df["jet1_fake"].sum()
    n_mixed = df["mixed"].sum() if "mixed" in df.columns else 0
    print(f"\nEvent counts: real-jet={n_real:,}  fake-jet={n_fake:,}  mixed={n_mixed:,}")

    all_results = []

    # ── Section 1 ─────────────────────────────────────────────────────────────
    print("\n[1/3] Muon ↔ Jet correlations …")
    res1 = plot_mu_jet_correlations(df, args.output)
    all_results.extend(res1)

    # ── Section 2 ─────────────────────────────────────────────────────────────
    print("\n[2/3] Mixed-event jet–jet correlations …")
    res2 = plot_mixed_event_correlations(df, args.output)
    all_results.extend(res2)

    # ── Section 3 ─────────────────────────────────────────────────────────────
    print("\n[3/3] Muon comparison …")
    plot_mu_comparison(df, args.output)

    # ── Section 4 ─────────────────────────────────────────────────────────────
    print("\n[4/5] Min ΔR(μ, jet) …")
    df = compute_min_dr_mu_jets(df)
    plot_min_dr(df, args.output)

    # ── Section 5 ─────────────────────────────────────────────────────────────
    print("\n[5/5] ΔΦ(MET, muons/jets) …")
    df = compute_met_dphi(df)
    plot_met_dphi(df, args.output)

    # ── Section 6 ─────────────────────────────────────────────────────────────────────────────
    print("\n[6/6] Jet composition / ID variables...")
    plot_jet_id_vars(df, args.output)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_results(all_results)

    # Save CSV
    if all_results:
        csv_path = os.path.join(args.output, "correlation_summary.csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"Correlation table saved to {csv_path}")

    print(f"\nAll outputs in: {args.output}/")


if __name__ == "__main__":
    main()
