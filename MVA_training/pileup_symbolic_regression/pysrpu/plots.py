import matplotlib.pyplot as plt
import numpy as np
from .metrics import compute_wp_vs_pt
import ROOT
import os

# ============================================================
# ROOT plot helpers
# ============================================================
def make_tgraph_finite(x, y, name, title, xlab, ylab):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return None
    g = ROOT.TGraph(int(m.sum()), x[m], y[m])
    g.SetName(name)
    g.SetTitle(f"{title};{xlab};{ylab}")
    g.SetLineWidth(2)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.0)
    return g


def plot_effrej_vs_pt(df: "pd.DataFrame", pass_mask: "np.ndarray", pt_bins: list[float], outpdf: str, tag: str) -> None:
    """Save dual-axis HS-eff and PU-rej vs pT plot to PDF."""
    pass


def plot_purej_vs_hseff(df: "pd.DataFrame", pass_mask: "np.ndarray", pt_bins: list[float], outpdf: str, tag: str) -> None:
    """Save ROC-like PU-rej vs HS-eff points (per pT bin) to PDF."""
    pass


def plot_eff_rej_vs_pt(pt_centers, hs_eff, pu_rej, outpath):
    plt.figure(figsize=(7,6))
    plt.plot(pt_centers, hs_eff, marker='o', label="HS efficiency")
    plt.plot(pt_centers, pu_rej, marker='s', label="PU rejection")
    plt.ylim(0,1)
    plt.xlabel("Jet pT [GeV]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_wp_vs_pt(df, pass_mask, pt_bins, outdir, tag):
    centers = 0.5 * (np.array(pt_bins[:-1]) + np.array(pt_bins[1:]))

    hs_eff, pu_rej = compute_wp_vs_pt(df, pass_mask, pt_bins)

    c = ROOT.TCanvas(f"c_eff_rej_{tag}", "", 800, 650)

    # --- HS efficiency (left axis)
    g_eff = make_tgraph_finite(
        centers, hs_eff,
        f"g_eff_{tag}",
        "",
        "jet p_{T} [GeV]", "HS efficiency"
    )

    # --- PU rejection (right axis)
    g_rej = make_tgraph_finite(
        centers, pu_rej,
        f"g_rej_{tag}",
        "",
        "", ""
    )

    g_eff.SetLineColor(ROOT.kBlue + 1)
    g_eff.SetMarkerColor(ROOT.kBlue + 1)
    g_eff.SetLineWidth(2)

    g_eff.Draw("APL")
    g_eff.GetYaxis().SetRangeUser(0.0, 1.0)
    g_eff.GetYaxis().SetTitle("HS efficiency")

    ROOT.gPad.Update()

    # --- Right axis
    axis = ROOT.TGaxis(
        ROOT.gPad.GetUxmax(),
        ROOT.gPad.GetUymin(),
        ROOT.gPad.GetUxmax(),
        ROOT.gPad.GetUymax(),
        0.0, 1.0, 510, "+L"
    )
    axis.SetTitle("PU rejection")
    axis.SetLineColor(ROOT.kRed + 1)
    axis.SetLabelColor(ROOT.kRed + 1)
    axis.SetTitleColor(ROOT.kRed + 1)
    axis.Draw()

    g_rej.SetLineColor(ROOT.kRed + 1)
    g_rej.SetMarkerColor(ROOT.kRed + 1)
    g_rej.SetLineWidth(2)

    g_rej.Draw("PL SAME")
    g_rej.GetYaxis().SetRangeUser(0.0, 1.0)

    # --- Legend
    leg = ROOT.TLegend(0.55, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_eff, "HS efficiency", "lp")
    leg.AddEntry(g_rej, "PU rejection", "lp")
    leg.Draw()

    # --- Save
    c.SaveAs(os.path.join(outdir, f"eff_and_rej_vs_pt_{tag}.pdf"))
