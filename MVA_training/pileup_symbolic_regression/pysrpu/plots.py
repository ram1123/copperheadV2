import matplotlib.pyplot as plt
import numpy as np
from .metrics import compute_wp_vs_pt
import ROOT
import os

def infer_range(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (0.0, 1.0)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def make_hist(name, title, nbins, xmin, xmax):
    h = ROOT.TH1F(name, title, nbins, xmin, xmax)
    h.SetStats(0)
    h.Sumw2()
    return h


def fill_hist(h, arr):
    for v in arr:
        if np.isfinite(v):
            h.Fill(float(v))


def normalize_hist(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())


def save_canvas(c, outpath):
    c.SaveAs(outpath)
    print(f"[Saved] {outpath}")


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


def plot_score_real_fake(score, y_hs, outpath, region):
    good = np.isfinite(score)
    score = score[good]
    y_hs = y_hs[good]

    xmin, xmax = infer_range(score)
    h_real = make_hist("h_real", f"{region} score;score;Normalized entries", 50, xmin, xmax)
    h_fake = make_hist("h_fake", f"{region} score;score;Normalized entries", 50, xmin, xmax)

    fill_hist(h_real, score[y_hs])
    fill_hist(h_fake, score[~y_hs])

    normalize_hist(h_real)
    normalize_hist(h_fake)

    h_real.SetLineColor(ROOT.kBlue + 1)
    h_real.SetLineWidth(2)
    h_fake.SetLineColor(ROOT.kRed + 1)
    h_fake.SetLineWidth(2)

    ymax = max(h_real.GetMaximum(), h_fake.GetMaximum()) * 1.25

    c = ROOT.TCanvas("c_score", "", 850, 700)
    h_real.SetMaximum(ymax)
    h_real.Draw("hist")
    h_fake.Draw("hist same")

    leg = ROOT.TLegend(0.62, 0.76, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_real, "Real (HS)", "l")
    leg.AddEntry(h_fake, "Fake (PU)", "l")
    leg.Draw()

    save_canvas(c, outpath)


def plot_stacked_before_after(values, y_hs, pass_mask, outbase, title, xlab, nbins=50, xmin=None, xmax=None, logy=False):
    if xmin is None or xmax is None:
        xmin, xmax = infer_range(np.asarray(values)[np.isfinite(values)])

    plot_stacked_total_real_fake(
        values, y_hs,
        outpath=f"{outbase}_before_log.pdf" if logy else f"{outbase}_before.pdf",
        title=f"{title} (before)",
        xlab=xlab,
        nbins=nbins, xmin=xmin, xmax=xmax, logy=logy
    )

    plot_stacked_total_real_fake(
        np.asarray(values)[pass_mask],
        np.asarray(y_hs)[pass_mask],
        outpath=f"{outbase}_after_log.pdf" if logy else f"{outbase}_after.pdf",
        title=f"{title} (after)",
        xlab=xlab,
        nbins=nbins, xmin=xmin, xmax=xmax, logy=logy
    )

# ============================================================
# Stacked real/fake + total
# ============================================================
def plot_stacked_total_real_fake(values, y_hs, outpath, title, xlab, nbins=50, xmin=None, xmax=None, logy=False):
    values = np.asarray(values)
    y_hs = np.asarray(y_hs).astype(bool)
    good = np.isfinite(values)
    values = values[good]
    y_hs = y_hs[good]

    if xmin is None or xmax is None:
        xmin, xmax = infer_range(values)

    h_real = make_hist("h_real", f"{title};{xlab};Entries", nbins, xmin, xmax)
    h_fake = make_hist("h_fake", f"{title};{xlab};Entries", nbins, xmin, xmax)
    h_tot  = make_hist("h_tot",  f"{title};{xlab};Entries", nbins, xmin, xmax)

    for v in values[y_hs]:
        h_real.Fill(float(v))
        h_tot.Fill(float(v))
    for v in values[~y_hs]:
        h_fake.Fill(float(v))
        h_tot.Fill(float(v))

    h_real.SetFillColor(ROOT.kAzure - 2)
    h_real.SetLineColor(ROOT.kAzure - 2)
    h_fake.SetFillColor(ROOT.kOrange)
    h_fake.SetLineColor(ROOT.kOrange + 1)

    h_tot.SetMarkerStyle(20)
    h_tot.SetMarkerSize(0.9)
    h_tot.SetLineColor(ROOT.kBlack)
    h_tot.SetMarkerColor(ROOT.kBlack)

    stack = ROOT.THStack("stack", f"{title};{xlab};Entries")
    stack.Add(h_fake)
    stack.Add(h_real)

    # ============================================================
    # Canvas with 2 pads
    # ============================================================
    c = ROOT.TCanvas("c_stack", "", 850, 800)

    pad1 = ROOT.TPad("pad1", "", 0, 0.30, 1, 1)
    pad2 = ROOT.TPad("pad2", "", 0, 0.00, 1, 0.30)

    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.00)
    pad2.SetBottomMargin(0.30)

    pad1.Draw()
    pad2.Draw()

    # ============================================================
    # Upper pad (stack)
    # ============================================================
    pad1.cd()
    if logy:
        pad1.SetLogy()

    stack.Draw("hist")
    h_tot.Draw("E1 same")

    ymax = max(stack.GetMaximum(), h_tot.GetMaximum()) * (20.0 if logy else 1.35)
    stack.SetMaximum(ymax)
    if logy:
        stack.SetMinimum(1e-1)

    leg = ROOT.TLegend(0.57, 0.72, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_tot, "Total jets", "lep")
    leg.AddEntry(h_real, "Real (HS)", "f")
    leg.AddEntry(h_fake, "Fake (PU)", "f")
    leg.Draw()

    # ============================================================
    # Ratio pad (Fake / Real)
    # ============================================================
    pad2.cd()

    h_ratio = h_real.Clone("h_ratio")
    h_ratio.Reset()

    for i in range(1, h_real.GetNbinsX() + 1):
        r = h_real.GetBinContent(i)
        f = h_fake.GetBinContent(i)

        if r > 0:
            val = f/(r+f)
            h_ratio.SetBinContent(i, val)
            h_ratio.SetBinError(i, 0.0)
        else:
            h_ratio.SetBinContent(i, 0.0)

    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("Fake/Total")
    h_ratio.GetXaxis().SetTitle(xlab)

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.GetYaxis().SetLabelSize(0.08)

    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.0)  # adjust if needed

    h_ratio.Draw("E1")

    # reference line at 1
    line = ROOT.TLine(xmin, 1.0, xmax, 1.0)
    line.SetLineStyle(2)
    line.Draw()

    # ============================================================
    save_canvas(c, outpath)

# ============================================================
# Physics safety plots
# ============================================================
def plot_jet_eta_before_after_mask(dfj, pass_mask, outdir, region):
    eta = dfj["eta"].to_numpy()

    h_before = make_hist("h_eta_before", "Jet #eta before/after PU cut;jet #eta;Entries", 60, -5.0, 5.0)
    h_after  = make_hist("h_eta_after",  "Jet #eta before/after PU cut;jet #eta;Entries", 60, -5.0, 5.0)

    fill_hist(h_before, eta[np.isfinite(eta)])
    fill_hist(h_after, eta[pass_mask & np.isfinite(eta)])


    h_before.SetLineColor(ROOT.kBlack)
    h_before.SetLineWidth(2)
    h_after.SetLineColor(ROOT.kRed + 1)
    h_after.SetLineWidth(2)
    h_after.SetLineStyle(2)

    # ============================================================
    # Canvas with ratio pad
    # ============================================================
    c = ROOT.TCanvas("c_eta_ba", "", 850, 800)

    pad1 = ROOT.TPad("pad1", "", 0, 0.30, 1, 1)
    pad2 = ROOT.TPad("pad2", "", 0, 0.00, 1, 0.30)

    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.30)

    pad1.Draw()
    pad2.Draw()

    # ============================================================
    # Top pad
    # ============================================================
    pad1.cd()

    ymax = max(h_before.GetMaximum(), h_after.GetMaximum()) * 1.25
    h_before.SetMaximum(ymax)

    h_before.Draw("hist")
    h_after.Draw("hist same")

    leg = ROOT.TLegend(0.62, 0.78, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_before, "Before cut", "l")
    leg.AddEntry(h_after, "After cut", "l")
    leg.Draw()

    # ============================================================
    # Ratio pad (After / Before)
    # ============================================================
    pad2.cd()

    h_ratio = h_after.Clone("h_ratio")
    h_ratio.Reset()

    for i in range(1, h_before.GetNbinsX() + 1):
        b = h_before.GetBinContent(i)
        a = h_after.GetBinContent(i)

        if b > 0:
            # h_ratio.SetBinContent(i, a / b)
            h_ratio.SetBinContent(i, (b-a)/b)
            h_ratio.SetBinError(i, 0.0)  # keep simple
        else:
            h_ratio.SetBinContent(i, 0.0)

    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("(Before-After)/Before")
    h_ratio.GetXaxis().SetTitle("jet #eta")

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.GetYaxis().SetLabelSize(0.08)

    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    h_ratio.SetMinimum(0.0)
    h_ratio.SetMaximum(1.0)  # important for efficiency plots

    h_ratio.Draw("E1")

    # reference line at 1
    line = ROOT.TLine(-5.0, 1.0, 5.0, 1.0)
    line.SetLineStyle(2)
    line.Draw()

    # ============================================================
    save_canvas(c, os.path.join(outdir, f"jet_eta_before_after_{region}.pdf"))
