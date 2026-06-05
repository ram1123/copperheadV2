import numpy as np
import dask.dataframe as dd
from pathlib import Path
import ROOT

from modules.classify_year import is_run2

# Years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023", "2023BPix", "2024"]
# Years = ["2016preVFP", "2016postVFP", "2017", "2018"]
Years = ["2016preVFP"]
nanoAODv = "v12"

var        = "gjj_mass"
wgt        = "wgt_nominal"
nbins      = 20
xmin, xmax = 0.0, 1000.0
cols       = ["gjj_mass", wgt, "jj_mass_nominal", "dimuon_mass", "dimuon_pt"]

# ── HELPER: ROOT FillN ────────────────────────────────────────────────────────
def fill_th1(name, title, df, var, wgt, nbins, xmin, xmax):
    """Weighted TH1D using ROOT FillN. NaN → -1 (<=1 jet, no dijet system)."""
    h = ROOT.TH1D(name, title, nbins, xmin, xmax)
    h.Sumw2()
    vals    = df[var].values.astype(np.float64)
    weights = df[wgt].values.astype(np.float64)
    nan_mask = ~np.isfinite(vals)
    if nan_mask.any():
        print(f"  [{name}] {nan_mask.sum()} NaN/Inf → filled at x=-1 (underflow)")
        vals[nan_mask] = -1.0
    h.FillN(len(vals), vals, weights)
    return h

# ── MAIN LOOP OVER YEARS ──────────────────────────────────────────────────────
for Year in Years:
    print(f"\n{'='*60}")
    print(f"  Processing Year: {Year}")
    print(f"{'='*60}")

    # ── PATHS ─────────────────────────────────────────────────────────────────
    inFilePath_run3 = Path("/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/"
                           f"Run3_nanoAODv12_FilterJets_May14_PySR07MayV2/stage1_output/{Year}/compacted/")
    inFilePath_run2 = Path("/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
                        #    f"Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage1_output/{Year}/compacted")
                           f"Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage1_output/{Year}/compacted")

    inFilePath = inFilePath_run2 if is_run2(Year) else inFilePath_run3

    if is_run2(Year):
        if nanoAODv == "v15": file50 = "dyTo2L_M-50_aMCatNLO/*/*.parquet" # v15
        else: file50 = "dy_M-50_MiNNLO/*/*.parquet"
    else:
        file50 = "dyTo2Mu_M-50_aMCatNLO/*/*.parquet" if Year == "2024" else "dyTo2L_M-50_incl/*/*.parquet"

    fileVBF       = "dy_VBF_filter/*/*.parquet"
    mjj_threshold = 350.0 if is_run2(Year) else 300.0

    # ── READ ──────────────────────────────────────────────────────────────────
    df50  = dd.read_parquet(str(inFilePath / file50),  columns=cols)
    dfVBF = dd.read_parquet(str(inFilePath / fileVBF), columns=cols)

    # ── FILTERS ───────────────────────────────────────────────────────────────
    dimuon_cut_50  = (df50["dimuon_mass"]  > 110) & (df50["dimuon_mass"]  < 150)
    dimuon_cut_VBF = (dfVBF["dimuon_mass"] > 110) & (dfVBF["dimuon_mass"] < 150)

    df50_cut  = df50[  (~(df50[var] > mjj_threshold)) & dimuon_cut_50  ].compute()
    dfVBF_cut = dfVBF[ dimuon_cut_VBF                                  ].compute()
    df50_nc   = df50[  dimuon_cut_50                                    ].compute()

    # ── LOOP OVER PLOT VARIABLES ───────────────────────────────────────────────
    for plot_var in ["gjj_mass", "jj_mass_nominal", "dimuon_mass", "dimuon_pt"]:
        print(f"\n  [{Year}] plot_var = {plot_var}")

        nbins      = 20
        xmin, xmax = 0.0, 1000.0
        if plot_var == "dimuon_mass":        
            nbins      = 40
            xmin, xmax = 110, 150.0
        if plot_var == "dimuon_pt": 
            nbins      = 30
            xmin, xmax = 0.0, 300.0                   
        h50    = fill_th1("h50",    f";{plot_var};Events / bin", df50_cut,  plot_var, wgt, nbins, xmin, xmax)
        hVBF   = fill_th1("hVBF",   f";{plot_var};Events / bin", dfVBF_cut, plot_var, wgt, nbins, xmin, xmax)
        h50_nc = fill_th1("h50_nc", f";{plot_var};Events / bin", df50_nc,   plot_var, wgt, nbins, xmin, xmax)

        # ── SCALE FACTOR DIAGNOSIS ─────────────────────────────────────────────
        hVBF_integral         = hVBF.Integral()
        h50_nc_integral_full  = h50_nc.Integral(1, nbins + 1)
        h50_nc_integral_above = h50_nc.Integral(h50_nc.FindBin(mjj_threshold) + 1, nbins + 1)
        VBF_ScaleFactor       = h50_nc_integral_above / hVBF_integral if hVBF_integral else 0.0

        print(f"  Yield DY-VBF                     : {hVBF_integral:.2f}")
        print(f"  Yield Inc DY (full range)         : {h50_nc_integral_full:.2f}")
        print(f"  Yield Inc DY (>{mjj_threshold:.0f})           : {h50_nc_integral_above:.2f}")
        print(f"  Scale factor for DY-VBF           : {VBF_ScaleFactor:.4f}")

        # hVBF.Scale(VBF_ScaleFactor)

        # ── STYLE ──────────────────────────────────────────────────────────────
        h50.SetFillColor(ROOT.kBlue - 7)
        h50.SetLineColor(ROOT.kBlue - 5)    # darker edge so error bars are visible
        h50.SetMarkerColor(ROOT.kBlue - 5)
        h50.SetMarkerStyle(20)
        h50.SetMarkerSize(0.6)

        hVBF.SetFillColor(ROOT.kRed - 7)
        hVBF.SetLineColor(ROOT.kRed - 5)
        hVBF.SetMarkerColor(ROOT.kRed - 5)
        hVBF.SetMarkerStyle(20)
        hVBF.SetMarkerSize(0.6)

        hSum = h50_nc.Clone("hSum")
        hSum.SetLineColor(ROOT.kBlack)
        hSum.SetLineWidth(2)
        hSum.SetMarkerStyle(20)
        hSum.SetMarkerSize(0.6)
        hSum.SetMarkerColor(ROOT.kBlack)

        # Stack sum for ratio numerator
        hStack_sum = h50.Clone("hStack_sum")
        hStack_sum.Add(hVBF)

        # Ratio: (h50 + hVBF) / h50_nc — should be ~1 if stitching is correct
        hRatio = hStack_sum.Clone("hRatio")
        hRatio.Divide(h50_nc)

        # ── STACK ──────────────────────────────────────────────────────────────
        stack = ROOT.THStack("stack", f"Drell-Yan Stitching Validation: {Year}")
        stack.Add(h50)
        stack.Add(hVBF)

        # ── CANVAS ─────────────────────────────────────────────────────────────
        ROOT.gStyle.SetOptStat(0)
        c = ROOT.TCanvas("c", "Stitching Validation", 800, 750)
        c.cd()

        # Upper pad: main distributions
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.30, 1, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.SetLeftMargin(0.12)
        pad1.SetLogy()
        pad1.Draw()

        # Lower pad: ratio
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.00, 1, 0.30)
        pad2.SetTopMargin(0.03)
        pad2.SetBottomMargin(0.35)
        pad2.SetLeftMargin(0.12)
        pad2.Draw()

        # ── PAD 1 ──────────────────────────────────────────────────────────────
        pad1.cd()
        stack.Draw("HIST E")            # E draws error bars on each component
        stack.GetXaxis().SetLabelSize(0)
        stack.GetYaxis().SetTitle("Events / bin")
        stack.GetYaxis().SetTitleSize(0.06)
        stack.GetYaxis().SetTitleOffset(0.9)
        hSum.Draw("HIST E SAME")        # error bars on reference line too

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.05)
        # latex.DrawLatex(0.12, 0.92, f"DY Stitching: {Year}  |  mjj_thresh = {mjj_threshold:.0f} GeV")

        leg = ROOT.TLegend(0.50, 0.62, 0.90, 0.88)
        leg.SetBorderSize(0)
        leg.AddEntry(h50,  f"DY-M50,  gjj_mass #leq {mjj_threshold:.0f}", "f")
        leg.AddEntry(hVBF, f"DY-VBF,  gjj_mass > {mjj_threshold:.0f}",    "f")
        leg.AddEntry(hSum,  "DY-M50 (no cut, reference)",                  "l")
        leg.Draw()

        # ── PAD 2 ──────────────────────────────────────────────────────────────
        pad2.cd()

        hRatio.SetTitle("")
axis_title = {"gjj_mass": "m_{jj} (GEN) [GeV]", "jj_mass_nominal": "m_{jj} (RECO) [GeV]", "dimuon_mass": "m_{#mu#mu} [GeV]", "dimuon_pt": "p_{T}^{#mu#mu} [GeV]"}.get(plot_var, plot_var)
        hRatio.GetXaxis().SetTitle(axis_title)
        hRatio.GetYaxis().SetTitle("Stack / Ref")
        hRatio.GetYaxis().SetRangeUser(0.5, 1.5)
        hRatio.GetYaxis().SetNdivisions(504)
        hRatio.GetYaxis().CenterTitle()

        hRatio.GetXaxis().SetTitleSize(0.13)
        hRatio.GetXaxis().SetLabelSize(0.11)
        hRatio.GetXaxis().SetTitleOffset(1.1)
        hRatio.GetYaxis().SetTitleSize(0.11)
        hRatio.GetYaxis().SetLabelSize(0.10)
        hRatio.GetYaxis().SetTitleOffset(0.45)

        hRatio.SetLineColor(ROOT.kBlack)
        hRatio.SetLineWidth(2)
        hRatio.SetMarkerStyle(20)
        hRatio.SetMarkerSize(0.8)
        hRatio.Draw("EP")

        # Reference line at 1
        refLine = ROOT.TLine(xmin, 1.0, xmax, 1.0)
        refLine.SetLineColor(ROOT.kRed)
        refLine.SetLineStyle(2)
        refLine.SetLineWidth(2)
        refLine.Draw()

        c.Update()
        c.SaveAs(f"validation/DY_Stiching/{nanoAODv}/stitching_validation_{Year}_{plot_var}.pdf")
        c.SaveAs(f"validation/DY_Stiching/{nanoAODv}/stitching_validation_{Year}_{plot_var}.png")

        # Clean up ROOT objects before next iteration to avoid name clashes
        for obj in [h50, hVBF, h50_nc, hSum, hStack_sum, hRatio, stack, c]:
            obj.Delete()

print("\nAll years done.")