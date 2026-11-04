#!/usr/bin/env python3
"""
Data/MC control plots and selection-comparison plots of the dimuon, muon, jet, dijet and all ggH-BDT / VBF-DNN
input variables, for every flag-skim selection (modules.selection.FLAG_SKIM_SELECTIONS) of a v2 flag-skim
stage-1 ntuple (do_flag_skim, docs/flag_skim.md). One pass fills all histograms (selection x phase space x
variable x sample group) into a pickle; --plot-only redraws from it.

Phase spaces (blinded: no 115-135 GeV): zpeak_incl (70-110 GeV, no category), zpeak_ggh / zpeak_vbf and
hsb_ggh / hsb_vbf (h-sidebands 110-115 + 135-150 GeV) with the cutbased categories of
modules.selection.applyRegionCatCuts (vbf: mjj>400, dEta>2.5, jet1 pT>35, b-veto; ggh: b-veto, not vbf).
MC weight = wgt_nominal (incl. Z-pT) x SF(WP)/SF(Medium) for Tight/Loose selections, as stage-2.
Outputs: <out>/hists.pkl, <out>/stack/<sel>/<ps>/<var>.{pdf,png} (data vs stacked MC + ratio) and
<out>/compare/<ps>/<var>.{pdf,png} (top: data/MC per selection; bottom: data shape / official shape).

How to run (worktree root, default pixi env; ~5-10 min fill with 32 processes, ~5 min plotting):
  ./run_in_pixi.sh default python scripts/flag_skim_control_plots.py --input <label>/stage1_output -y <year> \
      --out <dir> [--selections ...] [--stack-selections official looseIDIP] [--plot-only]
Example:
  ./run_in_pixi.sh default python scripts/flag_skim_control_plots.py \
      --input /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit/stage1_output \
      -y 2024 --out validation/flag_skim/2024_v2/control_plots
"""

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from modules.sample_config import get_bkg_sig_dicts
from modules.selection import FLAG_SKIM_ID_SF, FLAG_SKIM_SELECTIONS, _round_for_cut

# the 9 valid selections (every one has an electron veto; lepton veto only with IP)
DEFAULT_SELECTIONS = ["looseID", "looseIDIP", "looseIDlepIPVeto", "official", "ipOnly", "lepIPVeto",
                      "tightID", "tightIDIP", "tightIDlepIPVeto"]
PHASE_SPACES = ["zpeak_incl", "zpeak_ggh", "zpeak_vbf", "hsb_ggh", "hsb_vbf"]

# name: (column, nbins, lo, hi, ROOT x label, log y); values outside [lo, hi] go to the edge bins
VARIABLES = {
    "dimuon_mass": ("dimuon_mass", 40, 70, 150, "m_{#mu#mu} [GeV]", False),
    "dimuon_pt": ("dimuon_pt", 30, 0, 300, "p_{T}^{#mu#mu} [GeV]", True),
    "dimuon_pt_log": ("dimuon_pt_log", 35, -1, 6, "log(p_{T}^{#mu#mu}/GeV)", False),
    "dimuon_rapidity": ("dimuon_rapidity", 25, -2.5, 2.5, "y_{#mu#mu}", False),
    "dimuon_eta": ("dimuon_eta", 32, -8, 8, "#eta_{#mu#mu}", False),
    "dimuon_phi": ("dimuon_phi", 32, -3.2, 3.2, "#phi_{#mu#mu}", False),
    "dimuon_ebe_mass_res": ("dimuon_ebe_mass_res", 30, 0, 6, "#sigma_{m_{#mu#mu}} [GeV]", False),
    "dimuon_ebe_mass_res_rel": ("dimuon_ebe_mass_res_rel", 40, 0, 0.04, "#sigma_{m_{#mu#mu}}/m_{#mu#mu}", False),
    "dimuon_cos_theta_cs": ("dimuon_cos_theta_cs", 20, -1, 1, "cos#theta_{CS}", False),
    "dimuon_phi_cs": ("dimuon_phi_cs", 32, -3.2, 3.2, "#phi_{CS}", False),
    "mu1_pt": ("mu1_pt", 36, 20, 200, "p_{T}(#mu_{1}) [GeV]", True),
    "mu2_pt": ("mu2_pt", 26, 20, 150, "p_{T}(#mu_{2}) [GeV]", True),
    "mu1_eta": ("mu1_eta", 24, -2.4, 2.4, "#eta(#mu_{1})", False),
    "mu2_eta": ("mu2_eta", 24, -2.4, 2.4, "#eta(#mu_{2})", False),
    "mu1_phi": ("mu1_phi", 32, -3.2, 3.2, "#phi(#mu_{1})", False),
    "mu2_phi": ("mu2_phi", 32, -3.2, 3.2, "#phi(#mu_{2})", False),
    "mu1_pt_over_mass": ("mu1_pt_over_mass", 40, 0, 2, "p_{T}(#mu_{1})/m_{#mu#mu}", False),
    "mu2_pt_over_mass": ("mu2_pt_over_mass", 30, 0, 1.5, "p_{T}(#mu_{2})/m_{#mu#mu}", False),
    "jet1_pt": ("jet1_pt_nominal", 35, 25, 375, "p_{T}(j_{1}) [GeV]", True),
    "jet2_pt": ("jet2_pt_nominal", 35, 25, 200, "p_{T}(j_{2}) [GeV]", True),
    "jet1_eta": ("jet1_eta_nominal", 47, -4.7, 4.7, "#eta(j_{1})", False),
    "jet2_eta": ("jet2_eta_nominal", 47, -4.7, 4.7, "#eta(j_{2})", False),
    "jet1_phi": ("jet1_phi_nominal", 32, -3.2, 3.2, "#phi(j_{1})", False),
    "jet2_phi": ("jet2_phi_nominal", 32, -3.2, 3.2, "#phi(j_{2})", False),
    "jj_mass": ("jj_mass_nominal", 50, 0, 2500, "m_{jj} [GeV]", True),
    "jj_mass_log": ("jj_mass_log_nominal", 30, 2.5, 8.5, "log(m_{jj}/GeV)", False),
    "jj_dEta": ("jj_dEta_nominal", 36, 0, 9, "#Delta#eta_{jj}", False),
    "jj_dPhi": ("jj_dPhi_nominal", 32, 0, 3.2, "#Delta#phi_{jj}", False),
    "jj_pt": ("jj_pt_nominal", 30, 0, 300, "p_{T}^{jj} [GeV]", True),
    "njets": ("njets_nominal", 8, -0.5, 7.5, "N_{jets}", True),
    "nsoftjets5": ("nsoftjets5_nominal", 10, -0.5, 9.5, "N_{soft jets}(p_{T}>5 GeV)", True),
    "htsoft2": ("htsoft2_nominal", 40, 0, 200, "H_{T}^{soft}(p_{T}>2 GeV) [GeV]", True),
    "rpt": ("rpt_nominal", 20, 0, 1, "R_{p_{T}}", False),
    "ll_zstar_log": ("ll_zstar_log_nominal", 30, -8, 4, "log z*_{#mu#mu}", False),
    "mmj_min_dEta": ("mmj_min_dEta_nominal", 32, 0, 8, "min #Delta#eta(#mu#mu, j)", False),
    "mmj_min_dPhi": ("mmj_min_dPhi_nominal", 32, 0, 3.2, "min #Delta#phi(#mu#mu, j)", False),
    "pt_centrality": ("pt_centrality_nominal", 40, -10, 10, "p_{T} centrality", False),
    "zeppenfeld": ("zeppenfeld_nominal", 40, -5, 5, "Zeppenfeld z*", False),
    "nBtagLoose": ("nBtagLoose_nominal", 5, -0.5, 4.5, "N_{b-tag}^{loose}", True),
    "PV_npvsGood": ("PV_npvsGood", 40, 0, 80, "N_{PV}^{good}", False),
}
MASS_RANGE = {"zpeak": (70, 110), "hsb": (110, 150)}
# plotting-only rebin of the low-statistics phase spaces (largest factor <= this that divides nbins)
REBIN = {"zpeak_vbf": 2, "hsb_ggh": 2, "hsb_vbf": 4}
CAT_COLS = ["nBtagLoose_nominal", "nBtagMedium_nominal", "njets_nominal", "jet1_pt_nominal", "jj_mass_nominal",
            "jj_dEta_nominal"]
MC_ORDER = ["VV", "EWK", "TT", "DY"]  # stack bottom -> top
SIG_GROUPS = ["GGH", "VBF"]


def mass_edges(ps):
    lo, hi = MASS_RANGE[ps.split("_")[0]]
    return np.linspace(lo, hi, 41)


def edges_for(ps, var):
    if var == "dimuon_mass":
        return mass_edges(ps)
    _, nb, lo, hi, _, _ = VARIABLES[var]
    return np.linspace(lo, hi, nb + 1)


def phase_space_masks(df):
    m = df["dimuon_mass"].to_numpy()
    nbl, nbm, nj = (df[c].fillna(-1).to_numpy() for c in CAT_COLS[:3])
    bveto = ~((nbl >= 2) | ((nbm >= 1) & (nj >= 2)))
    vbf = ((df["jj_mass_nominal"].to_numpy() > 400) & (_round_for_cut(df["jj_dEta_nominal"].to_numpy()) > 2.5)
           & (df["jet1_pt_nominal"].to_numpy() > 35))
    zpk = (m >= 70) & (m < 110)
    hsb = ((m >= 110) & (m < 115)) | ((m >= 135) & (m < 150))
    return {"zpeak_incl": zpk, "zpeak_ggh": zpk & bveto & ~vbf, "zpeak_vbf": zpk & bveto & vbf,
            "hsb_ggh": hsb & bveto & ~vbf, "hsb_vbf": hsb & bveto & vbf}


def fill_task(args):
    files, group, selections = args
    is_data = group == "data"
    names = set(pq.read_schema(files[0]).names)
    cols = {c for c, *_ in VARIABLES.values()} | set(CAT_COLS) | {"dimuon_mass", "wgt_nominal", "muIDIso_sf_MediumID",
                                                                   "muIDIso_sf_TightID", "muIDIso_sf_LooseID"}
    cols |= {c for c in names if c.startswith(("pass_", "pair_", "nMuons_"))}
    cols = [c for c in cols if c in names]
    out = {}
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        if df.empty:
            continue
        w0 = np.ones(len(df)) if is_data else df["wgt_nominal"].fillna(0.0).to_numpy()
        ps_masks = phase_space_masks(df)
        values = {v: df[c].to_numpy(dtype=float) for v, (c, *_) in VARIABLES.items()}
        for sel in selections:
            smask = FLAG_SKIM_SELECTIONS[sel](df).fillna(False).to_numpy(dtype=bool)
            w = w0
            wp = FLAG_SKIM_ID_SF.get(sel)
            if wp and not is_data:
                nom = df["muIDIso_sf_MediumID"].to_numpy()
                w = w0 * np.where(nom != 0, df[f"muIDIso_sf_{wp}"].to_numpy() / np.where(nom != 0, nom, 1.0), 1.0)
            for ps, pmask in ps_masks.items():
                mask = smask & pmask
                if not mask.any():
                    continue
                ww = w[mask]
                for var, x in values.items():
                    e = edges_for(ps, var)
                    xv = x[mask]
                    ok = np.isfinite(xv)
                    xv = np.clip(xv[ok], e[0], e[-1] - 1e-9 * (e[-1] - e[0]))  # fold under/overflow
                    sw, _ = np.histogram(xv, e, weights=ww[ok])
                    sw2, _ = np.histogram(xv, e, weights=ww[ok] ** 2)
                    key = (sel, ps, var)
                    if key in out:
                        out[key][0] += sw
                        out[key][1] += sw2
                    else:
                        out[key] = [sw, sw2]
    return group, out


def fill(args):
    base = f"{args.input}/{args.year}/compacted"
    bkg, sig, _ = get_bkg_sig_dicts(yaml_path=args.sample_config, year=args.year)
    groups = {"data": sorted(os.path.basename(d) for d in glob.glob(f"{base}/data*"))}
    groups.update({g: s for g, s in {**sig, **bkg}.items() if g != "DYVBF" and s})
    tasks = []
    for g, samples in groups.items():
        for s in samples:
            files = sorted(glob.glob(f"{base}/{s}/**/*.parquet", recursive=True))
            tasks += [(files[i:i + 25], g, args.selections) for i in range(0, len(files), 25)]
    print(f"{len(tasks)} tasks over {sum(len(t[0]) for t in tasks)} files; groups {list(groups)}", flush=True)
    hists = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (g, out) in enumerate(ex.map(fill_task, tasks)):
            for (sel, ps, var), (sw, sw2) in out.items():
                k = (sel, ps, var, g)
                if k in hists:
                    hists[k][0] += sw
                    hists[k][1] += sw2
                else:
                    hists[k] = [sw, sw2]
            if i % 20 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)
    return {"hists": hists, "groups": list(groups), "selections": args.selections, "year": args.year}


# ---------------------------------------------------------------- plotting (PyROOT)
def plot(store, out, stack_sels):
    import ROOT
    from modules.root_2dColorProfile import set_gradient_style

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    set_gradient_style()
    ROOT.TGaxis.SetMaxDigits(5)  # no x10^n exponent clashing with the CMS label
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    hists, sels, year = store["hists"], store["selections"], store["year"]
    colors = {"DY": ROOT.kOrange - 3, "TT": ROOT.kAzure - 4, "EWK": ROOT.kViolet - 4, "VV": ROOT.kGreen - 6}
    id_color = {"loose": ROOT.kBlue + 1, "medium": ROOT.kBlack, "tight": ROOT.kRed + 1}

    def th1(name, sel, ps, var, group):
        e = edges_for(ps, var)
        h = ROOT.TH1D(name, "", len(e) - 1, e)
        h.SetDirectory(0)
        sw, sw2 = hists.get((sel, ps, var, group), [np.zeros(len(e) - 1)] * 2)
        for i in range(len(e) - 1):
            h.SetBinContent(i + 1, sw[i])
            h.SetBinError(i + 1, np.sqrt(sw2[i]))
        k = REBIN.get(ps, 1)
        while k > 1 and (len(e) - 1) % k:
            k -= 1
        return h.Rebin(k) if k > 1 else h

    def graph(h):
        """Points of a ratio histogram, skipping empty/undefined bins (drawn as gaps, not drops to 0)."""
        g = ROOT.TGraph()
        for i in range(1, h.GetNbinsX() + 1):
            y = h.GetBinContent(i)
            if np.isfinite(y) and y > 0:
                g.SetPoint(g.GetN(), h.GetBinCenter(i), y)
        return g

    def style_of(sel):
        idk = "loose" if sel.startswith("loose") else "tight" if sel.startswith("tight") else "medium"
        variant = "lepveto" if "lepIPVeto" in sel else "ip" if (sel.endswith("IP") or sel == "ipOnly") else "noip"
        return id_color[idk], {"noip": 2, "ip": 1, "lepveto": 3}[variant], {"noip": 24, "ip": 20, "lepveto": 21}[variant]

    def save(c, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        c.SaveAs(path + ".pdf")
        c.SaveAs(path + ".png")

    def label(text, x=0.16, y=0.93, size=0.045):
        t = ROOT.TLatex()
        t.SetNDC()
        t.SetTextSize(size)
        t.DrawLatex(x, y, text)
        return t

    ps_title = {"zpeak_incl": "Z peak, inclusive", "zpeak_ggh": "Z peak, ggH cat.", "zpeak_vbf": "Z peak, VBF cat.",
                "hsb_ggh": "H sidebands, ggH cat.", "hsb_vbf": "H sidebands, VBF cat."}
    for ps in PHASE_SPACES:
        for var, (_, _, _, _, xlab, logy) in VARIABLES.items():
            # --- data vs stacked MC, one per selection
            for sel in stack_sels:
                c = ROOT.TCanvas(f"c_{sel}_{ps}_{var}", "", 700, 700)
                top = ROOT.TPad("t", "", 0, 0.3, 1, 1)
                bot = ROOT.TPad("b", "", 0, 0, 1, 0.3)
                top.SetBottomMargin(0.02)
                bot.SetTopMargin(0.03)
                bot.SetBottomMargin(0.35)
                for p in (top, bot):
                    p.SetLeftMargin(0.14)
                    p.Draw()
                top.cd()
                top.SetLogy(logy)
                data = th1(f"d_{sel}{ps}{var}", sel, ps, var, "data")
                stack = ROOT.THStack(f"s_{sel}{ps}{var}", "")
                mc_tot = None
                leg = ROOT.TLegend(0.62, 0.55, 0.9, 0.88)
                leg.SetBorderSize(0)
                leg.AddEntry(data, f"Data ({data.Integral():.0f})", "pe")
                mcs = []
                for g in MC_ORDER:
                    h = th1(f"{g}_{sel}{ps}{var}", sel, ps, var, g)
                    h.SetFillColor(colors[g])
                    h.SetLineColor(colors[g])
                    stack.Add(h)
                    mcs.append((g, h))
                    if mc_tot is None:
                        mc_tot = h.Clone(f"tot_{sel}{ps}{var}")
                    else:
                        mc_tot.Add(h)
                for g, h in reversed(mcs):
                    leg.AddEntry(h, f"{g} ({h.Integral():.4g})", "f")
                sig = th1(f"sig_{sel}{ps}{var}", sel, ps, var, SIG_GROUPS[0])
                sig.Add(th1(f"sig2_{sel}{ps}{var}", sel, ps, var, SIG_GROUPS[1]))
                sig_int = sig.Integral()
                sig.Scale(100.0)
                sig.SetLineColor(ROOT.kRed)
                sig.SetLineWidth(2)
                leg.AddEntry(sig, f"ggH+VBF #times100 ({sig_int:.3g})", "l")
                ymax = max(data.GetMaximum(), mc_tot.GetMaximum(), sig.GetMaximum())
                stack.SetMaximum(ymax * (50 if logy else 1.5))
                stack.SetMinimum(max(1e-2, 0.5 * min([b for b in (mc_tot.GetBinContent(i) for i in range(1, mc_tot.GetNbinsX() + 1)) if b > 0] or [1])) if logy else 0)
                stack.Draw("HIST")
                stack.GetXaxis().SetLabelSize(0)
                stack.GetYaxis().SetTitle("Events / bin")
                unc = mc_tot.Clone(f"unc_{sel}{ps}{var}")
                unc.SetFillColor(ROOT.kGray + 2)
                unc.SetFillStyle(3354)
                unc.SetMarkerSize(0)
                unc.Draw("E2 SAME")
                sig.Draw("HIST SAME")
                data.SetMarkerStyle(20)
                data.SetMarkerSize(0.8)
                data.Draw("PE SAME")
                leg.Draw()
                t1 = label(f"#bf{{CMS}} #it{{Work in progress}}   {year}", size=0.042)
                t2 = label(f"{sel}, {ps_title[ps]}", y=0.85, size=0.038)
                bot.cd()
                r = data.Clone(f"r_{sel}{ps}{var}")
                r.Divide(mc_tot)
                ru = mc_tot.Clone(f"ru_{sel}{ps}{var}")
                for i in range(1, ru.GetNbinsX() + 1):
                    b = ru.GetBinContent(i)
                    ru.SetBinError(i, ru.GetBinError(i) / b if b > 0 else 0)
                    ru.SetBinContent(i, 1.0)
                ru.SetFillColor(ROOT.kGray + 1)
                ru.SetFillStyle(1001)
                ru.SetMinimum(0.5)
                ru.SetMaximum(1.5)
                ru.GetYaxis().SetTitle("Data/MC")
                ru.GetXaxis().SetTitle(xlab)
                for ax, s in ((ru.GetXaxis(), 0.12), (ru.GetYaxis(), 0.11)):
                    ax.SetTitleSize(s)
                    ax.SetLabelSize(0.1)
                ru.GetYaxis().SetTitleOffset(0.55)
                ru.GetYaxis().SetNdivisions(505)
                ru.Draw("E2")
                line = ROOT.TLine(ru.GetXaxis().GetXmin(), 1, ru.GetXaxis().GetXmax(), 1)
                line.SetLineStyle(2)
                line.Draw()
                r.Draw("PE SAME")
                save(c, f"{out}/stack/{sel}/{ps}/{var}")
                del t1, t2
            # --- selection comparison
            c = ROOT.TCanvas(f"cc_{ps}_{var}", "", 700, 700)
            top = ROOT.TPad("t", "", 0, 0.42, 1, 1)
            bot = ROOT.TPad("b", "", 0, 0, 1, 0.42)
            top.SetBottomMargin(0.02)
            bot.SetTopMargin(0.03)
            bot.SetBottomMargin(0.28)
            for p in (top, bot):
                p.SetLeftMargin(0.14)
                p.Draw()
            top.SetTopMargin(0.30)  # legend band above the frame
            ref = th1(f"ref_{ps}{var}", "official", ps, var, "data")
            ref_int = ref.Integral()
            leg = ROOT.TLegend(0.14, 0.715, 0.95, 0.90)
            leg.SetNColumns(3)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.04)
            fr_top = ref.Clone(f"frt_{ps}{var}")
            fr_top.Reset()
            fr_top.SetMinimum(0.5)
            fr_top.SetMaximum(1.5)
            fr_top.GetYaxis().SetTitle("Data / MC")
            fr_top.GetYaxis().SetTitleSize(0.07)
            fr_top.GetYaxis().SetTitleOffset(0.85)
            fr_top.GetYaxis().SetLabelSize(0.06)
            fr_top.GetXaxis().SetLabelSize(0)
            fr_bot = ref.Clone(f"frb_{ps}{var}")
            fr_bot.Reset()
            fr_bot.SetMinimum(0.85)
            fr_bot.SetMaximum(1.15)
            fr_bot.GetYaxis().SetTitle("data shape / official")
            fr_bot.GetXaxis().SetTitle(xlab)
            for ax, sz in ((fr_bot.GetXaxis(), 0.08), (fr_bot.GetYaxis(), 0.065)):
                ax.SetTitleSize(sz)
                ax.SetLabelSize(0.065)
            fr_bot.GetYaxis().SetTitleOffset(0.95)
            fr_bot.GetYaxis().SetNdivisions(505)
            top.cd()
            fr_top.Draw("AXIS")
            bot.cd()
            fr_bot.Draw("AXIS")
            keep = []
            for i, sel in enumerate(sels):
                col, ls, ms = style_of(sel)
                d = th1(f"cd_{sel}{ps}{var}", sel, ps, var, "data")
                mc = None
                for g in MC_ORDER:
                    h = th1(f"cm_{g}{sel}{ps}{var}", sel, ps, var, g)
                    if mc is None:
                        mc = h
                    else:
                        mc.Add(h)
                r = d.Clone(f"cr_{sel}{ps}{var}")
                r.Divide(mc)
                s = d.Clone(f"cs_{sel}{ps}{var}")
                if s.Integral() > 0 and ref_int > 0:
                    s.Scale(ref_int / s.Integral())
                s.Divide(ref)
                gr, gs = graph(r), graph(s)
                for g in (gr, gs):
                    g.SetLineColor(col)
                    g.SetMarkerColor(col)
                    g.SetLineStyle(ls)
                    g.SetMarkerStyle(ms)
                    g.SetMarkerSize(0.7)
                    g.SetLineWidth(2)
                top.cd()
                if gr.GetN():
                    gr.Draw("LP SAME")
                leg.AddEntry(gr, sel, "lp")
                bot.cd()
                if gs.GetN():
                    gs.Draw("LP SAME")
                keep += [gr, gs]
            lines = []
            for p in (top, bot):
                p.cd()
                ln = ROOT.TLine(ref.GetXaxis().GetXmin(), 1, ref.GetXaxis().GetXmax(), 1)
                ln.SetLineStyle(2)
                ln.SetLineColor(ROOT.kGray + 2)
                ln.Draw()
                lines.append(ln)
            top.cd()
            leg.Draw()
            t1 = label(f"#bf{{CMS}} #it{{Work in progress}}   {year}   {ps_title[ps]}", x=0.14, y=0.94, size=0.05)
            save(c, f"{out}/compare/{ps}/{var}")
            del t1
        print(f"plotted {ps}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="<label>/stage1_output")
    ap.add_argument("-y", "--year", default="2024")
    ap.add_argument("--selections", nargs="+", default=DEFAULT_SELECTIONS, choices=sorted(FLAG_SKIM_SELECTIONS))
    ap.add_argument("--stack-selections", nargs="+", default=["official", "looseIDIP"])
    ap.add_argument("--sample_config", default=os.environ.get("SAMPLE_CONFIG") or "configs/samples/samples.yaml")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--plot-only", action="store_true", help="redraw from <out>/hists.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pkl = f"{args.out}/hists.pkl"
    if args.plot_only:
        with open(pkl, "rb") as f:
            store = pickle.load(f)
    else:
        if "official" not in args.selections:
            ap.error("--selections must include official (comparison reference)")
        store = fill(args)
        with open(pkl, "wb") as f:
            pickle.dump(store, f)
        print(f"wrote {pkl}", flush=True)
    plot(store, args.out, args.stack_selections)


if __name__ == "__main__":
    main()
