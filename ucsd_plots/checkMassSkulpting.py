import ROOT
import ROOT as rt
import os
import uproot
from typing import Tuple, List, Dict
import pandas as pd


def normalizeRooHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Scale(1/THist.Integral())
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized


def addRooHists(x: rt.RooRealVar,rooHist_l: List[rt.RooDataHist]) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values all added on
    """
    x_name = x.GetName()
    THist = rooHist_l[0].createHistogram(x_name).Clone("clone") # clone it just in case
    print(f"{0}th THist.Integral(): {THist.Integral()}")
    print(f"{0}th rooHist_l.sumEntries(): {rooHist_l[0].sumEntries()}")
    for ix in range(1, len(rooHist_l)):
        THist_ix = rooHist_l[ix].createHistogram(x_name).Clone("clone")
        print(f"{ix}th THist.Integral(): {THist_ix.Integral()}")
        print(f"{ix}th rooHist_l.sumEntries(): {rooHist_l[ix].sumEntries()}")
        THist.Add(THist_ix)
    combinedHist_name = f"combined category of {x_name}"
    # THist.Print("v")
    print(f"roo_hist_combined.Integral(): {THist.Integral()}")
    roo_hist_combined = rt.RooDataHist(combinedHist_name, combinedHist_name, rt.RooArgSet(x), THist) 
    print(f"roo_hist_combined.sumEntries(): {roo_hist_combined.sumEntries()}")
    roo_hist_combined.Print("v")
    return roo_hist_combined


def rebinnHist(mass, rooHist):
    x_name = mass.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") 
    # target_nbins = 100
    rebin_factor = 8
    THist = THist.Rebin(rebin_factor, "hist_rebinned")
    rebinned_rooHist = rt.RooDataHist(rooHist.GetName(), rooHist.GetName(), rt.RooArgSet(mass), THist)
    return rebinned_rooHist


def plotMassSkuplting(mass, rooHist_dict, save_path):
    colors = [rt.kRed, rt.kBlue, rt.kGreen, rt.kMagenta, rt.kOrange] 
    name = "Canvas"
    canvas = rt.TCanvas(name, name, 800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    frame.SetTitle(f"Combined data histograms of all categories")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    color_ix = 0
    for cat_name, roo_hist in rooHist_dict.items():
        color = colors[color_ix % len(colors)]
        color_ix += 1
        name = cat_name
        roo_hist.plotOn(frame, ROOT.RooFit.DrawOption("B"), ROOT.RooFit.FillColor(10), ROOT.RooFit.LineColor(1) , ROOT.RooFit.YErrorSize(0), Name=name, )
        # roo_hist.plotOn(frame, drawOptions="L", Name=name, LineColor=color)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw() 
    canvas.SetTicks(2, 2)
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{save_path}/massSkulpt_ucsd.pdf")
    canvas.SaveAs(f"{save_path}/massSkulpt_ucsd.png")



if __name__ == "__main__":

    plot_save_path = "./plots"
    mass_name = "mh_ggh"
    # mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    mass = rt.RooRealVar(mass_name, mass_name, 120, 115, 135)
    mass.setRange("h_peak", 115, 135 )

    rooHist_dict = {}
    
    for cat_ix in range(5):
        file = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")

        hist = file["w"].obj(f"data_cat{cat_ix}_ggh")
        hist_reduced = hist.reduce(rt.RooFit.CutRange("h-peak"))
        # print(f"RooHist X Range: [{hist_reduced.GetXmin()}, {hist_reduced.GetXmax()}]")
        hist_reduced = rebinnHist(mass, hist_reduced)
        hist_normalized = normalizeRooHist(mass, hist_reduced)
        rooHist_dict[f"cat{cat_ix}"] = hist_normalized


    plotMassSkuplting(mass, rooHist_dict, plot_save_path)


