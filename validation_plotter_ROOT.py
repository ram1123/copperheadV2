import ROOT
import awkward as ak
import dask_awkward as dak
import numpy as np
import json
import argparse
import os

def setTDRStyle():
  ROOT.gStyle.SetCanvasBorderMode(0);
  ROOT.gStyle.SetCanvasColor(0);
  ROOT.gStyle.SetCanvasDefH(600);
  ROOT.gStyle.SetCanvasDefW(600);
  ROOT.gStyle.SetCanvasDefX(0);
  ROOT.gStyle.SetCanvasDefY(0);

  ROOT.gStyle.SetPadBorderMode(0);
  ROOT.gStyle.SetPadColor(0); 
  ROOT.gStyle.SetPadGridX(0);
  ROOT.gStyle.SetPadGridY(0);
  ROOT.gStyle.SetGridColor(0);
  ROOT.gStyle.SetGridStyle(3);
  ROOT.gStyle.SetGridWidth(1);

  ROOT.gStyle.SetFrameBorderMode(0);
  ROOT.gStyle.SetFrameBorderSize(1);
  ROOT.gStyle.SetFrameFillColor(0);
  ROOT.gStyle.SetFrameFillStyle(0);
  ROOT.gStyle.SetFrameLineColor(1);
  ROOT.gStyle.SetFrameLineStyle(1);
  ROOT.gStyle.SetFrameLineWidth(1);
  ROOT.gStyle.SetHistLineColor(1);
  ROOT.gStyle.SetHistLineStyle(0);
  ROOT.gStyle.SetHistLineWidth(1);

  ROOT.gStyle.SetEndErrorSize(2);
  ROOT.gStyle.SetFuncColor(2);
  ROOT.gStyle.SetFuncStyle(1);
  ROOT.gStyle.SetFuncWidth(1);
  ROOT.gStyle.SetOptDate(0);
  
  ROOT.gStyle.SetOptFile(0);
  ROOT.gStyle.SetOptStat(0);
  ROOT.gStyle.SetStatColor(0); 
  ROOT.gStyle.SetStatFont(42);
  ROOT.gStyle.SetStatFontSize(0.04);
  ROOT.gStyle.SetStatTextColor(1);
  ROOT.gStyle.SetStatFormat("6.4g");
  ROOT.gStyle.SetStatBorderSize(1);
  ROOT.gStyle.SetStatH(0.1);
  ROOT.gStyle.SetStatW(0.15);

  ROOT.gStyle.SetPadTopMargin(0.07);
  ROOT.gStyle.SetPadBottomMargin(0.13);
  ROOT.gStyle.SetPadLeftMargin(0.12);
  ROOT.gStyle.SetPadRightMargin(0.05);

  ROOT.gStyle.SetOptTitle(0);
  ROOT.gStyle.SetTitleFont(42);
  ROOT.gStyle.SetTitleColor(1);
  ROOT.gStyle.SetTitleTextColor(1);
  ROOT.gStyle.SetTitleFillColor(10);
  ROOT.gStyle.SetTitleFontSize(0.05);

  ROOT.gStyle.SetTitleColor(1, "XYZ");
  ROOT.gStyle.SetTitleFont(42, "XYZ");
  ROOT.gStyle.SetTitleSize(0.05, "XYZ");
  ROOT.gStyle.SetTitleXOffset(0.9);
  ROOT.gStyle.SetTitleYOffset(1.05);
 
  ROOT.gStyle.SetLabelColor(1, "XYZ");
  ROOT.gStyle.SetLabelFont(42, "XYZ");
  ROOT.gStyle.SetLabelOffset(0.007, "XYZ");
  ROOT.gStyle.SetLabelSize(0.04, "XYZ");

  ROOT.gStyle.SetAxisColor(1, "XYZ");
  ROOT.gStyle.SetStripDecimals(1); 
  ROOT.gStyle.SetTickLength(0.025, "XYZ");
  ROOT.gStyle.SetNdivisions(510, "XYZ");
  ROOT.gStyle.SetPadTickX(1); 
  ROOT.gStyle.SetPadTickY(1);

  ROOT.gStyle.SetOptLogx(0);
  ROOT.gStyle.SetOptLogy(0);
  ROOT.gStyle.SetOptLogz(0);

  ROOT.gStyle.SetPaperSize(20.,20.);
  ROOT.gStyle.SetPaintTextFormat(".2f");

def CMS_lumi( pad,  lumi,  up = False,  skipPreliminary = True, reduceSize = False, offset = 0,offsetLumi = 0):
  latex2 = ROOT.TLatex();
  latex2.SetNDC();
  latex2.SetTextSize(0.6*pad.GetTopMargin());
  latex2.SetTextFont(42);
  latex2.SetTextAlign(31);
  if(reduceSize):
    latex2.SetTextSize(0.5*pad.GetTopMargin());
  
  if(lumi != ""):
    latex2.DrawLatex(0.94+offsetLumi, 0.95,(lumi+" fb^{-1} (13 TeV)"));
  else:
    latex2.DrawLatex(0.88+offsetLumi, 0.95,(lumi+"(13 TeV)"));

  if(up):
    latex2.SetTextSize(0.65*pad.GetTopMargin());
    if(reduceSize):
      latex2.SetTextSize(0.5*pad.GetTopMargin());
    latex2.SetTextFont(62);
    latex2.SetTextAlign(11);    
    latex2.DrawLatex(0.15+offset, 0.95, "CMS");
  else:
    latex2.SetTextSize(0.6*pad.GetTopMargin());
    if(reduceSize):
      latex2.SetTextSize(0.45*pad.GetTopMargin());
    elif(reduceSize == 2):
      latex2.SetTextSize(0.40*pad.GetTopMargin());

    latex2.SetTextFont(62);
    latex2.SetTextAlign(11);    
    latex2.DrawLatex(0.175+offset, 0.86, "CMS");

  if(not skipPreliminary):
    
    if(up):
      latex2.SetTextSize(0.55*pad.GetTopMargin());
      latex2.SetTextFont(52);
      latex2.SetTextAlign(11);
      latex2.DrawLatex(0.235+offset, 0.95, "Preliminary");
    
    else:
      latex2.SetTextSize(0.6*pad.GetTopMargin());
      if(reduceSize):
          latex2.SetTextSize(0.45*pad.GetTopMargin());
      latex2.SetTextFont(52);
      latex2.SetTextAlign(11);    
      if(reduceSize):
          latex2.DrawLatex(0.235+offset, 0.86, "Preliminary");
      else:
          latex2.DrawLatex(0.28+offset, 0.86, "Preliminary");

def reweightROOTH(hist, weight: float):
    """
    reweight the histogram values and its errors
    the given weight value
    """
    for idx in range(1, hist.GetNbinsX()+1):
        hist.SetBinContent(idx, hist.GetBinContent(idx)*weight)
        hist.SetBinError(idx, hist.GetBinError(idx)*weight)
    return
    

# real process arrangement
group_data_processes = ["data_A", "data_B", "data_C", "data_D",]
# group_DY_processes = ["dy_M-100To200", "dy_M-50"] # dy_M-50 is not used in ggH BDT training input
group_DY_processes = ["dy_M-100To200"]
group_Top_processes = ["ttjets_dl", "ttjets_sl"]
group_Ewk_processes = []
group_VV_processes = []# diboson
group_ggH_processes = ["ggh_powheg"]
group_VBF_processes = ["vbf_powheg"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    default=['jet'],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/test_full/f0_1",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./validation/figs/ROOT/",
    action="store",
    help="save path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    available_processes = []
    # take data
    data_samples = args.data_samples
    if len(data_samples) >0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")
    # take bkg
    bkg_samples = args.bkg_samples
    if len(bkg_samples) >0:
        for bkg_sample in bkg_samples:
            if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                available_processes.append("dy_M-50")
                available_processes.append("dy_M-100To200")
            elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                available_processes.append("ttjets_dl")
                available_processes.append("ttjets_sl")
            elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                available_processes.append("st_tw_top")
                available_processes.append("st_tw_antitop")
            elif bkg_sample.upper() == "DB": # enforce upper case to prevent confusion
                available_processes.append("ww_2l2nu")
                available_processes.append("wz_3lnu")
                available_processes.append("wz_2l2q")
                available_processes.append("wz_1l1nu2q")
                available_processes.append("zz")
            elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                available_processes.append("ewk_lljj_mll50_mjj120")
            else:
                print(f"unknown background {bkg_sample} was given!")
        
    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) >0:
        for sig_sample in sig_samples:
            if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                available_processes.append("ggh_powheg")
            elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                available_processes.append("vbf_powheg")
            else:
                print(f"unknown signal {sig_sample} was given!")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    variables2plot = []
    if len(args.variables) == 0:
        print("no variables to plot!")
        raise ValueError
    for particle in args.variables:
        if ("mu" in particle) or ("jet" in particle):
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}")
                variables2plot.append(f"{particle}2_{kinematic}")
        elif "dimuon" in particle:
            variables2plot.append(f"{particle}_{mass}")
            for kinematic in kinematic_vars:
                variables2plot.append(f"{particle}_{kinematic}")
        else:
            print(f"Unsupported variable: {particle} is given!")
    print(f"variables2plot: {variables2plot}")
    #Plotting part
    setTDRStyle()
    canvas = ROOT.TCanvas("canvas","",600,750);
    canvas.cd();
    
    pad = ROOT.TPad("pad","pad",0,0.,1,1);
    pad.SetFillColor(0);
    pad.SetFillStyle(0);
    pad.SetTickx(1);
    pad.SetTicky(1);
    pad.SetBottomMargin(0.3);
    pad.SetRightMargin(0.06);
    pad.Draw();
    pad.cd();

    # obtain plot settings from config file
    with open("./histogram/plot_settings.json", "r") as file:
        plot_settings = json.load(file)


    # available_processes = ["dy_M-100To200", "dy_M-50","data_A", "data_B", "data_C", "data_D", "ttjets_dl", "ttjets_sl", "ggh_powheg","vbf_powheg"]
    
    
    fraction_weight = 1.0 # to be used later in reweightROOTH after all histograms are filled
    # var = "jet1_pt"
    for var in variables2plot:
        if var not in plot_settings.keys():
            print(f"variable {var} not configured in plot settings!")
            continue
        binning = np.linspace(*plot_settings[var]["binning_linspace"])
        group_data_hists = []
        group_DY_hists = []
        group_Top_hists = []
        group_Ewk_hists = []
        group_VV_hists = []
        group_other_hists = []
        group_ggH_hists = [] # there should only be one ggH histogram, but making a list for consistency
        group_VBF_hists = [] # there should only be one VBF histogram, but making a list for consistency
        
        # group_other_hists = [] # histograms not belonging to any other group
        
        for process in available_processes:
            print(f"process: {process}")
            full_load_path = args.load_path+f"/{process}/*/*.parquet"
            events = dak.from_parquet(full_load_path) 
            # obtain fraction weight, this should be the same for each process 
            fraction_weight = 1/events.fraction[0].compute()
            print(f"fraction_weight: {fraction_weight}")
            # obtain the category selection
            vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
            region = events.h_sidebands | events.h_peak
            btag_cut =(events.nBtagLoose >= 2) | (events.nBtagMedium >= 1)
            category_selection = (
                ~vbf_cut & # we're interested in ggH category
                region &
                btag_cut # btag cut is for VH and ttH categories
            ).compute()
            category_selection = ak.to_numpy(category_selection) # this will be multiplied with weights
            weights = 1*category_selection
            np_hist, _ = np.histogram(events[var].compute(), bins=binning, weights = weights)
            # print(f"max(np_hist): {max(np_hist)}")
            # print(f"(np_hist): {(np_hist)}")
            # print(f"(np_hist): {np.any(np_hist==0)}")
            
            if process in group_data_processes:
                print("data activated")
                var_hist_data = ROOT.TH1F( var+'_hist_data', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_data.SetBinContent(1+idx, np_hist[idx])
                group_data_hists.append(var_hist_data)
            #-------------------------------------------------------
            elif process in group_DY_processes:
                print("DY activated")
                var_hist_DY = ROOT.TH1F( var+'_hist_DY', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_DY.SetBinContent(1+idx, np_hist[idx])
                group_DY_hists.append(var_hist_DY)
            #-------------------------------------------------------
            elif process in group_Top_processes:
                print("top activated")
                var_hist_Top = ROOT.TH1F( var+'_hist_Top', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_Top.SetBinContent(1+idx, np_hist[idx])
                group_Top_hists.append(var_hist_Top)
            #-------------------------------------------------------
            elif process in group_Ewk_processes:
                print("Ewk activated")
                var_hist_Ewk = ROOT.TH1F( var+'_hist_Ewk', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_Ewk.SetBinContent(1+idx, np_hist[idx])
                group_Ewk_hists.append(var_hist_Ewk)
            #-------------------------------------------------------
            elif process in group_VV_processes:
                print("VV activated")
                var_hist_VV = ROOT.TH1F( var+'_hist_VV', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_VV.SetBinContent(1+idx, np_hist[idx])
                group_VV_hists.append(var_hist_VV)
            #-------------------------------------------------------
            elif process in group_ggH_processes:
                print("ggH activated")
                var_hist_ggH = ROOT.TH1F( var+'_hist_ggH', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_ggH.SetBinContent(1+idx, np_hist[idx])
                group_ggH_hists.append(var_hist_ggH)
            #-------------------------------------------------------
            elif process in group_VBF_processes:
                print("VBF activated")
                var_hist_VBF = ROOT.TH1F( var+'_hist_VBF', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_VBF.SetBinContent(1+idx, np_hist[idx])
                group_VBF_hists.append(var_hist_VBF)
            #-------------------------------------------------------
            else: # put into "other" bkg group
                if "dy_M-50" in process:
                    # print("dy_M-50 activated")
                    continue
                print("other activated")
                var_hist_other = ROOT.TH1F( var+'_hist_other', var, len(binning)-1, min(binning), max(binning))
                for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    var_hist_other.SetBinContent(1+idx, np_hist[idx])
                group_other_hists.append(var_hist_other)
    
        dummy_hist = ROOT.TH1F('dummy_hist', "dummy", len(binning)-1, min(binning), max(binning))
        dummy_hist.GetXaxis().SetTitleSize(0);
        dummy_hist.GetXaxis().SetLabelSize(0);
        dummy_hist.GetYaxis().SetTitle("Events")
        dummy_hist.Draw("EP");
        
        all_MC_hist_list = []
        
        if len(group_DY_hists) > 0:
            DY_hist_stacked = group_DY_hists[0]
            if len(group_DY_hists) > 1:
                for idx in range(1, len(group_DY_hists)):
                    DY_hist_stacked.Add(group_DY_hists[idx])
            DY_hist_stacked.SetLineColor(1);
            DY_hist_stacked.SetFillColor(ROOT.kOrange+1);
            all_MC_hist_list.append(DY_hist_stacked)
        #----------------------------------------------
        if len(group_Top_hists) > 0:
            Top_hist_stacked = group_Top_hists[0]
            if len(group_Top_hists) > 1:
                for idx in range(1, len(group_Top_hists)):
                    Top_hist_stacked.Add(group_Top_hists[idx])
            Top_hist_stacked.SetLineColor(1);
            Top_hist_stacked.SetFillColor(ROOT.kGreen+1);
            all_MC_hist_list.append(Top_hist_stacked)
        #----------------------------------------------
        if len(group_Ewk_hists) > 0:
            Ewk_hist_stacked = group_Ewk_hists[0]
            if len(group_Ewk_hists) > 1:
                for idx in range(1, len(group_Ewk_hists)):
                    Ewk_hist_stacked.Add(group_Ewk_hists[idx])
            Ewk_hist_stacked.SetLineColor(1);
            Ewk_hist_stacked.SetFillColor(ROOT.kMagenta+1);
            all_MC_hist_list.append(Ewk_hist_stacked)
        #----------------------------------------------
        if len(group_VV_hists) > 0:
            VV_hist_stacked = group_VV_hists[0]
            if len(group_VV_hists) > 1:
                for idx in range(1, len(group_VV_hists)):
                    VV_hist_stacked.Add(group_VV_hists[idx])
            VV_hist_stacked.SetLineColor(1);
            VV_hist_stacked.SetFillColor(ROOT.kAzure+1);
            all_MC_hist_list.append(VV_hist_stacked)
        #----------------------------------------------
        if len(group_other_hists) > 0:
            other_hist_stacked = group_other_hists[0]
            if len(group_other_hists) > 1:
                for idx in range(1, len(group_other_hists)):
                    other_hist_stacked.Add(group_other_hists[idx])
            other_hist_stacked.SetLineColor(1);
            other_hist_stacked.SetFillColor(ROOT.kGray);
            all_MC_hist_list.append(other_hist_stacked)
        #----------------------------------------------
        
        
        # aggregate all MC hist by stacking them and then plot
        all_MC_hist_stacked = ROOT.THStack("all_MC_hist_stacked", "");
        
        if len(all_MC_hist_list) > 0:
            all_MC_hist_list.reverse() # add smallest histgrams first, so from other -> DY
            for MC_hist_stacked in all_MC_hist_list: 
                MC_hist_stacked.Sumw2() # set the hist mode to Sumw2 before stacking
                all_MC_hist_stacked.Add(MC_hist_stacked) 
            
            # now reweight each TH1F stacked in all_MC_hist_stacked
            for idx in range(all_MC_hist_stacked.GetStack().GetEntries()):
                all_MC_hist = all_MC_hist_stacked.GetStack().At(idx) # get the TH1F portion of THStack
                reweightROOTH(all_MC_hist, fraction_weight) # reweight histogram bins and errors
            all_MC_hist_stacked.Draw("hist same");
        
        # stack and plot data 
        if len(group_data_hists) > 0:
            data_hist_stacked = group_data_hists[0]
            if len(group_data_hists) > 1:
                for idx in range(1, len(group_data_hists)):
                    data_hist_stacked.Add(group_data_hists[idx])
            data_hist_stacked.Sumw2()
        
            # decorate the data_histogram
            xlabel = plot_settings[var]["xlabel"].replace('$', '')
            data_hist_stacked.GetXaxis().SetTitle(xlabel);
            data_hist_stacked.GetXaxis().SetTitleOffset(1.10);
            data_hist_stacked.GetYaxis().SetTitleOffset(1.15);
        
            data_hist_stacked.SetMarkerStyle(20);
            data_hist_stacked.SetMarkerSize(1);
            data_hist_stacked.SetMarkerColor(1);
            data_hist_stacked.SetLineColor(1);
            reweightROOTH(data_hist_stacked, fraction_weight) # reweight histogram bins and errors
            data_hist_stacked.Draw("EPsame");        
        
        
        # plot signals: ggH and VBF
        if len(group_ggH_hists) > 0:
            hist_ggH = group_ggH_hists[0]
            hist_ggH.SetLineColor(ROOT.kBlack);
            hist_ggH.SetLineWidth(3);
            hist_ggH.Sumw2()
            reweightROOTH(hist_ggH, fraction_weight) # reweight histogram bins and errors
            hist_ggH.Draw("hist same");
        if len(group_VBF_hists) > 0:
            hist_VBF = group_VBF_hists[0]
            hist_VBF.SetLineColor(ROOT.kRed);
            hist_VBF.SetLineWidth(3);
            hist_VBF.Sumw2()
            reweightROOTH(hist_VBF, fraction_weight) # reweight histogram bins and errors
            hist_VBF.Draw("hist same");
    
        # Ratio pad
        if not args.no_ratio:
            pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9);
            pad2.SetFillColor(0);
            pad2.SetGridy(1);
            pad2.SetFillStyle(0);
            pad2.SetTickx(1);
            pad2.SetTicky(1);
            pad2.SetTopMargin(0.7);
            pad2.SetRightMargin(0.06);
            pad2.Draw();
            pad2.cd();
            
            if (len(group_data_hists) > 0) and (len(all_MC_hist_list) > 0):
                print("ratio activated")
                num_hist = data_hist_stacked.Clone("num_hist");
                den_hist = all_MC_hist_stacked.Clone("den_hist").GetStack().Last(); # to get TH1F from THStack, one needs to call .GetStack().Last()
                print(num_hist)
                print(den_hist)
                num_hist.Divide(den_hist); # we assume Sumw2 mode was previously activated
                num_hist.SetStats(ROOT.kFALSE);
                num_hist.SetLineColor(ROOT.kBlack);
                num_hist.SetMarkerColor(ROOT.kBlack);
                num_hist.SetMarkerSize(0.8);
                
                # get MC statistical errors 
                mc_ratio = all_MC_hist_stacked.Clone("den_hist").GetStack().Last();
                # set all of its errors to zero to prevent double counting of same error
                for idx in range(1, mc_ratio.GetNbinsX()+1):
                    mc_ratio.SetBinError(idx, 0)
                mc_ratio.Divide(den_hist) # divide by itself, errors from den_hist are propagated
                mc_ratio.SetLineColor(0);
                mc_ratio.SetMarkerColor(0);
                mc_ratio.SetMarkerSize(0);
                mc_ratio.SetFillColor(ROOT.kGray);
            
                # get ratio line 
                ratio_line = data_hist_stacked.Clone("num_hist");
                for idx in range(1, mc_ratio.GetNbinsX()+1):
                    ratio_line.SetBinContent(idx, 1)
                    ratio_line.SetBinError(idx, 0)
                ratio_line.SetMarkerSize(0);
                ratio_line.SetLineColor(ROOT.kBlack);
                ratio_line.SetLineStyle(2);
                ratio_line.SetFillColor(0);
                ratio_line.GetYaxis().SetTitle("Data/Pred.");
                ratio_line.GetYaxis().SetRangeUser(0.5,1.5);
                ratio_line.GetYaxis().SetTitleSize(num_hist.GetYaxis().GetTitleSize()*0.85);
                ratio_line.GetYaxis().SetLabelSize(num_hist.GetYaxis().GetLabelSize()*0.85);
                ratio_line.GetYaxis().SetNdivisions(505);
            
                ratio_line.Draw("SAME");
                num_hist.Draw("PE1 SAME");
                mc_ratio.Draw("E2 SAME");
                pad2.RedrawAxis("sameaxis");
    
        # setup legends
        if args.no_ratio:
            leg = ROOT.TLegend(0.40,0.70,0.96,0.9)
        else: # plot ratio
            leg = ROOT.TLegend(0.40,0.80,0.96,1.0)
        
        leg.SetFillColor(0);
        leg.SetFillStyle(0);
        leg.SetBorderSize(0);
        leg.SetNColumns(2);
        if len(group_data_hists) > 0:
            leg.AddEntry(data_hist_stacked,"Data","PEL")
        if len(group_DY_hists) > 0:
            leg.AddEntry(DY_hist_stacked,"DY","F")
        if len(group_Top_hists) > 0:
            leg.AddEntry(Top_hist_stacked,"TOP","F")
        if len(group_Ewk_hists) > 0:
            leg.AddEntry(Ewk_hist_stacked,"Ewk","F")
        if len(group_VV_hists) > 0:
            leg.AddEntry(VV_hist_stacked,"VV","F")
        if len(group_other_hists) > 0:
            leg.AddEntry(other_hist_stacked,"Other","F")
        if len(group_ggH_hists) > 0:
            leg.AddEntry(hist_ggH,"ggH","L")
        if len(group_VBF_hists) > 0:
            leg.AddEntry(hist_VBF,"VBF","L")
        leg.Draw("same");
        
        
        pad.RedrawAxis("sameaxis");
            
        pad.cd();
        # dummy_hist.GetYaxis().SetRangeUser(0.01, data_hist_stacked.GetMaximum()*10000);
        dummy_hist.GetYaxis().SetRangeUser(0.01, 1e9);
        pad.SetLogy();
        pad.Modified();
        pad.Update();
        CMS_lumi(canvas, args.lumi, up=True, reduceSize=True);
        pad.RedrawAxis("sameaxis");
        full_save_path = args.save_path
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)
        canvas.SaveAs(f"{full_save_path}/{var}.pdf");