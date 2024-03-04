import awkward as ak
import dask_awkward as dak
import numpy as np
import json
import argparse
import os
from histogram.ROOT_utils import setTDRStyle, CMS_lumi, reweightROOTH
    

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
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
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
    default=[],
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
    default="./validation/figs/",
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
    "--status",
    dest="status",
    default="",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--ROOT_style",
    dest="ROOT_style",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses pyROOT functionality instead of mplhep",
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
            variables2plot.append(f"{particle}_mass")
            # for kinematic in kinematic_vars:
            #     variables2plot.append(f"{particle}_{kinematic}")
        else:
            print(f"Unsupported variable: {particle} is given!")
    print(f"variables2plot: {variables2plot}")
    # obtain plot settings from config file
    with open("./histogram/plot_settings.json", "r") as file:
        plot_settings = json.load(file)
    status = args.status.replace("_", " ")
    if args.ROOT_style:
        import ROOT
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
                # collect weights
                if "data" in process.lower():
                    weights = np.ones_like(events["mu1_pt"].compute())
                else:
                    weights = ak.to_numpy(events["weight_nominal"].compute() )
                # obtain fraction weight, this should be the same for each process, so doesn't matter if we keep reassigning it
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
            CMS_lumi(canvas, args.lumi, up=True, reduceSize=True, status=status);
            pad.RedrawAxis("sameaxis");
            full_save_path = f"{args.save_path}/{args.year}/ROOT"
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            canvas.SaveAs(f"{full_save_path}/{var}.pdf");
    else:
        import mplhep as hep
        import matplotlib.pyplot as plt
        import matplotlib
        hep.style.use("CMS")
        # Dictionary for histograms and binnings


        for var in variables2plot:
        # for process in available_processes:
            if var not in plot_settings.keys():
                print(f"variable {var} not configured in plot settings!")
                continue
            #-----------------------------------------------
            # intialize variables for filling histograms
            binning = np.linspace(*plot_settings[var]["binning_linspace"])
            group_data_hists = []
            group_DY_hists = []
            group_Top_hists = []
            group_Ewk_hists = []
            group_VV_hists = []
            group_other_hists = []  # histograms not belonging to any other mc bkg group
            group_ggH_hists = [] # there should only be one ggH histogram, but making a list for consistency
            group_VBF_hists = [] # there should only be one VBF histogram, but making a list for consistency
        
            
            # for var in variables2plot:
            for process in available_processes:    
                print(f"process: {process}")
                full_load_path = args.load_path+f"/{process}/*/*.parquet"      
                events = dak.from_parquet(full_load_path)
                # collect weights
                if "data" in process.lower():
                    weights = np.ones_like(events["mu1_pt"].compute())
                else:
                    weights = ak.to_numpy(events["weight_nominal"].compute() )
                #-----------------------------------------------    
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
                weights = weights*category_selection # weights where category_selection==False -> zero

                np_hist, _ = np.histogram(events[var].compute(), bins=binning, weights = weights)
                # print(f"max(np_hist): {max(np_hist)}")
                # print(f"(np_hist): {(np_hist)}")
                # print(f"(np_hist): {np.any(np_hist==0)}")
                
                if process in group_data_processes:
                    print("data activated")
                    group_data_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_DY_processes:
                    print("DY activated")
                    group_DY_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_Top_processes:
                    print("top activated")
                    group_Top_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_Ewk_processes:
                    print("Ewk activated")
                    group_Ewk_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_VV_processes:
                    print("VV activated")
                    var_hist_VV = ROOT.TH1F( var+'_hist_VV', var, len(binning)-1, min(binning), max(binning))
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_VV.SetBinContent(1+idx, np_hist[idx])
                    group_VV_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_ggH_processes:
                    print("ggH activated")
                    group_ggH_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_VBF_processes:
                    print("VBF activated")
                    group_VBF_hists.append(np_hist)
                #-------------------------------------------------------
                else: # put into "other" bkg group
                    if "dy_M-50" in process:
                        # print("dy_M-50 activated")
                        continue
                    print("other activated")
                    group_other_hists.append(np_hist)


                
                
            all_MC_hist_list = []
            groups = []
            if len(group_DY_hists) > 0:
                DY_hist_stacked = np.sum(np.asarray(group_DY_hists), axis=0)
                all_MC_hist_list.append(DY_hist_stacked)
                groups.append("DY")
            #----------------------------------------------
            if len(group_Top_hists) > 0:
                Top_hist_stacked = np.sum(np.asarray(group_Top_hists), axis=0)
                all_MC_hist_list.append(Top_hist_stacked)
                groups.append("Top")
            #----------------------------------------------
            if len(group_Ewk_hists) > 0:
                Ewk_hist_stacked = np.sum(np.asarray(group_Ewk_hists), axis=0)
                all_MC_hist_list.append(Ewk_hist_stacked)
                groups.append("Ewk")
            #----------------------------------------------
            if len(group_VV_hists) > 0:
                VV_hist_stacked = np.sum(np.asarray(group_VV_hists), axis=0)
                all_MC_hist_list.append(VV_hist_stacked)
                groups.append("VV")
            #----------------------------------------------
            if len(group_other_hists) > 0:
                other_hist_stacked = np.sum(np.asarray(group_other_hists), axis=0)
                all_MC_hist_list.append(other_hist_stacked)
                groups.append("other")
            #----------------------------------------------
                
            
            #     for plot_name, settings in plot_settings.items():
            #         hist, _ = np.histogram(events[var].compute(), weights=weights, bins=binning)
            
            #         if plot_name not in histogram_dict:
            #             histogram_dict[plot_name] = {}
            
            #         if process not in histogram_dict[plot_name]:
            #             histogram_dict[plot_name][process] = []
            
            #         histogram_dict[plot_name][process].append(hist)
            
            #         if plot_name not in binning_dict:
            #             binning_dict[plot_name] = {}
            
            #         if process not in binning_dict[plot_name]:
            #             binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])
            
            # # Plotting
            # for plot_name, histograms in histogram_dict.items():
            #     print('INFO: Now making plot for', plot_name, '...')
            #     binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
            #     do_stack = not plot_settings[plot_name].get("density")
            #     data_hist = np.zeros(len(binning)-1)
            #     if args.groupProcesses:
            #         hist_DY = np.zeros(len(binning)-1)
            #         hist_Top = np.zeros(len(binning)-1)
            #     hist_ggh= None
            #     hist_vbf= None
            #     hists_to_plot = []
            #     labels = []
            
            #     for process, histogram in histograms.items():
            #         histogram = np.asarray(histogram)
            #         # print(f"histogram.shape: {histogram.shape}")
            #         histogram = histogram.flatten()
            #         # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
            #         if "data" in process:
            #             data_hist += histogram
            #         else: # MC
            #             if "ggh" in process:
            #                 hist_ggh = histogram
            #             elif  "vbf" in process:
            #                 hist_vbf = histogram
            #             else:
            #                 if args.groupProcesses:
            #                     if process in group_Top_processes:
            #                         hist_Top += histogram
            #                     elif process in group_DY_processes:
            #                         hist_DY += histogram
            #                 else:
            #                     if process == "DYJetsToLL":
            #                         hists_to_plot.append(histogram)
            #                         labels.append(process)
            #                     else:
            #                         hists_to_plot.append(histogram)
            #                         labels.append(process)
                
            #     if args.groupProcesses:
            #         hists_to_plot.append(hist_Top)
            #         labels.append('Top')
            #         hists_to_plot.append(hist_DY)
            #         labels.append('DY')
                    
                    
                
            # colours = hep.style.cms.cmap_petroff[0:3]
            colours = hep.style.cms.cmap_petroff[0:2]
            # print(f"colours: {colours}")
            # print(f"labels: {labels}")
            if not args.no_ratio:
                fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            else: # skip ratio
                fig, ax_main = plt.subplots()
            
            fig.subplots_adjust(hspace=0.1)
            # obtain fraction weight, this should be the same for all processes and rows
            fraction_weight = 1/events.fraction[0].compute() # directly apply these to np hists
            print(f"fraction_weight: {(fraction_weight)}")
            print(f"all_MC_hist_list: {(all_MC_hist_list)}")
            #------------------------------------------
            mc_sum_histogram = np.sum(np.asarray(all_MC_hist_list), axis=0) # to be used in ratio plot later
            group_color_map = {
                "DY" : "Orange",
                "Top" : "Green",
                "EwK" : "Magenta",
                "VV" : "Azure",
                "other" : "Gray"
            }
            colours = [group_color_map[group] for group in groups]
            for hist in all_MC_hist_list:
                hist *= fraction_weight #hists are pointers so this gets
            hep.histplot(all_MC_hist_list, bins=binning, 
                         stack=True, histtype='fill', 
                         label=groups, 
                         sort='label_r', 
                         color=colours, 
                         # density=plot_settings[plot_name].get("density"), 
                         ax=ax_main)

            if len(group_ggH_hists) > 0: # there should be only one element or be empty
                hist_ggh = group_ggH_hists[0]*fraction_weight
                hep.histplot(hist_ggh, bins=binning, 
                             histtype='step', 
                             label="ggH", 
                             sort='label_r', 
                             # color =  hep.style.cms.cmap_petroff[5],
                             color =  "black",
                             # density=plot_settings[plot_name].get("density"), 
                             ax=ax_main)
            if len(group_VBF_hists) > 0: # there should be only one element or be empty
                hist_vbf = group_VBF_hists[0]*fraction_weight
                hep.histplot(hist_vbf, bins=binning, 
                             histtype='step', 
                             label="VBF", 
                             sort='label_r', 
                             # color =  hep.style.cms.cmap_petroff[4],
                             color = "red",
                             # density=plot_settings[plot_name].get("density"), 
                             ax=ax_main)
            
            
            # data_rel_err = np.zeros_like(data_hist)
            # data_rel_err[data_hist>0] = np.sqrt(data_hist)**(-1) # poisson err / value == inverse sqrt()
            #apply fraction weight to data hist and yerr
            data_hist = np.sum(np.asarray(group_data_hists), axis=0)
            data_err = np.sqrt(data_hist) # get yerr b4 fraction weight is applied
            data_hist = data_hist*fraction_weight
            data_err = data_err*fraction_weight
            hep.histplot(data_hist, xerr=True, yerr=data_err,
                         bins=binning, stack=False, histtype='errorbar', color='black', 
                         label='Data', ax=ax_main)
            ax_main.set_ylabel(plot_settings[var].get("ylabel"))
            ax_main.set_yscale('log')
            ax_main.set_ylim(0.01, 1e9)
            ax_main.legend(loc="upper right")
            
            if not args.no_ratio:
                # sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
                mc_yerr = np.sqrt(mc_sum_histogram)
                mc_yerr *= fraction_weight # re apply fraction weights
                mc_sum_histogram  *= fraction_weight # re apply fraction weights
                ratio_hist = np.zeros_like(data_hist)
                ratio_hist[mc_sum_histogram>0] = data_hist[mc_sum_histogram>0]/  mc_sum_histogram[mc_sum_histogram>0]
                # add rel unc of data and mc by quadrature
                rel_unc_ratio = np.sqrt((mc_yerr/mc_sum_histogram)**2 + (data_err/data_hist)**2)
                ratio_err = rel_unc_ratio*ratio_hist
                
                hep.histplot(ratio_hist, 
                             bins=binning, histtype='errorbar', yerr=ratio_err, 
                             color='black', label='Ratio', ax=ax_ratio)
                # hep.histplot(np.ones_like(ratio_hist), 
                #              bins=binning, histtype='fill', yerr=(mc_yerr/mc_sum_histogram).flatten(), 
                #              color='blue', label='MC err', ax=ax_ratio)
                # print("flag3")
                ax_ratio.axhline(1, color='gray', linestyle='--')
                ax_ratio.set_xlabel(plot_settings[var].get("xlabel"))
                ax_ratio.set_ylabel('Data / MC')
                ax_ratio.set_xlim(binning[0], binning[-1])
                # ax_ratio.set_ylim(0.6, 1.4)
                ax_ratio.set_ylim(0.5,1.5) 
            else:  
                ax_main.set_xlabel(plot_settings[var].get("xlabel"))
            # Decorating with CMS label
            if args.lumi == '':
                hep.cms.label(data=True, loc=0, label=status, com=13, ax=ax_main)
            else:
                hep.cms.label(data=True, loc=0, label=status, com=13, lumi=args.lumi, ax=ax_main)

            
            # Saving with special name
            full_save_path = args.save_path+f"/{args.year}/mplhep"
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            plt.savefig(f"{full_save_path}/{var}.pdf")
            plt.clf()

