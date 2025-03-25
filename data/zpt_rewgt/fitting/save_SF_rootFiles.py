import ROOT
import numpy as np
import dask_awkward as dak
import awkward as ak
import argparse
import sys
from distributed import LocalCluster, Client, progress
import os
from omegaconf import OmegaConf
import copy
from array import array
from ROOT import RooFit

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events
if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    # from dask_gateway import Gateway
    # gateway = Gateway(
    #     "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    #     proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    # )
    # cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    # client = gateway.connect(cluster_info.name).get_client()
    # print("Gateway Client created")
    cluster = LocalCluster(processes=True)
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    print("Local scale Client created")

    run_label = "V2_Jan09_ForZptReWgt"
    # year = "2018"
    for year in ["2017","2016postVFP","2016preVFP"]:
        
    
        base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0" # define the save path of stage1 outputs
        
        
        # load the data and dy samples
        data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")
        dy_events = dak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")
        
        # apply z-peak region filter and nothing else
        data_events = filterRegion(data_events, region="z-peak")
        dy_events = filterRegion(dy_events, region="z-peak")
    
        
        njet_field = "njets_nominal"
        for njet in [0,1,2]:
        # for njet in [2]:
            # njet = 0
            
            
            if njet != 2:
                data_events_loop = data_events[data_events[njet_field] ==njet]
                dy_events_loop = dy_events[dy_events[njet_field] ==njet]
            else:
                data_events_loop = data_events[data_events[njet_field] >=njet]
                dy_events_loop = dy_events[dy_events[njet_field] >=njet]
    
    
            fields2load = ["wgt_nominal", "dimuon_pt"]
            data_dict = {field: ak.to_numpy(data_events_loop[field].compute()) for field in fields2load}
            dy_dict = {field: ak.to_numpy(dy_events_loop[field].compute()) for field in fields2load}
    
            # print(f"data_dict: {data_dict}")
            # print(f"dy_dict: {dy_dict}")
    
            # binning_path = f"../binning/{year}_njet{njet}.yml"
            # # Load the YAML file
            # config = OmegaConf.load(binning_path)
            
            # binning = config["rewgt_binning"]
            # # Convert the list of bin edges to a C-style array
            # # binning_array = np.array(binning)
            binning_array = np.linspace(0,200, 501)
            
            # Step 2: Create the histogram with variable bin widths
            hist_data = ROOT.TH1F("hist_data", "Data", len(binning_array) - 1, binning_array)
            hist_dy = ROOT.TH1F("hist_dy", "DY", len(binning_array) - 1, binning_array)
    
    
            # fill the histograms
            # values = np.random.uniform(low=0.5, high=13.3, size=(1000,)) # temp test
            # weights = np.random.uniform(low=0.5, high=13.3, size=(1000,)) # temp test
    
    
            
            values = data_dict["dimuon_pt"]
            values = array('d', values)
            weights = data_dict["wgt_nominal"]
            weights = array('d', weights)
            hist_data.FillN(len(values), values, weights)
            
            values = dy_dict["dimuon_pt"]
            values = array('d', values)
            weights = dy_dict["wgt_nominal"]
            weights = array('d', weights)
            hist_dy.FillN(len(values), values, weights)

            
            # generate SF histogram (Data/MC)
            hist_SF = hist_data.Clone("hist_SF")
            hist_SF.Divide(hist_dy)

            # save the histograms in workspace
            workspace = ROOT.RooWorkspace("zpt_Workspace")

            # Import the histograms into the workspace
            getattr(workspace, "import")(hist_data)  # Use getattr to call 'import' (Python keyword)
            getattr(workspace, "import")(hist_dy)
            
            # Save the workspace to a ROOT file
            output_file = ROOT.TFile(f"{year}_njet{njet}.root", "RECREATE")
            workspace.Write()  # Write the workspace to the file
            output_file.Close()

    
    
            # # Sanity check: Loop through the bins and calculate the relative error
            # print("data Hist Bin | Content | Error | Relative Error (%)")
            # print("------------------------------------------")
            # hist = hist_data
            # for i in range(1, hist.GetNbinsX() + 1):  # Loop over bins (1-indexed in ROOT)
            #     content = hist.GetBinContent(i)      # Bin content
            #     error = hist.GetBinError(i)          # Bin error
            #     if content != 0:
            #         relative_error = (error / content) * 100  # Relative error in percentage
            #     else:
            #         relative_error = 0  # Avoid division by zero
                
            #     # Print the results
            #     print(f"{i:3} | {content:7.2f} | {error:6.2f} | {relative_error:6.2f} %")
            # print("dy Hist Bin | Content | Error | Relative Error (%)")
            # print("------------------------------------------")
            # hist = hist_dy
            # for i in range(1, hist.GetNbinsX() + 1):  # Loop over bins (1-indexed in ROOT)
            #     content = hist.GetBinContent(i)      # Bin content
            #     error = hist.GetBinError(i)          # Bin error
            #     if content != 0:
            #         relative_error = (error / content) * 100  # Relative error in percentage
            #     else:
            #         relative_error = 0  # Avoid division by zero
                
            #     # Print the results
            #     print(f"{i:3} | {content:7.2f} | {error:6.2f} | {relative_error:6.2f} %")
            # print("SF Hist Bin | Content | Error | Relative Error (%)")
            # print("------------------------------------------")
            # hist = hist_SF
            # for i in range(1, hist.GetNbinsX() + 1):  # Loop over bins (1-indexed in ROOT)
            #     content = hist.GetBinContent(i)      # Bin content
            #     error = hist.GetBinError(i)          # Bin error
            #     if content != 0:
            #         relative_error = (error / content) * 100  # Relative error in percentage
            #     else:
            #         relative_error = 0  # Avoid division by zero
                
            #     # Print the results
            #     print(f"{i:3} | {content:7.2f} | {error:6.2f} | {relative_error:6.2f} %")

            canvas = ROOT.TCanvas("canvas", f"histogram {run_label}", 800, 600)
            # Set histogram styles
            hist_data.SetLineColor(ROOT.kRed)
            hist_dy.SetLineColor(ROOT.kBlue)
            hist_data.Draw()
            hist_dy.Draw("SAME")

            # Add a legend
            legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Legend coordinates (x1, y1, x2, y2)
            legend.AddEntry(hist_data, "Data 1", "l")  # "l" means line style
            legend.AddEntry(hist_dy, "DY", "l")
            legend.Draw()
            
            
            # Save the canvas as an image
            canvas.SaveAs(f"dataDy_{year}_njet{njet}.png")

            canvas.SetLogy(1)
            canvas.Update()
            canvas.SaveAs(f"dataDy_{year}_njet{njet}_logScale.png")
            canvas.SetLogy(0)
            canvas.Update()
            # ---------------------------------------------------------
            
    
            canvas = ROOT.TCanvas("canvas", f"SF histogram {run_label}", 800, 600)
            hist_SF.Draw()
            
            # Save the canvas as an image
            canvas.SaveAs(f"SF_{year}_njet{njet}.png")
    
    
            # # convert to RooHist and do polynomial fit
            # x = ROOT.RooRealVar("dimuon_pt", "dimuon_pt", 0, 200)
            # x.setRange("fit_range", 0, 200 )
            
            # # Step 3: Convert the histogram to a RooDataHist
            # roohist = ROOT.RooDataHist("data_hist", "RooDataHist from TH1F", ROOT.RooArgList(x), hist_SF)
    
            
            # # Step 4: Define the polynomial parameters and model
            # # a0 = RooRealVar("a0", "a0", 0, -10, 10)  # Constant term
            # # a1 = RooRealVar("a1", "a1", 0, -10, 10)  # Linear coefficient
            # # a2 = RooRealVar("a2", "a2", 0, -10, 10)  # Quadratic coefficient
            # coeff_l = []
            # for ix in range(1,6):
            #     coeff_l.append(ROOT.RooRealVar(f"a{ix}", f"a{ix}", 0, -10, 10))
            
            # polynomial = ROOT.RooPolynomial("polynomial", "polynomial PDF", x, coeff_l)
            
            # # Step 5: Fit the polynomial model to the histogram data
            # _ = polynomial.fitTo(roohist, RooFit.Range("fit_range"), RooFit.Save())
            # fit_result = polynomial.fitTo(roohist, RooFit.Range("fit_range"), RooFit.Save())
            # fitResult.Print()
            
            # # Step 6: Visualize the results
            # frame = x.frame()
            # roohist.plotOn(frame)          # Plot the histogram
            # polynomial.plotOn(frame)         # Plot the fitted model
            
            # # Draw the frame
            # canvas = ROOT.TCanvas("canvas", "Fit to Histogram", 800, 600)
            # frame.Draw()
            # canvas.SaveAs("histogram_fit.png")
        
