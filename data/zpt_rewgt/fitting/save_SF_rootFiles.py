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
import argparse

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

    events = events[region]
    return events

def zipAndCompute(events, fields2load):
    zpt_wgt_name = "separate_wgt_zpt_wgt"
    if zpt_wgt_name in events.fields:
        events["wgt_nominal"] = events["wgt_nominal"] / events["separate_wgt_zpt_wgt"] # turn off Zpt

    return_zip = ak.zip({
        field : events[field] for field in fields2load
    })
    return return_zip.compute() # compute and return

if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-save",
    "--plot_path",
    dest="plot_path",
    default="plots",
    action="store",
    help="save path to store plots, not SF root files. That's saved locally",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    # option for dy samples: aMCatNLO, MiNNLO, or VBF_filter
    parser.add_argument(
    "-dy_sample",
    "--dy_sample",
    dest="dy_sample",
    default="MiNNLO",
    choices=["MiNNLO", "aMCatNLO", "VBF_filter"],
    action="store",
    help="choose the type of DY samples to use for Zpt reweighting",
    )
    args = parser.parse_args()

    # intialize dask client
    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
    else: # use local cluster
        client = Client(n_workers=63,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
        print("Local scale Client created")

    # specify stage1 output label
    run_label = args.label
    if args.year == "all":
        # years =  ["2018", "2017","2016postVFP","2016preVFP"]
        years =  ["2017","2016postVFP","2016preVFP"]
    else:
        years = [args.year]



    for year in years:
        plot_path = f"{args.plot_path}/zpt_rewgt/{run_label}/{args.dy_sample}/{year}"
        os.makedirs(plot_path, exist_ok=True)
        print(f"years: {years}")

        # get user name
        user_name = os.getenv("USER")
        if user_name is None:
            print("Error: USER environment variable is not set.")
            sys.exit(1)

        base_path = f"/depot/cms/users/{user_name}/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0" # define the save path of stage1 outputs

        # load the data and dy samples
        print(f"base path data : {base_path}/data_*/*/*.parquet")
        print(f"base path dy: {base_path}/dy*{args.dy_sample}/*/*.parquet")
        data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")

        if args.dy_sample == "MiNNLO":
            dy_events = dak.from_parquet(f"{base_path}/dy*MiNNLO/*/*.parquet")
        elif args.dy_sample == "aMCatNLO":
            dy_events = dak.from_parquet(f"{base_path}/dy*_aMCatNLO/*/*.parquet")
        elif args.dy_sample == "VBF_filter":
            dy_events = dak.from_parquet(f"{base_path}/dy_VBF_filter/*/*.parquet")
        else:
            raise ValueError(f"Unknown dy_sample option: {args.dy_sample}. Choose from MiNNLO, aMCatNLO, or VBF_filter.")

        # apply z-peak region filter and nothing else
        data_events = filterRegion(data_events, region="z-peak")
        dy_events = filterRegion(dy_events, region="z-peak")
        print(f"dy_events: {dy_events}")
        # print(f"dy_events.dimuon_mass.compute(): {dy_events.dimuon_mass.compute()}")

        njet_field = "njets_nominal"
        for njet in [0,1,2]:
            if njet != 2:
                data_events_loop = data_events[data_events[njet_field] ==njet]
                dy_events_loop = dy_events[dy_events[njet_field] ==njet]
            else:
                data_events_loop = data_events[data_events[njet_field] >=njet]
                dy_events_loop = dy_events[dy_events[njet_field] >=njet]


            fields2load = ["wgt_nominal", "dimuon_pt"]
            # data_dict = {field: ak.to_numpy(data_events_loop[field].compute()) for field in fields2load}
            # dy_dict = {field: ak.to_numpy(dy_events_loop[field].compute()) for field in fields2load}
            data_dict = zipAndCompute(data_events_loop, fields2load)
            data_dict = {field: ak.to_numpy(data_dict[field]) for field in data_dict.fields}
            dy_dict = zipAndCompute(dy_events_loop, fields2load)
            dy_dict = {field: ak.to_numpy(dy_dict[field]) for field in dy_dict.fields}

            # print(f"data_dict: {data_dict}")
            # print(f"dy_dict: {dy_dict}")


            # binning = config["rewgt_binning"]
            # # Convert the list of bin edges to a C-style array
            # # binning_array = np.array(binning)
            binning_array = np.linspace(0,200, 2001) # 2000 bins from 0 to 200 GeV

            # Step 2: Create the histogram with variable bin widths
            hist_data = ROOT.TH1F("hist_data", "Data", len(binning_array) - 1, binning_array)
            hist_dy = ROOT.TH1F("hist_dy", "DY", len(binning_array) - 1, binning_array)


            #  Step 3: Fill the histogram with data
            # Convert the values and weights to arrays
            # Note: The values and weights should be numpy arrays or lists
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


            # Step 4: Get the Scale Factor (SF) histogram, by dividing data histogram by DY histogram
            hist_SF = hist_data.Clone("hist_SF")
            hist_SF.Divide(hist_dy)

            # Step 5: save the histograms in workspace
            workspace = ROOT.RooWorkspace("zpt_Workspace")

            # Import the histograms into the workspace
            getattr(workspace, "import")(hist_data)  # Use getattr to call 'import' (Python keyword)
            getattr(workspace, "import")(hist_dy)

            # Step 6: Save the workspace to a ROOT file
            copy_file = ROOT.TFile(f"{plot_path}/{year}_njet{njet}.root", "RECREATE")
            workspace.Write()  # Write the workspace to the file
            copy_file.Close()


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
            canvas.SaveAs(f"{plot_path}/dataDy_{year}_njet{njet}.pdf")

            canvas.SetLogy(1)
            canvas.Update()
            canvas.SaveAs(f"{plot_path}/dataDy_{year}_njet{njet}_logScale.pdf")
            canvas.SetLogy(0)
            canvas.Update()
            # ---------------------------------------------------------


            canvas = ROOT.TCanvas("canvas", f"SF histogram {run_label}", 800, 600)
            hist_SF.Draw()

            # Save the canvas as an image
            canvas.SaveAs(f"{plot_path}/SF_{year}_njet{njet}.pdf")
