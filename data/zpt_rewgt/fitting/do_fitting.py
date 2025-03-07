import ROOT
import numpy as np
import dask_awkward as dak
import awkward as ak
import argparse
import os
from array import array
from ROOT import RooFit
import logging

from modules.basic_functions import filterRegion
from modules.utils import logger

parser = argparse.ArgumentParser()
parser.add_argument("--run_label", type=str, help="Run label", required=True)
parser.add_argument("--years", type=str, nargs="+", help="Year", required=True)
parser.add_argument("--njet", type=int, nargs="+", default=[0,1,2], help="Number of jets (default: [0, 1, 2])", required=False)
parser.add_argument("--input_path", type=str, help="Input path", required=True)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--outAppend", type=str, help="Append to output file name", default="")
parser.add_argument("--nbins", type=int, help="Number of bins", default=501)
args = parser.parse_args()

# Set logging level
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

# create directory to save workspace and plots
outputDirectory = f"./plots_WS_{args.run_label}{args.outAppend}"
os.makedirs(outputDirectory, exist_ok=True)

def log_histogram(hist, name):
    logger.debug(f"{name} Histogram | Bin | Content | Error | Relative Error (%)")
    for i in range(1, hist.GetNbinsX() + 1):
        content = hist.GetBinContent(i)
        error = hist.GetBinError(i)
        rel_error = (error / content) * 100 if content else 0
        logger.debug(f"{i:3} | {content:7.2f} | {error:6.2f} | {rel_error:6.2f} %")


if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    for year in args.years:
        base_path = f"{args.input_path}/stage1_output/{year}/f1_0"
        logger.debug(f"base path: {base_path}")

        # load the data and dy samples
        data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")
        dy_events = dak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")

        # apply z-peak region filter and nothing else
        data_events = filterRegion(data_events, region="z-peak")
        dy_events = filterRegion(dy_events, region="z-peak")

        njet_field = "njets_nominal"
        CustomBins = [np.concatenate([np.linspace(0, 50, 126), np.linspace(60, 200, 15)]),
                      np.concatenate([np.linspace(0, 50, 100), np.linspace(55, 200, 30)]),
                      np.concatenate([np.linspace(0, 50, 50), np.linspace(55, 200, 30)])]
        for njet in args.njet:
            if njet < 2:
                data_events_filtered = data_events[data_events[njet_field] == njet]
                dy_events_filtered = dy_events[dy_events[njet_field] == njet]
            else:
                data_events_filtered = data_events[data_events[njet_field] >= njet]
                dy_events_filtered = dy_events[dy_events[njet_field] >= njet]

            fields2load = ["wgt_nominal", "dimuon_pt"]
            data_dict = {field: ak.to_numpy(data_events_filtered[field].compute()) for field in fields2load}
            dy_dict = {field: ak.to_numpy(dy_events_filtered[field].compute()) for field in fields2load}

            logger.debug(f"Data Dictionary: {data_dict}")
            logger.debug(f"DY Dictionary: {dy_dict}")

            # Define histogram binning
            # for nbins in [25, 50, 100, 200, 300, 400, 500, "CustomBins"]:
            for nbins in ["CustomBins"]:
                if nbins == "CustomBins":
                    binning_array = CustomBins[njet]
                else:
                    binning_array = np.linspace(0, 200, nbins+1)

                # Create histograms
                hist_data = ROOT.TH1F("hist_data", "Data", len(binning_array) - 1, binning_array)
                hist_dy = ROOT.TH1F("hist_dy", "DY", len(binning_array) - 1, binning_array)

                # Fill histograms
                for val, weight in zip(data_dict["dimuon_pt"], data_dict["wgt_nominal"]):
                    hist_data.Fill(val, weight)

                for val, weight in zip(dy_dict["dimuon_pt"], dy_dict["wgt_nominal"]):
                    hist_dy.Fill(val, weight)

                # generate SF histogram (Data/MC)
                hist_SF = hist_data.Clone("hist_SF")
                hist_SF.Divide(hist_dy)

                # save the histograms in workspace
                workspace = ROOT.RooWorkspace("zpt_Workspace")
                # Import the histograms into the workspace
                getattr(workspace, "import")(hist_data)
                getattr(workspace, "import")(hist_dy)
                getattr(workspace, "import")(hist_SF)

                # Save the workspace to a ROOT file
                output_file = ROOT.TFile(f"{outputDirectory}/{year}_njet{njet}_nbins{nbins}.root", "RECREATE")
                workspace.Write()
                output_file.Close()

                # Sanity check: Loop through the bins and calculate the relative error
                if args.debug:
                    log_histogram(hist_data, "Data")
                    log_histogram(hist_dy, "DY")
                    log_histogram(hist_SF, "SF")

                # Plot Data and DY distributions
                canvas = ROOT.TCanvas("canvas", f"Data vs DY {args.run_label}", 800, 600)
                hist_data.SetLineColor(ROOT.kRed)
                hist_dy.SetLineColor(ROOT.kBlue)
                hist_data.Draw()
                hist_dy.Draw("SAME")

                legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
                legend.AddEntry(hist_data, "Data", "l")
                legend.AddEntry(hist_dy, "DY", "l")
                legend.Draw()

                canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}.png")
                canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}.pdf")
                canvas.SetLogy(1)
                canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}_log.png")
                canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}_log.pdf")

                # Plot SF histogram
                canvas.Clear()
                canvas.SetLogy(0)
                hist_SF.Draw()
                canvas.SaveAs(f"{outputDirectory}/SF_{year}_njet{njet}_nbins{nbins}.png")
                canvas.SaveAs(f"{outputDirectory}/SF_{year}_njet{njet}_nbins{nbins}.pdf")

                # clear the canvas
                canvas.Clear()
                canvas.Close()
                del hist_data, hist_dy, hist_SF
    logger.info("All done!")
