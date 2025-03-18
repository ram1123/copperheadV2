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
parser.add_argument("--run_label", type=str, required=True, help="Run label")
parser.add_argument("--years", type=str, nargs="+", required=True, help="Year")
parser.add_argument("--njet", type=int, nargs="+", default=[0, 1, 2], help="Number of jets")
parser.add_argument("--input_path", type=str, required=True, help="Input path")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--outAppend", type=str, default="", help="Append to output file name")
args = parser.parse_args()

# Set logging level
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

# create directory to save workspace and plots
outputDirectory = f"./plots_WS_{args.run_label}{args.outAppend}"
os.makedirs(outputDirectory, exist_ok=True)

# Custom Binning for each jet multiplicity
CustomBins = {
    0: np.concatenate([np.linspace(0, 50, 126), np.linspace(60, 200, 15)]),
    1: np.concatenate([np.linspace(0, 50, 100), np.linspace(55, 200, 30)]),
    2: np.concatenate([np.linspace(0, 50, 50), np.linspace(55, 200, 30)]),
}

def log_histogram(hist, name):
    """Log histogram details for debugging."""
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
        logger.info(f"Processing year: {year}")
        logger.info(f"Base path: {base_path}")

        try:
            # load the data and dy samples
            data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")
            dy_events = dak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")

            # Apply Z-peak region filter
            data_events = filterRegion(data_events, region="z-peak")
            dy_events = filterRegion(dy_events, region="z-peak")
        except Exception as e:
            logger.error(f"Error loading parquet files for {year}: {e}")
            exit()

        for njet in args.njet:
            logger.info(f"Processing njet{njet} for {year}")

            # Apply jet selection
            njet_field = "njets_nominal" # FIXME: Hardcoded field name
            if njet < 2:
                data_events_filtered = data_events[data_events[njet_field] == njet]
                dy_events_filtered = dy_events[dy_events[njet_field] == njet]
            else:
                data_events_filtered = data_events[data_events[njet_field] >= njet]
                dy_events_filtered = dy_events[dy_events[njet_field] >= njet]

            fields2load = ["wgt_nominal", "dimuon_pt"]
            try:
                data_dict = {field: ak.to_numpy(data_events_filtered[field].compute()) for field in fields2load}
                dy_dict = {field: ak.to_numpy(dy_events_filtered[field].compute()) for field in fields2load}
            except Exception as e:
                logger.error(f"Error extracting arrays for {year} njet{njet}: {e}")
                exit()

            logger.debug(f"Data: {len(data_dict['dimuon_pt'])} events")
            logger.debug(f"DY: {len(dy_dict['dimuon_pt'])} events")

            bins = [25, 50, 100, 200, 300, 400, 500, "CustomBins"]
            bins = ["CustomBins"]
            for nbins in bins:
                logger.info(f"Processing {year} njet{njet} nbins: {nbins}")
                if nbins == "CustomBins":
                    binning_array = CustomBins.get(njet, np.linspace(0, 200, 200 + 1))
                else:
                    binning_array = np.linspace(0, 200, nbins + 1)

                hist_data = ROOT.TH1F(f"hist_data_njet{njet}_nbins{nbins}", "Data", len(binning_array) - 1, binning_array)
                hist_dy = ROOT.TH1F(f"hist_dy_njet{njet}_nbins{nbins}", "DY", len(binning_array) - 1, binning_array)

                # Fill histograms
                for val, weight in zip(data_dict["dimuon_pt"], data_dict["wgt_nominal"]):
                    hist_data.Fill(val, weight)

                for val, weight in zip(dy_dict["dimuon_pt"], dy_dict["wgt_nominal"]):
                    hist_dy.Fill(val, weight)

                # Generate SF histogram (Data/MC)
                hist_SF = hist_data.Clone(f"hist_SF_njet{njet}_nbins{nbins}")
                hist_SF.Divide(hist_dy)

                # Save histograms in a RooWorkspace
                workspace = ROOT.RooWorkspace("zpt_Workspace") # FIXME: Hardcoded workspace name
                for hist in [hist_data, hist_dy, hist_SF]:
                    getattr(workspace, "import")(hist)

                # Save the workspace to a ROOT file
                output_file = ROOT.TFile(f"{outputDirectory}/{year}_njet{njet}_nbins{nbins}.root", "RECREATE")
                workspace.Write()
                output_file.Close()

                # Sanity check: Loop through the bins and calculate the relative error
                if args.debug:
                    log_histogram(hist_data, "Data")
                    log_histogram(hist_dy, "DY")
                    log_histogram(hist_SF, "SF")

                # Plot Histograms
                canvas = ROOT.TCanvas("canvas", f"Data vs DY {args.run_label}", 800, 600)
                hist_data.SetLineColor(ROOT.kRed)
                hist_dy.SetLineColor(ROOT.kBlue)
                hist_data.Draw()
                hist_dy.Draw("SAME")

                legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
                legend.AddEntry(hist_data, "Data", "l")
                legend.AddEntry(hist_dy, "DY", "l")
                legend.Draw()

                # Save plots
                for ext in ["png", "pdf"]:
                    canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}.{ext}")
                    canvas.SetLogy(1)
                    canvas.SaveAs(f"{outputDirectory}/dataDy_{year}_njet{njet}_nbins{nbins}_log.{ext}")
                    canvas.SetLogy(0)

                # SF histogram
                canvas.Clear()
                canvas.SetLogy(0)
                hist_SF.Draw()
                for ext in ["png", "pdf"]:
                    canvas.SaveAs(f"{outputDirectory}/SF_{year}_njet{njet}_nbins{nbins}.{ext}")

                # clear the canvas and delete the histograms
                canvas.Clear()
                canvas.Close()
                del hist_data, hist_dy, hist_SF
                logger.info(f"Completed {year} njet{njet} nbins{nbins}")

    logger.info("All done!")
