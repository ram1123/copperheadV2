import ROOT
import numpy as np
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

# Define muon pT categories
muon_pT_categories = {
    "mu1_20_30_mu2_20_30": ((20, 30), (20, 30)),
    "mu1_30_40_mu2_20_30": ((30, 40), (20, 30)),
    "mu1_40_50_mu2_20_30": ((40, 50), (20, 30)),
    "mu1_gt50_mu2_20_30": ((50, None), (20, 30)),
    "mu1_30_40_mu2_30_40": ((30, 40), (30, 40)),
    "mu1_40_50_mu2_30_40": ((40, 50), (30, 40)),
    "mu1_gt50_mu2_30_40": ((50, None), (30, 40)),
    "mu1_40_50_mu2_40_50": ((40, 50), (40, 50)),
    "mu1_gt50_mu2_40_50": ((50, None), (40, 50)),
    "mu1_gt50_mu2_gt50": ((50, None), (50, None)),
    "mu1_gt20_mu2_gt20": ((20, None), (20, None)),
    "mu1_gt0_mu2_gt0": ((0, None), (0, None)),
}

if __name__ == "__main__":
    for year in args.years:
        base_path = f"{args.input_path}/stage1_output/{year}/f1_0"
        logger.info(f"Processing year: {year}")

        try:
            data_events = ak.from_parquet(f"{base_path}/data_*/*/*.parquet")
            dy_events = ak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")

            # Apply Z-peak region filter
            data_events = filterRegion(data_events, region="z-peak")
            dy_events = filterRegion(dy_events, region="z-peak")
        except Exception as e:
            logger.error(f"Error loading parquet files for {year}: {e}")
            exit()

        for njet in args.njet:
            logger.info(f"Processing njet{njet} for {year}")

            njet_field = "njets_nominal"
            if njet < 2:
                data_events_filtered = data_events[data_events[njet_field] == njet]
                dy_events_filtered = dy_events[dy_events[njet_field] == njet]
            else:
                data_events_filtered = data_events[data_events[njet_field] >= njet]
                dy_events_filtered = dy_events[dy_events[njet_field] >= njet]

            fields2load = ["wgt_nominal", "dimuon_pt", "mu1_pt", "mu2_pt"]

            try:
                data_dict = {field: ak.to_numpy(data_events_filtered[field]) for field in fields2load}
                dy_dict = {field: ak.to_numpy(dy_events_filtered[field]) for field in fields2load}
            except Exception as e:
                logger.error(f"Error extracting arrays for {year} njet{njet}: {e}")
                exit()

            for cat_label, ((mu1_min, mu1_max), (mu2_min, mu2_max)) in muon_pT_categories.items():
                data_mask = (data_dict["mu1_pt"] > mu1_min) & (data_dict["mu2_pt"] > mu2_min)
                dy_mask = (dy_dict["mu1_pt"] > mu1_min) & (dy_dict["mu2_pt"] > mu2_min)

                if mu1_max:
                    data_mask &= (data_dict["mu1_pt"] < mu1_max)
                    dy_mask &= (dy_dict["mu1_pt"] < mu1_max)
                if mu2_max:
                    data_mask &= (data_dict["mu2_pt"] < mu2_max)
                    dy_mask &= (dy_dict["mu2_pt"] < mu2_max)

                data_dimuon_pt = data_dict["dimuon_pt"][data_mask]
                dy_dimuon_pt = dy_dict["dimuon_pt"][dy_mask]
                data_weights = data_dict["wgt_nominal"][data_mask]
                dy_weights = dy_dict["wgt_nominal"][dy_mask]

                hist_data = ROOT.TH1F(f"hist_data_{cat_label}", "Data", 50, 0, 200)
                hist_dy = ROOT.TH1F(f"hist_dy_{cat_label}", "DY", 50, 0, 200)

                for val, weight in zip(data_dimuon_pt, data_weights):
                    hist_data.Fill(val, weight)
                for val, weight in zip(dy_dimuon_pt, dy_weights):
                    hist_dy.Fill(val, weight)

                canvas = ROOT.TCanvas("canvas", "Ratio Plot", 800, 600)
                hist_data.SetLineColor(ROOT.kBlack)
                hist_dy.SetLineColor(ROOT.kRed)

                ratio_plot = ROOT.TRatioPlot(hist_data, hist_dy)
                ratio_plot.Draw()
                # Fix range of ratio plot
                ratio_plot.GetLowerRefGraph().SetMinimum(0.0)
                ratio_plot.GetLowerRefGraph().SetMaximum(2.0)

                # add legend
                ratio_plot.GetUpperPad().cd()
                legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
                legend.AddEntry(hist_data, "Data", "l")
                legend.AddEntry(hist_dy, "DY", "l")
                legend.Draw()
                canvas.SaveAs(f"{outputDirectory}/{year}_njet{njet}_{cat_label}_ratio.png")

                hist_SF = hist_data.Clone(f"hist_SF_{cat_label}")
                hist_SF.Divide(hist_dy)

                output_file = ROOT.TFile(f"{outputDirectory}/{year}_njet{njet}_{cat_label}.root", "RECREATE")
                hist_data.Write()
                hist_dy.Write()
                hist_SF.Write()
                output_file.Close()

                logger.info(f"Completed {year} njet{njet} category {cat_label}")
                canvas.Clear()
    logger.info("All done!")
