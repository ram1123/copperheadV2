"""
Step 0 of the Z-pT reweighting derivation: Data and DY dimuon-pT histograms in the Z peak per njet bin
(0, 1, >=2) -> <save_path>/zpt_rewgt/<label>/<dy_sample>/<year>/<year>_njet<N>.root, read by do_f_test.py and
get_polyFit.py (they fit hist_data / hist_dy). Any Z-pT weight already in the ntuple is divided out of DY.
Options for flag-skim ntuples and background subtraction:
  --flag_skim_selection <name>  apply modules.selection.FLAG_SKIM_SELECTIONS[name] (required for a flag-skim ntuple)
  --subtract_nondy              hist_data = Data - non-DY MC (all background groups except DY/DYVBF), so the
                                weight is (Data - non-DY)/DY; raw data and non-DY are kept as hist_data_raw/hist_nondy.
How to run (repo root, default pixi env; Gateway workers must import this repo):
  python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py -l <label> -y <year> --input_path <ntuple dir> \
      -dy_sample <output tag> [--flag_skim_selection official] [--subtract_nondy] [--use_gateway]
Example:
  ./run_in_pixi.sh default python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py \
      -l Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit -y 2024 \
      --input_path /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit \
      -dy_sample IncDY_aMCatNLO_FSv2official_BkgSub --flag_skim_selection official --subtract_nondy --use_gateway

  ./run_in_pixi.sh default python src/copperhead/zpt_rewgt/derive/save_SF_rootFiles.py \
      -l Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit -y 2024 \
      --input_path /work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FlagSkimV2_Oct30_OfficialRecomendation_Systematics_LumiSplit \
      -dy_sample IncDY_aMCatNLO_FSv2official --flag_skim_selection official --use_gateway      
"""
import json
import os
import sys
from array import array
from datetime import datetime, timezone
import glob

import awkward as ak
import dask_awkward as dak
import numpy as np
import ROOT
from cli.common_argparser import build_common_parser
from modules import classify_year, selection
from modules.dask_utils import get_dask_client
from modules.dask_utils import close_dask_client
from modules.git_utils import get_git_state
from modules.sample_config import get_sample_dict
from sample_resolution import collect_process_paths, resolve_dy_processes

# stage-1 column holding the applied Z-pT weight; renamed in 5d73ffa (2026-07-02), older ntuples use the first
ZPT_WGT_COLUMNS = ("separate_wgt_zpt_wgt", "separate_wgt_zpt")


def zipAndCompute(events, fields2load):
    for zpt_wgt_name in ZPT_WGT_COLUMNS:
        if zpt_wgt_name in events.fields:
            events["wgt_nominal"] = events["wgt_nominal"] / events[zpt_wgt_name] # turn off Zpt
            break

    return_zip = ak.zip({
        field : events[field] for field in fields2load
    })
    return return_zip.compute() # compute and return


def _flag_skim_partition(events, name):
    return selection.applyFlagSkimSelection(events, name)


def apply_flag_skim(events, name):
    # eager selection per partition (one row per selected event + Loose/Tight SF rescale), as stage-2/plotter
    return dak.map_partitions(_flag_skim_partition, events, name, label=f"flag_skim_{name}")

if __name__ == "__main__":
    """
    This file is meant to define the Zpt histogram binning for zpt fitting
    """
    parser = build_common_parser()
    parser.add_argument(
        "--flag_skim_selection",
        default="",
        choices=[""] + sorted(selection.FLAG_SKIM_SELECTIONS),
        help="Named selection on a flag-skim stage-1 ntuple (required for such input).",
    )
    parser.add_argument(
        "--subtract_nondy",
        action="store_true",
        help="Fit (Data - non-DY MC)/DY instead of Data/DY.",
    )
    args = parser.parse_args()

    # intialize dask client
    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)

    # specify stage1 output label
    run_label = args.label
    if args.year == "all":
        # years =  ["2018", "2017","2016postVFP","2016preVFP"]
        years =  ["2017","2016postVFP","2016preVFP"]
    else:
        years = [args.year]

    for year in years:
        plot_path = f"{args.save_path}/zpt_rewgt/{run_label}/{args.dy_sample}/{year}"
        os.makedirs(plot_path, exist_ok=True)
        print(f"years: {years}")

        # get user name
        user_name = os.getenv("USER")
        if user_name is None:
            print("Error: USER environment variable is not set.")
            sys.exit(1)

        base_path = f"{args.input_path}/stage1_output/{year}/compacted"  # define the save path of stage1 outputs

        # if base_path does not exist, check to replace "compacted" with "f1_0"
        if not os.path.exists(base_path):
            base_path = base_path.replace("compacted", "f1_0")
            if not os.path.exists(base_path):
                raise RuntimeError(f"Base path does not exist: {base_path}")

        # load the data and dy samples
        print(f"base path data : {base_path}/data_*/*/*.parquet")
        data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")

        dy_processes = resolve_dy_processes(year, args.sample_config)
        dy_paths, matched_dy_processes, missing_dy_processes = collect_process_paths(
            base_path, dy_processes
        )
        print(f"Resolved DY processes from {args.sample_config}: {dy_processes}")
        print(f"Matched DY processes: {matched_dy_processes}")
        if missing_dy_processes:
            print(f"WARNING: Missing DY processes under {base_path}: {missing_dy_processes}")
        print(f"Total DY parquet files found: {len(dy_paths)}")

        # Provenance for future reference: when/by whom/from which inputs this
        # year's Data and DY histograms (and hence everything downstream in
        # get_polyFit.py) were produced.
        matched_data_dirs = sorted(
            os.path.basename(p) for p in glob.glob(f"{base_path}/data_*") if os.path.isdir(p)
        )
        git_state = get_git_state(plot_path)
        provenance = {
            "step": "save_SF_rootFiles.py",
            "produced_at": datetime.now(timezone.utc).isoformat(),
            "produced_by": user_name,
            "year": year,
            "run_label": run_label,
            # This is the output-directory tag chosen for this derivation run
            # (--dy_sample on the CLI), *not* the physical DY MC sample(s)
            # actually read - those are resolved independently from
            # sample_config below and recorded in matched_dy_processes.
            "dy_sample_label": args.dy_sample,
            "input_path": args.input_path,
            "stage1_base_path": base_path,
            "sample_config": args.sample_config,
            "matched_data_dirs": matched_data_dirs,
            "matched_dy_processes": matched_dy_processes,
            "missing_dy_processes": missing_dy_processes,
            "git_commit": git_state["commit"],
            "git_dirty": git_state["dirty"],
            "git_diff_file": git_state["diff_file"],
        }
        dy_events = dak.from_parquet(dy_paths)
        is_flag_skim = "pass_sel_official" in data_events.fields
        if is_flag_skim and not args.flag_skim_selection:
            raise ValueError("flag-skim stage-1 input needs --flag_skim_selection (one row per muon pair otherwise)")
        dy_zpt_cols = [c for c in ZPT_WGT_COLUMNS if c in dy_events.fields]
        print(f"Z-pT weight divided out of DY: {dy_zpt_cols[0] if dy_zpt_cols else 'none (column absent)'}")
        provenance["flag_skim_selection"] = args.flag_skim_selection
        provenance["subtract_nondy"] = args.subtract_nondy
        provenance["dy_zpt_weight_divided_out"] = dy_zpt_cols[0] if dy_zpt_cols else None

        samples = {"data": data_events, "dy": dy_events}
        if args.subtract_nondy:
            bkg_groups = get_sample_dict(yaml_path=args.sample_config, section="background", year=str(year))
            nondy_processes = [p for g, ps in bkg_groups.items() if g not in ("DY", "DYVBF") for p in ps]
            nondy_paths, matched_nondy, missing_nondy = collect_process_paths(base_path, nondy_processes)
            print(f"Non-DY processes subtracted from data: {matched_nondy} (missing: {missing_nondy})")
            provenance["matched_nondy_processes"] = matched_nondy
            provenance["missing_nondy_processes"] = missing_nondy
            samples["nondy"] = dak.from_parquet(nondy_paths)
        with open(f"{plot_path}/provenance.json", "w") as prov_file:
            json.dump(provenance, prov_file, indent=2)

        # apply z-peak region filter (and the flag-skim selection) and nothing else; one compute per sample
        njet_field = "njets_nominal"
        fields2load = ["wgt_nominal", "dimuon_pt", njet_field]
        arrays = {}
        for name, events in samples.items():
            if args.flag_skim_selection:
                events = apply_flag_skim(events, args.flag_skim_selection)
            _, events = selection.filterRegion(events, region="z-peak")
            if name == "dy" and not dy_zpt_cols:
                print("WARNING: DY has no Z-pT weight column; assuming no Z-pT weight was applied at stage-1")
            computed = zipAndCompute(events, fields2load)
            arrays[name] = {field: ak.to_numpy(computed[field]) for field in fields2load}

        for njet in [0,1,2]:
            sel = {}
            for name, arr in arrays.items():
                mask = (arr[njet_field] == njet) if njet != 2 else (arr[njet_field] >= njet)
                sel[name] = {field: arr[field][mask] for field in ("wgt_nominal", "dimuon_pt")}
            data_dict, dy_dict = sel["data"], sel["dy"]

            # print(f"data_dict: {data_dict}")
            # print(f"dy_dict: {dy_dict}")

            # binning = config["rewgt_binning"]
            # # Convert the list of bin edges to a C-style array
            # # binning_array = np.array(binning)
            binning_array = np.linspace(0, 200, 2001)  # 2000 bins from 0 to 200 GeV
            # binning_array = np.linspace(0, 200, 80)  # 2000 bins from 0 to 200 GeV

            # Step 2: Create the histogram with variable bin widths
            hist_data = ROOT.TH1F("hist_data", "Data", len(binning_array) - 1, binning_array)
            hist_dy = ROOT.TH1F("hist_dy", "DY", len(binning_array) - 1, binning_array)
            # Sumw2 must be enabled *before* filling with per-event weights, otherwise
            # ROOT falls back to sqrt(bin content) bin errors, which is wrong for a
            # weighted MC histogram (and biases the Data/DY ratio errors used by every
            # downstream fit).
            hist_data.Sumw2()
            hist_dy.Sumw2()

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

            extra_hists = []
            if args.subtract_nondy:
                # fit target becomes (Data - non-DY)/DY; downstream steps read "hist_data"
                hist_nondy = ROOT.TH1F("hist_nondy", "non-DY MC", len(binning_array) - 1, binning_array)
                hist_nondy.Sumw2()
                values = array('d', sel["nondy"]["dimuon_pt"])
                weights = array('d', sel["nondy"]["wgt_nominal"])
                hist_nondy.FillN(len(values), values, weights)
                hist_data_raw = hist_data.Clone("hist_data_raw")
                hist_data.Add(hist_nondy, -1.0)
                extra_hists = [hist_data_raw, hist_nondy]
                print(
                    f"[{year} njet{njet}] raw Data={hist_data_raw.Integral():.6g}, non-DY MC={hist_nondy.Integral():.6g} "
                    f"({hist_nondy.Integral() / hist_data_raw.Integral():.4f} of data), subtracted"
                )

            data_integral = hist_data.Integral()
            dy_integral = hist_dy.Integral()
            print(
                f"[{year} njet{njet}] Data integral={data_integral:.6g}, "
                f"DY integral={dy_integral:.6g}, "
                f"raw Data/DY ratio={data_integral / dy_integral if dy_integral > 0.0 else float('nan'):.6g}"
            )

            # Step 4: Get the Scale Factor (SF) histogram directly from Data / DY.
            hist_SF = hist_data.Clone("hist_SF")
            hist_SF.Divide(hist_dy)

            # Step 5: save the histograms in workspace
            workspace = ROOT.RooWorkspace("zpt_Workspace")

            # Import the histograms into the workspace
            getattr(workspace, "import")(hist_data)  # Use getattr to call 'import' (Python keyword)
            getattr(workspace, "import")(hist_dy)
            for h in extra_hists:
                getattr(workspace, "import")(h)
            getattr(workspace, "import")(hist_SF)

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

            # plot data and dy histograms together using TRatioPlot
            canvas = ROOT.TCanvas("canvas", f"Data vs DY {run_label}", 800, 800)
            ratio_plot = ROOT.TRatioPlot(hist_data, hist_dy)
            ratio_plot.SetH1DrawOpt("E")  # Draw data with error bars
            ratio_plot.SetH2DrawOpt("E")  # Draw DY with error bars

            # y-range set auto based on minimum and maximum of the ratio histogram
            min_ratio = hist_SF.GetMinimum()
            max_ratio = hist_SF.GetMaximum()

            ratio_plot.Draw()

            # ---------------------------
            # LOWER PAD (ratio)
            # ---------------------------
            # NOTE: Set min/max or title should go after `ratio_plot.Draw()`
            ratio_plot.GetLowerRefYaxis().SetTitle("Data / DY")
            ratio_plot.GetLowerRefXaxis().SetTitle("Dimuon pT (GeV)")
            ratio_plot.GetLowerRefGraph().SetMinimum(max(min_ratio * 0.8, 0.0));
            ratio_plot.GetLowerRefGraph().SetMaximum(min(max_ratio * 1.2, 5.0));

            # ---------------------------
            # UPPER PAD (Data + DY)
            # ---------------------------
            upper_pad = ratio_plot.GetUpperPad()
            upper_pad.cd()

            max_data = hist_data.GetMaximum()
            max_dy   = hist_dy.GetMaximum()

            ymax = max(max_data, max_dy)
            ymax *= 1.3  # 30% headroom

            # Set range on upper histogram axis
            ratio_plot.GetUpperRefYaxis().SetRangeUser(0.0, ymax)

            canvas.Update()

            # add legend
            legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Legend coordinates (x1, y1, x2, y2)
            legend.AddEntry(hist_data, "Data", "l")  # "l" means line style
            legend.AddEntry(hist_dy, "DY", "l")
            legend.Draw()

            # Save the canvas as an image
            canvas.SaveAs(f"{plot_path}/dataOverDY_{year}_njet{njet}.pdf")
            canvas.SaveAs(f"{plot_path}/dataOverDY_{year}_njet{njet}.png")

    if client is not None:
        close_dask_client()
