import os
import sys
import argparse
import json
import logging
import time
import glob
import copy
import numpy as np
import awkward as ak
import dask_awkward as dak
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import mplhep as hep
from collections import OrderedDict
from distributed import Client
from rich import print
import tqdm

from modules.utils import logger, ifPathExists

# CMS style and plotting imports
import cmsstyle as CMS
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main_dir)
from src.lib.histogram.plotting import plotDataMC_compare

# Define groupings
GROUPS = {
    "data": [f"data_{x}" for x in "ABCDEFGH"],
    "DY": ["dy_M-50", "dy_M-100To200", "dy_m105_160_amc", "dy_M-100To200_MiNNLO", "dy_M-50_MiNNLO"],
    "TT": ["ttjets_dl", "ttjets_sl"],
    "ST": ["st_tw_top", "st_tw_antitop", "st_t_top", "st_t_antitop"],
    "VV": ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"],
    "EWK": ["ewk_lljj_mll50_mjj120"],
    "OTHER": ["www", "wwz", "wzz", "zzz"],
    "ggH": ["ggh_powhegPS"],
    "VBF": ["vbf_powheg_dipole"]
}

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", default="2018", help="year: 2016preVFP, 2016postVFP, 2017, 2018")
    parser.add_argument("--label", dest="label", help="output label")
    parser.add_argument("--in", dest="in_path", help="input directory")
    parser.add_argument("-data", dest="data_samples", nargs="*", default=list("ABCDEFGH"))
    parser.add_argument("-bkg", dest="bkg_samples", nargs="*", default=['DY','TT','ST','VV','EWK','OTHER'])
    parser.add_argument("-sig", dest="sig_samples", nargs="*", default=[])
    parser.add_argument("--use_gateway", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_setting", default="../validation/zpt_rewgt/plot_settings_Zpt_reWgt.json")
    parser.add_argument("--xcache", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--zpt_on", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("-aod_v", "--NanoAODv", type=int, default=9, choices=[9, 12])
    parser.add_argument("--run2_rereco", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--log-level", default=logging.INFO, type=lambda x: getattr(logging, x))
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    return parser

def gather_processes(data, bkg, sig):
    procs = []
    procs.extend([f"data_{x.upper()}" for x in data])
    for bk in bkg:
        procs.extend(GROUPS.get(bk.upper(), []))
    for sg in sig:
        procs.extend(GROUPS.get(sg.upper(), []))
    return procs

def setup_dask_client(use_gateway):
    if use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        client = gateway.connect(gateway.list_clusters()[0].name).get_client()
        logger.info("Using Dask Gateway client")
    else:
        client = Client(n_workers=31, threads_per_worker=1, memory_limit='4GiB')
        logger.info("Using Local Dask client")
    return client

def load_parquet_samples(processes, path, variables2plot_orig):
    loaded_events = {}
    for proc in tqdm.tqdm(processes):
        paths = glob.glob(os.path.join(path, proc, "*/*.parquet")) or glob.glob(os.path.join(path, proc, "*.parquet"))
        if not paths:
            logger.warning(f"No files found for {proc}. Skipping.")
            continue
        try:
            events = dak.from_parquet(paths)
        except Exception as e:
            logger.warning(f"Failed loading {proc}: {e}")
            continue
        fields = variables2plot_orig + [
            "wgt_nominal", "nBtagLoose_nominal", "nBtagMedium_nominal",
            "dimuon_mass", "zeppenfeld_nominal", "jj_mass_nominal",
            "jet1_pt_nominal", "jj_dEta_nominal", "dimuon_pt", "njets_nominal"
        ]
        if "dy" in proc.lower():
            fields.append("separate_wgt_zpt_wgt")
        schema = pq.read_schema(paths[0])
        fields = list(set(fields).intersection(schema.names))
        events = events[fields]
        events = ak.zip({f: events[f] for f in events.fields}).compute()
        loaded_events[proc] = events
    return loaded_events

def main():
    parser = create_parser()
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    year = args.year
    lumi = {"2018":59.83, "2017":41.48, "2016postVFP":19.5, "2016preVFP":16.81}[year]
    region = "signal"
    load_path = os.path.join(args.in_path, f"stage1_output/{year}/f1_0")
    ifPathExists(load_path)
    ifPathExists(args.plot_setting)

    # vars2plot = ['dimuon', 'mu', 'jet']
    vars2plot = ['dimuon']
    var_exp = []
    for var in vars2plot:
        if var == "dimuon":
            # var_exp += [f"{var}_mass", f"{var}_pt", f"{var}_eta", "mmj_min_dPhi_nominal", "mmj_min_dEta_nominal", "rpt_nominal", "ll_zstar_log_nominal"]
            var_exp += [f"{var}_mass", f"{var}_pt", f"{var}_eta"]
        elif var == "jet":
            var_exp += ["njets_nominal"] + [f"jet{i}_{k}_nominal" for i in (1, 2) for k in ('pt', 'eta', 'phi')]
        elif var == "mu":
            var_exp += [f"mu{i}_{k}" for i in (1, 2) for k in ('pt', 'eta', 'phi')]

    if "dimuon_mass" in var_exp:
        var_exp = ["dimuon_mass_zpeak"] + var_exp

    available_procs = gather_processes(args.data_samples, args.bkg_samples, args.sig_samples)
    client = setup_dask_client(args.use_gateway)
    loaded = load_parquet_samples(available_procs, load_path, var_exp)
    logger.info("Finished loading all samples")

    # plotting logic would continue here...
    with open(args.plot_setting) as f:
        plot_settings = json.load(f)

    plt.style.use(hep.style.CMS)

    for var in var_exp:
        logger.info(f"Preparing to plot: {var}")
        if var not in plot_settings:
            logger.warning(f"Skipping {var} — not found in plot settings")
            continue

        binning = np.linspace(*plot_settings[var].get("binning_linspace", [0, 100, 20]))
        x_title = plot_settings[var].get("xlabel", var)
        y_title = plot_settings[var].get("ylabel", "Events")
        do_logscale = plot_settings[var].get("logscale", True)

        # Build data_dict and bkg_MC_dict (dummy logic for now)
        data_dict = {"values": np.array([]), "weights": np.array([])}
        bkg_MC_dict = OrderedDict()
        sig_MC_dict = OrderedDict()

        for proc, events in loaded.items():
            if var not in events.fields:
                logger.warning(f"{var} not in {proc}, skipping")
                continue

            values = ak.to_numpy(ak.fill_none(events[var], -999))
            weights = ak.to_numpy(ak.fill_none(events.get("wgt_nominal", np.ones_like(values)), 0))
            values = values[values != -999]
            weights = weights[:len(values)]
            if len(values) == 0:
                logger.warning(f"{proc} - all values filtered out for {var}")
                continue


            if proc in GROUPS["data"]:
                data_dict["values"] = np.concatenate([data_dict["values"], values]) if data_dict["values"].size else values
                data_dict["weights"] = np.concatenate([data_dict["weights"], weights]) if data_dict["weights"].size else weights
            elif proc in GROUPS.get("DY", []) + GROUPS.get("TT", []) + GROUPS.get("ST", []) + GROUPS.get("VV", []) + GROUPS.get("EWK", []) + GROUPS.get("OTHER", []):
                label = [k for k, v in GROUPS.items() if proc in v][0]
                if label not in bkg_MC_dict:
                    bkg_MC_dict[label] = {"values": values, "weights": weights}
                else:
                    bkg_MC_dict[label]["values"] = np.concatenate([bkg_MC_dict[label]["values"], values])
                    bkg_MC_dict[label]["weights"] = np.concatenate([bkg_MC_dict[label]["weights"], weights])
            elif proc in GROUPS.get("ggH", []) + GROUPS.get("VBF", []):
                label = [k for k, v in GROUPS.items() if proc in v][0]
                sig_MC_dict[label] = {"values": values, "weights": weights}

        save_dir = os.path.join("plots_output", args.year, args.label or "", var)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{var}.pdf")

        if (
            data_dict["values"].size == 0 and
            all(len(group["values"]) == 0 for group in bkg_MC_dict.values())
        ):
            logger.warning(f"Skipping plot for {var}: no data or background entries.")
            continue

        plotDataMC_compare(
            binning,
            data_dict,
            bkg_MC_dict,
            save_path,
            sig_MC_dict=sig_MC_dict,
            x_title=x_title,
            y_title=y_title,
            lumi=lumi,
            log_scale=do_logscale,
            status="Private Work"
        )

if __name__ == "__main__":
    main()
