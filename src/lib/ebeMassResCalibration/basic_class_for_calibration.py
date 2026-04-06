"""
Collection of basic functions  for the mass resolution calibration
"""
import numpy as np
import ROOT as rt
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


import dask

import yaml
import fnmatch
from copy import deepcopy

from contextlib import contextmanager

import correctionlib
import ROOT

from modules.utils import logger
from modules.correctionlib_file_cache import get_corrset, get_corr_input_names

# surpress RooFit printout
rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)

# ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gErrorIgnoreLevel = ROOT.kFatal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT.gSystem.Load(f"{CURRENT_DIR}/PDFs/RooCMSShape_cc.so")


# Configuration constants
CONFIG = {
    "n_workers": 16,
    "threads_per_worker": 1,
    "memory_limit": "8 GiB",
    "zcr_filter_range": (75, 105),
    "nbins": 120,
    "fields_of_interest": ["mu1_pt", "mu1_eta", "mu2_eta", "dimuon_mass", "wgt_nominal"],
    "fields_with_errors": ["mu1_pt", "mu1_ptErr", "mu2_pt", "mu2_ptErr", "mu1_eta", "mu2_eta", "dimuon_mass", "wgt_nominal"],
}


@contextmanager
def timed(msg):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info(f"[TIMER] {msg}: {dt:.3f} s")


def filter_region(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region == "h-peak":
        region_filter = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region == "h-sidebands":
        region_filter = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region == "signal":
        region_filter = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region == "z-peak" or region == "z_peak":
        region_filter = (dimuon_mass >= 75) & (dimuon_mass <= 105.0)
        # region_filter = (dimuon_mass >= 80) & (dimuon_mass <= 100.0)
        # region_filter = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)
    return events[region_filter]


# Define the calibration categories ---
def get_calib_categories(events):
    """
    Returns a dictionary of 30 boolean masks based on muon1_eta, muon2_eta and mu1_pt.
    Eta bins:
      B: |eta| <= 0.9
      O: 0.9 < |eta| <= 1.8
      E: 1.8 < |eta| <= 2.4
    pT bins for mu1_pt:
      Bin1: (30, 45]
      Bin2: (45, 52]
      Bin3: (52, 62]
      Bin4: (62, 200]

    For the lowest pT bin, the eta combinations are merged into three groups.
    For the other bins, each of the nine eta combinations is kept separately.
    Referece: HIG-19-006, AN-19-124
    """
    BB = ((np.abs(events["mu1_eta"])<=0.9) & (np.abs(events["mu2_eta"])<=0.9))
    BO = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"]) <=1.8)))
    BE = ((np.abs(events["mu1_eta"])<=0.9) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"]) <=2.4)))
    OB = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & (np.abs(events["mu2_eta"])<=0.9))
    OO = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    OE = (((np.abs(events["mu1_eta"])>0.9) & (np.abs(events["mu1_eta"])<=1.8)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))
    EB = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & (np.abs(events["mu2_eta"])<=0.9))
    EO = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>0.9) & (np.abs(events["mu2_eta"])<=1.8)))
    EE = (((np.abs(events["mu1_eta"])>1.8) & (np.abs(events["mu1_eta"])<=2.4)) & ((np.abs(events["mu2_eta"])>1.8) & (np.abs(events["mu2_eta"])<=2.4)))

    # pT bins for mu1_pt
    mask_30_45 = (events["mu1_pt"] > 30) & (events["mu1_pt"] <= 45)
    mask_45_52 = (events["mu1_pt"] > 45) & (events["mu1_pt"] <= 52)
    mask_52_62 = (events["mu1_pt"] > 52) & (events["mu1_pt"] <= 62)
    mask_62_200 = (events["mu1_pt"] > 62) & (events["mu1_pt"] <= 200)

    # For pT bin 30-45, group the eta combinations into three categories.
    # cat_30_45_1 = mask_30_45 & (BB | OB | EB)
    # cat_30_45_2 = mask_30_45 & (BO | OO | EO)
    # cat_30_45_3 = mask_30_45 & (BE | OE | EE)
    cats_30_45 = {
        "30-45_BB": mask_30_45 & BB,
        "30-45_BO": mask_30_45 & BO,
        "30-45_BE": mask_30_45 & BE,
        "30-45_OB": mask_30_45 & OB,
        "30-45_OO": mask_30_45 & OO,
        "30-45_OE": mask_30_45 & OE,
        "30-45_EB": mask_30_45 & EB,
        "30-45_EO": mask_30_45 & EO,
        "30-45_EE": mask_30_45 & EE,
    }

    # For the remaining bins, each eta combination is its own category.
    cats_45_52 = {
        "45-52_BB": mask_45_52 & BB,
        "45-52_BO": mask_45_52 & BO,
        "45-52_BE": mask_45_52 & BE,
        "45-52_OB": mask_45_52 & OB,
        "45-52_OO": mask_45_52 & OO,
        "45-52_OE": mask_45_52 & OE,
        "45-52_EB": mask_45_52 & EB,
        "45-52_EO": mask_45_52 & EO,
        "45-52_EE": mask_45_52 & EE,
    }

    cats_52_62 = {
        "52-62_BB": mask_52_62 & BB,
        "52-62_BO": mask_52_62 & BO,
        "52-62_BE": mask_52_62 & BE,
        "52-62_OB": mask_52_62 & OB,
        "52-62_OO": mask_52_62 & OO,
        "52-62_OE": mask_52_62 & OE,
        "52-62_EB": mask_52_62 & EB,
        "52-62_EO": mask_52_62 & EO,
        "52-62_EE": mask_52_62 & EE,
    }

    cats_62_200 = {
        "62-200_BB": mask_62_200 & BB,
        "62-200_BO": mask_62_200 & BO,
        "62-200_BE": mask_62_200 & BE,
        "62-200_OB": mask_62_200 & OB,
        "62-200_OO": mask_62_200 & OO,
        "62-200_OE": mask_62_200 & OE,
        "62-200_EB": mask_62_200 & EB,
        "62-200_EO": mask_62_200 & EO,
        "62-200_EE": mask_62_200 & EE,
    }

    # categories = {
    #     "30-45_BB_OB_EB": cat_30_45_1,
    #     "30-45_BO_OO_EO": cat_30_45_2,
    #     "30-45_BE_OE_EE": cat_30_45_3
    # }
    categories = cats_30_45
    # categories.update(cats_30_45)
    categories.update(cats_45_52)
    categories.update(cats_52_62)
    categories.update(cats_62_200)

    return categories

CLOSURE_BINS_Run2_AN = [
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.3, 1.4),
    (1.4, 1.5),
    (1.5, 1.7),
    (1.7, 2.0),
    (2.0, 2.5),
    (2.5, 3.5),
]

CLOSURE_BINS = [ # Merge the first three bins due to lack of statistics.
    (0.6, 0.9),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.3, 1.4),
    (1.4, 1.5),
    (1.5, 1.7),
    (1.7, 2.0),
    (2.0, 2.5),
    (2.5, 3.5),
]

# Plotting range for each closure bin
RANGE = {
    1: (0.5, 1.0),
    2: (0.5, 1.0),
    3: (0.5, 1.0),
    4: (0.5, 1.3),
    5: (0.5, 1.3),
    6: (0.5, 1.5),
    7: (0.5, 1.8),
    8: (0.5, 2.0),
    9: (0.5, 2.0),
    10: (1.0, 2.5),
    11: (1.0, 3.0),
    12: (1.5, 4.0),
}


def load_fit_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_cat_config(cfg: dict, cat_idx: str) -> dict:
    """
    Merge global + best-matching overrides into one dict.
    Precedence:
      global < wildcard overrides (in file order) < exact override
    """
    base = deepcopy(cfg.get("global", {}))
    overrides = cfg.get("overrides", {}) or {}

    # apply wildcard overrides in insertion order
    for key, block in overrides.items():
        if "*" in key or "?" in key or "[" in key:
            if fnmatch.fnmatch(cat_idx, key):
                logger.debug(f"Checking override key: {key} for cat_idx: {cat_idx}")
                base = deep_merge(base, block)

    # exact override wins last
    if cat_idx in overrides:
        base = deep_merge(base, overrides[cat_idx])

    return base

def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def apply_roorealvar_cfg(var, pcfg: dict):
    """
    pcfg supports: val, min, max, const
    Only apply keys that exist.
    """
    if pcfg is None:
        return var
    if "min" in pcfg and "max" in pcfg:
        var.setRange(float(pcfg["min"]), float(pcfg["max"]))
    if "val" in pcfg:
        var.setVal(float(pcfg["val"]))
    if "const" in pcfg:
        var.setConstant(bool(pcfg["const"]))
    return var


def weighted_median(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return x[np.searchsorted(cdf, 0.5)]


def plot_histogram(data, bins, range, xlabel, ylabel, title, output_path, median=None):
    plt.figure()
    plt.hist(data, bins=bins, range=range, color="C0", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if median is not None:
        plt.axvline(
            median,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.4f}",
        )
        plt.legend()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def _add_calibration_column(df, correction):
    cal = correction.evaluate(
        df["mu1_pt"].values,
        np.abs(df["mu1_eta"].values),
        np.abs(df["mu2_eta"].values),
    )

    df = df.assign(calibration=cal)
    return df

def closure_test_resolution_binning(
    ddf,
    output_dir,
    CalibrationFactorJSONFile,
    pdfFile_ExtraText="",
    ifbinned=True,
    fix_bin=None,
):
    """
    Validate calibration using bins in predicted per-event resolution (Table-32 style).
    For each resolution bin:
      - fit Z mass -> sigma_fit
      - compute median predicted resolution (before/after)
    """
    logger.info("Starting closure test in resolution binning...")
    os.makedirs(output_dir, exist_ok=True)

    # load correction
    cset = get_corrset(CalibrationFactorJSONFile)
    corr = cset["BS_ebe_mass_res_calibration"]

    # build predicted resolutions in a single compute
    logger.info("Computing predicted resolutions...")
    ddf2 = ddf.map_partitions(_add_calibration_column, correction=corr)

    ddf2 = ddf2.assign(
        dpt1=(ddf2["mu1_ptErr"] / ddf2["mu1_pt"]) * (ddf2["dimuon_mass"] / 2.0),
        dpt2=(ddf2["mu2_ptErr"] / ddf2["mu2_pt"]) * (ddf2["dimuon_mass"] / 2.0),
    ).assign(
        sigma_pred_noncal=lambda x: np.sqrt(x["dpt1"] ** 2 + x["dpt2"] ** 2),
        sigma_pred_cal=lambda x: x["sigma_pred_noncal"] * x["calibration"],
    )

    logger.info("Computing dataframe for closure test...")
    # FIXME: for some reason ddf2.compute() is not running on gateway.
    with timed("Computing Dask dataframe"):
        with dask.config.set(scheduler="threads"):
            if fix_bin is not None:
                logger.info(f"Computing only for fixed bin {fix_bin}...")
                lo, hi = CLOSURE_BINS[int(fix_bin)-1]
                logger.debug(f"Fixed bin range: [{lo}, {hi})")
                ddf2 = ddf2[(ddf2["sigma_pred_cal"] >= lo) & (ddf2["sigma_pred_cal"] < hi)]
            df = ddf2.compute()

    logger.info("Dataframe computed.")
    rows = []
    for i, (lo, hi) in enumerate(CLOSURE_BINS, start=1):
        if fix_bin is not None and i != int(fix_bin):
            logger.warning(f"Skipping bin {i} as per fix_bin={fix_bin}")
            continue
        mask = (df["sigma_pred_cal"] >= lo) & (df["sigma_pred_cal"] < hi)
        df_bin = df[mask]
        if df_bin.empty:
            logger.warning(f"Closure bin {i} [{lo},{hi}) empty, skipping.")
            continue

        # predicted medians
        weights = df_bin["wgt_nominal"].to_numpy() if "wgt_nominal" in df_bin else None
        if np.any(weights < 0):
            logger.warning("Negative weights detected in closure bin!")  
        if weights is not None:
            med_noncal = weighted_median(df_bin["sigma_pred_noncal"], weights)
            med_cal = weighted_median(df_bin["sigma_pred_cal"], weights)
        else:
            med_noncal = df_bin["sigma_pred_noncal"].median()
            med_cal = df_bin["sigma_pred_cal"].median()

        # plot mass resolution distribution both calibrated and non-calibrated
        plt.figure(figsize=(8, 6))
        plt.hist(df_bin["sigma_pred_cal"], bins=CONFIG["nbins"], range=RANGE[i], weights=weights, color='C0', alpha=0.7, label="Calibrated")
        plt.hist(df_bin["sigma_pred_noncal"], bins=CONFIG["nbins"], range=RANGE[i], weights=weights, color='C1', alpha=0.7, label="Non-Calibrated")
        plt.xlabel("Dimuon mass resolution (GeV)")
        plt.ylabel("Events")
        plt.title(f"Category {i}\n Median Cal = {med_cal:.4f} GeV, Median NonCal = {med_noncal:.4f} GeV")
        plt.legend()

        plt.axvline(med_cal, color="red", linestyle="dashed", linewidth=2, label=f"Median Cal: {med_cal:.4f}")
        plt.axvline(med_noncal, color="blue", linestyle="dashed", linewidth=2, label=f"Median NonCal: {med_noncal:.4f}")
        plt.legend()
        plt.savefig(f"{output_dir}/mass_resolution_resBin{i}_Calibrated_{pdfFile_ExtraText}.pdf")
        plt.close()

        # measured sigma from Z-mass fit in THIS bin
        mass = df_bin["dimuon_mass"].to_numpy()
        weights = df_bin["wgt_nominal"].to_numpy() if "wgt_nominal" in df_bin else None
        if weights is not None and np.any(weights < 0):
            logger.warning("Negative weights detected in closure bin!")        
        df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])
        cat_name = f"resBin{i}"
        with timed(f"Fitting Z mass for closure bin {i}..."):
            df_fit = generateBWxDCB_RooCMSShape_plot(
                mass,
                weights,
                cat_name,
                nbins=CONFIG["nbins"],
                df_fit=df_fit,
                output_dir=output_dir,
                logfile=f"ClosureLog_{pdfFile_ExtraText}.txt",
                ifbinned=ifbinned,
                pdfFile_ExtraText=pdfFile_ExtraText,
                fit_cfg_path=f"{CURRENT_DIR}/fit_config.yml",
            )
        sigma_fit = float(df_fit.loc[df_fit["cat_name"] == cat_name, "fit_val"].iloc[0])
        sigma_err = float(df_fit.loc[df_fit["cat_name"] == cat_name, "fit_err"].iloc[0])

        rows.append(
            {
                "cat_name": i,
                "bin_low": lo,
                "bin_high": hi,
                "nEvents": float(np.sum(weights)) if weights is not None else len(df_bin),
                "fit_val": sigma_fit,
                "fit_err": sigma_err,
                "median_val_NonCal": med_noncal,
                "median_val": med_cal,  # after-cal prediction
            }
        )

    logger.info("Creating output dataframe for closure test...")
    df_out = pd.DataFrame(rows)
    # df_out.to_csv(f"{output_dir}/closure_results_resolutionBinning.csv", index=False)
    return df_out


def save_fit_params_to_json( inputFilePath, ifbinned, fit_result, cat_idx, json_path, model_name="BWxDCB", chi2_val=None):
    param_dict = {}
    sigma_val = None
    sigma_err = None

    for i in range(fit_result.floatParsFinal().getSize()):
        p = fit_result.floatParsFinal().at(i)
        param_dict[p.GetName()] = {
            "val": p.getVal(),
            "err": p.getError(),
            "const": p.isConstant(),
            "min": p.getMin(),
            "max": p.getMax()
        }
        if p.GetName().lower() == "sigma":
            sigma_val = p.getVal()
            sigma_err = p.getError()

    # Build full record
    fit_metadata = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nllVal": fit_result.minNll(),
        "status": fit_result.status(),
        "chi2": chi2_val,
        "sigma": sigma_val,
        "sigma_err": sigma_err,
        "params": param_dict
    }

    # Load existing fits
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_fits = json.load(f)
    else:
        all_fits = {}

    all_fits.setdefault("inputFilePath", inputFilePath)
    all_fits.setdefault("ifbinned", ifbinned)
    all_fits[cat_idx] = fit_metadata

    print(f"===> json_path : {json_path}")
    with open(json_path, "w") as f:
        json.dump(all_fits, f, indent=2)


def generateVoigtian_plot(mass_arr, cat_idx: int, nbins, df_fit, logfile="CalibrationLog.txt", output_dir=""):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,80,100)
    mass.setBins(nbins)
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    # workspace.Import(mass)
    frame = mass.frame(Title=f"ZCR Dimuon Mass Voigtian calibration fit for category {cat_idx}")

    # Voigtian --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    sigma = rt.RooRealVar("sigma" , "sigma", 2, 0.5, 2.5)
    bwWidth.setConstant(True)
    model1 = rt.RooVoigtian("signal" , "signal", mass, bwmZ, bwWidth, sigma)

    # # Exp x Erfc Background --------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)

    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 91.2, 75, 105)
    # erfc_coeff = rt.RooRealVar("erfc_coeff" , "erfc_coeff", 0.1, 0, 1.5)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @2) * @1", rt.RooArgList(mass, erfc_coeff, erfc_center))
    # model2_2a = rt.RooFit.bindFunction("erfc", rt.TMath.Erfc, erfc_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erfc","erfc", model2_2a) # turn bound function to pdf
    # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate ExpxErfc bkg

    # Landau Background --------------------------------------------------------------------------
    mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", 95, 90, 150)
    sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 2, 0.5, 8.5)
    model2 = rt.RooLandau("bkg", "bkg", mass, mean_landau, sigma_landau) # generate Landau bkg

    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0, 1.0)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])

    time_step = time.time()
    # fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    _ = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    fit_result = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    logger.info(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    # do plotting
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # name is explicitly defined so chiSquare can find it
    # roo_hist.plotOn(frame, Name="data_hist")
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8])
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    # calculate chi2 and add to plot
    n_free_params = fit_result.floatParsFinal().getSize()
    logger.info(f"n_free_params: {n_free_params}")
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float('%.3g' % chi2) # get upt to 3 sig fig
    logger.info(f"chi2: {chi2}")
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2}")
    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 8)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()
    logger.info(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save to df_fit
    # df_fit.loc[cat_idx] = [sigma.getVal(), sigma.getError()]
    # df_fit = df_fit.append({"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}, ignore_index=True)
    new_row = pd.DataFrame({"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}, index=[0])
    df_fit = pd.concat([df_fit, new_row], ignore_index=True)

    # Save the cat_idx and sigma value to a log file
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    # save plot
    canvas.SaveAs(f"{output_dir}/calibration_fitCat{cat_idx}.pdf")
    del canvas
    # # consider script to wait a second for stability?
    # time.sleep(1)
    return df_fit


def generateBWxDCB_RooCMSShape_plot(
    mass_arr,
    weights,
    cat_idx: str,
    nbins,
    df_fit=None,
    out_string="",
    logfile="CalibrationLog.txt",
    output_dir="",
    ifbinned=True,
    pdfFile_ExtraText="",
    inputFilePath="",
    fit_cfg_path="", fit_cfg=None
):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: str name of specific calibration category the mass_arr is from

    Returns:
        - The df_fit with columns ["cat_name", "fit_val", "fit_err"]
        - Saves the fit plot as a PDF in output_dir
        - Appends fit results to logfile in output_dir
        - Saves fit parameters to a JSON file in output_dir
    """
    logger.info("Starting BWxDCB fit...")
    logger.info(f"cat_idx: {cat_idx}")
    if df_fit is None:
        df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])

    if fit_cfg is None:
        if not fit_cfg_path:
            raise ValueError("Provide fit_cfg_path or fit_cfg dict")
        logger.info(f"Loading fit config from {fit_cfg_path}")
        fit_cfg = load_fit_config(fit_cfg_path)

    logger.info(f"Resolving category config for cat_idx={cat_idx}")
    cat_cfg = resolve_cat_config(fit_cfg, str(cat_idx))

    logger.info(f"Using category config: {cat_cfg}")
    obs = cat_cfg.get("observable", {})
    lo, hi = obs.get("range", [80.0, 100.0])
    fit_lo, fit_hi = obs.get("fit_range", [82.0, 96.0])
    full_lo, full_hi = obs.get("full_range", [lo, hi])
    nbins = int(obs.get("nbins", nbins))

    logger.info(f"Mass range: [{lo}, {hi}], fit_range: [{fit_lo}, {fit_hi}], full_range: [{full_lo}, {full_hi}], nbins: {nbins}")

    mass_name = obs.get("name", "dimuon_mass")
    mass_title = obs.get("title", "mass (GeV)")

    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()

    mass =  rt.RooRealVar(mass_name, mass_title, 100, float(lo), float(hi))
    mass.setBins(nbins)

    # pick the preferred fit window
    mass.setRange("fitRange", float(fit_lo), float(fit_hi))
    mass.setRange("fullRange", float(full_lo), float(full_hi))

    # FFT cache settings
    cache_bins = int(obs.get("fft_cache_bins", 1000))
    cache_lo, cache_hi = obs.get("fft_cache_range", [float(lo), float(hi)])
    mass.setBins(cache_bins,"cache") # This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache",cache_lo)
    mass.setMax("cache",cache_hi)

    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    if weights is not None:
        logger.info("Using weighted RooDataSet")

        wvar = rt.RooRealVar("weight", "weight", 1.0)
        data_dict = {
            mass_name: mass_arr,
            "weight": weights,
        }

        roo_dataset = rt.RooDataSet.from_numpy(
            data_dict,
            [mass, wvar],
            weight_name="weight"
        )
    else:
        roo_dataset = rt.RooDataSet.from_numpy(
            {mass_name: mass_arr},
            [mass]
        )    
    if roo_dataset.numEntries() == 0:
        logger.error(f"No entries in RooDataSet for category {cat_idx}. Skipping.")
        return df_fit

    frame = mass.frame(Title=f"ZCR Dimuon Mass BWxDCB + RooCMSShape calibration fit for category {cat_idx}")

    pc = cat_cfg.get("params", {})
    # BWxDCB --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    apply_roorealvar_cfg(bwmZ, pc.get("bwz_mZ"))

    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    apply_roorealvar_cfg(bwWidth, pc.get("bwz_Width"))

    model1_1 = rt.RooBreitWigner("bwz", "BWZ",mass, bwmZ, bwWidth)

    """
    Note from Jan: sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can
    put whatevere restrictions we want.
    """
    mean = rt.RooRealVar("mean" , "mean", 0, -10, 10) # mean is mean relative to BW
    apply_roorealvar_cfg(mean, pc.get("mean"))

    sigma = rt.RooRealVar("sigma" , "sigma", 2, .001, 4.0)
    apply_roorealvar_cfg(sigma, pc.get("sigma"))

    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
    apply_roorealvar_cfg(alpha1, pc.get("alpha1"))

    n1 = rt.RooRealVar("n1" , "n1", 137, 1, 185)
    apply_roorealvar_cfg(n1, pc.get("n1"))

    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
    apply_roorealvar_cfg(alpha2, pc.get("alpha2"))

    n2 = rt.RooRealVar("n2" , "n2",   2, 1, 20)
    apply_roorealvar_cfg(n2, pc.get("n2"))

    model1_2 = rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)

    # merge BW with DCB via convolution
    model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB

    logger.info("Configured BWxDCB model.")


    # Add RooCMSShape Background --------------------------------------------------------------------------
    exp_alpha = rt.RooRealVar("exp_alpha", "#alpha", 101.0, 0.0, 300.0)
    apply_roorealvar_cfg(exp_alpha, pc.get("exp_alpha"))

    exp_beta = rt.RooRealVar("exp_beta", "#beta", 0.15, 0.0, 2.0)
    apply_roorealvar_cfg(exp_beta, pc.get("exp_beta"))

    exp_gamma = rt.RooRealVar("exp_gamma", "#gamma", 0.1, 0.0, 10.0)
    apply_roorealvar_cfg(exp_gamma, pc.get("exp_gamma"))

    exp_peak = rt.RooRealVar("exp_peak", "peak", 91.1876)  # 91.1876
    apply_roorealvar_cfg(exp_peak, pc.get("exp_peak"))

    model2 = rt.RooCMSShape("bkg", "bkg", mass, exp_alpha, exp_beta, exp_gamma, exp_peak)


    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.999, 0.5, 0.99999999)
    apply_roorealvar_cfg(sigfrac, pc.get("sigfrac"))

    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])
    # final_model = model1_2

    time_step = time.time()

    if ifbinned:
        if weights is not None:
            roo_hist = rt.RooDataHist(
                "data_hist",
                "binned weighted dataset",
                rt.RooArgSet(mass),
                roo_dataset,
                1.0  # normalization factor (keep 1)
            )
        else:
            roo_hist = rt.RooDataHist(
                "data_hist",
                "binned dataset",
                rt.RooArgSet(mass),
                roo_dataset
            )

        if roo_hist.numEntries() == 0:
            logger.error(f"No entries in RooDataHist for category {cat_idx}. Skipping.")
            return df_fit
    else:
        roo_hist = roo_dataset

    fit_cfg_block = cat_cfg.get("fit", {})
    stage1 = fit_cfg_block.get("stage1", {})
    stage2 = fit_cfg_block.get("stage2", {})

    def roo_fit_opts(block):
        opts = [rt.RooFit.Save(True), rt.RooFit.EvalBackend("cpu")]
        if "Strategy" in block:
            opts.append(rt.RooFit.Strategy(int(block["Strategy"])))
        if "Minos" in block:
            opts.append(rt.RooFit.Minos(bool(block["Minos"])))
        if "Hesse" in block:
            opts.append(rt.RooFit.Hesse(bool(block["Hesse"])))
        if "Offset" in block:
            opts.append(rt.RooFit.Offset(bool(block["Offset"])))
        if "Range" in block:
            opts.append(rt.RooFit.Range(str(block["Range"])))
        if "PrintLevel" in block:
            opts.append(rt.RooFit.PrintLevel(int(block["PrintLevel"])))
        # add Minos, SumW2Error, Extended, NumCPU similarly if you want
        return opts

    # do fitting
    rt.EnableImplicitMT()

    _ = final_model.fitTo(roo_hist, *roo_fit_opts(stage1))
    fit_result = final_model.fitTo(roo_hist, *roo_fit_opts(stage2))

    logger.info(f"Fit results for category {cat_idx}:")
    fit_result.Print("v")

    n_free_params = fit_result.floatParsFinal().getSize()
    logger.info(f"n_free_params: {n_free_params}")
    # logger.info("Fit status:", fit_result.status())
    # logger.info("CovQual:", fit_result.covQual())
    logger.info("------------------------------")

    # # Save model and variables into RooWorkspace
    # w = rt.RooWorkspace("w", "workspace")
    # getattr(w, 'import')(mass, rt.RooFit.RecycleConflictNodes())
    # getattr(w, 'import')(final_model, rt.RooFit.RecycleConflictNodes())
    # getattr(w, 'import')(fit_result, rt.RooFit.RecycleConflictNodes())

    # # Save to file
    # model_dir = f"{output_dir}/final_models"
    # os.makedirs(model_dir, exist_ok=True)
    # ws_output_path = f"{model_dir}/workspace_cat{cat_idx}.root"
    # w.writeToFile(ws_output_path)
    # logger.info(f"Workspace saved to {ws_output_path}")

    logger.info(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    # do plotting
    # NOTE: Remember to provide "Name" argument to plotOn so that legend and chi2 can find the correct objects
    if weights is not None:
        roo_dataset.plotOn(
            frame,
            rt.RooFit.DataError(rt.RooAbsData.SumW2),
            rt.RooFit.Name("data_hist")
        )
    else:
        roo_dataset.plotOn(
            frame,
            rt.RooFit.Name("data_hist")
        )    
    # roo_hist.plotOn(frame, Name="data_hist") # name is explicitly defined so chiSquare can find it
    final_model.plotOn(frame, Components="signal", Name="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", Name="bkg", LineColor=rt.kRed)
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8],
                                # Label="Fit Result",
                                # Format="NEU", AutoPrecision=1
                                )
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    # NOTE: compute chi2 after all plotOn calls to ensure correct components are drawn
    # calculate chi2 and add to plot
    # chiSquare(pdfName, histName, nFreeParams) returns chi2/ndf
    chi2_per_ndf = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2_per_ndf = float("%.3g" % chi2_per_ndf)  # get up to 3 sig fig
    logger.info(f"chi2_per_ndf: {chi2_per_ndf}")

    n_points = frame.getHist("data_hist").GetN()
    ndf = n_points - n_free_params

    logger.info(f"chi2/ndf = {chi2_per_ndf:.4g}  (n_points={n_points}, n_free={n_free_params}, ndf~{ndf})")

    # If you want an approximate chi2 (not always super meaningful in RooFit, but ok for monitoring):
    chi2 = chi2_per_ndf * ndf if ndf > 0 else float("nan")
    logger.info(f"chi2 ~ {chi2:.4g}")

    print(f"===> output dir: {output_dir}/fit_params.json")
    # store the fit result in a json file
    save_fit_params_to_json(inputFilePath, ifbinned, fit_result, cat_idx, f"{output_dir}/fit_params.json", model_name="BWxDCB+RooCMSShape", chi2_val=chi2)

    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2_per_ndf}")

    # Add legend for components
    legend = rt.TLegend(0.1, 0.75, 0.45, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.AddEntry(frame.findObject("data_hist"), "Data", "lep")
    legend.AddEntry(frame.findObject("final_model"), "Total Fit", "l")
    legend.AddEntry(frame.findObject("signal"), "Signal (BWxDCB)", "l")
    legend.AddEntry(frame.findObject("bkg"), "Background (RooCMSShape)", "l")
    legend.Draw("same")

    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 5)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()
    # add referecne line at 0
    line = rt.TLine(full_lo, 0, full_hi, 0)
    # line.SetNDC()
    line.SetLineColor(rt.kBlack)
    line.SetLineWidth(2)
    line.SetLineStyle(2)
    line.Draw("same")
    # add reference line at +/-2
    line2 = rt.TLine(full_lo, 2, full_hi, 2)
    # line.SetNDC()
    line2.SetLineColor(rt.kBlack)
    line2.SetLineWidth(2)
    line2.SetLineStyle(2)
    line2.Draw("same")
    line3 = rt.TLine(full_lo, -2, full_hi, -2)
    # line.SetNDC()
    line3.SetLineColor(rt.kBlack)
    line3.SetLineWidth(2)
    line3.SetLineStyle(2)
    line3.Draw("same")

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()

    # logger.info(f"mean_landau: {mean_landau.getVal()}")
    # logger.info(f"sigma_landau: {sigma_landau.getVal()}")
    logger.info(f"n1: {n1.getVal()}")
    logger.info(f"n2: {n2.getVal()}")
    logger.info(f"alpha1: {alpha1.getVal()}")
    logger.info(f"alpha2: {alpha2.getVal()}")
    logger.info(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save cat_idx and sigma value to a pandas dataframe
    if not df_fit.empty:
        new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
        df_fit = pd.concat([df_fit, new_row], ignore_index=True)
    else:
        df_fit = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])

    # Save the cat_idx and sigma value to a log file
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    full_path = f"{output_dir}/calibration_fitCat{cat_idx}.pdf"
    if pdfFile_ExtraText:
        full_path = full_path.replace(".pdf", f"_{pdfFile_ExtraText}.pdf")
    canvas.SaveAs(full_path)

    os.makedirs(f"{output_dir}/fits_root", exist_ok=True)
    canvas.SaveAs(f"{output_dir}/fits_root/calibration_fitCat{cat_idx}.root")

    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)
    return df_fit


def generateBWxDCB_plot(
    mass_arr,
    cat_idx: str,
    nbins,
    df_fit=None,
    out_string="",
    logfile="CalibrationLog.txt",
    output_dir="",
    ifbinned=True,
    pdfFile_ExtraText="",
    inputFilePath="",
):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: str name of specific calibration category the mass_arr is from

    Returns:
        - The df_fit with columns ["cat_name", "fit_val", "fit_err"]
        - Saves the fit plot as a PDF in output_dir
        - Appends fit results to logfile in output_dir
        - Saves fit parameters to a JSON file in output_dir
    """
    logger.info("Starting BWxDCB fit...")
    logger.info(f"cat_idx: {cat_idx}")
    if df_fit is None:
        df_fit = pd.DataFrame(columns=["cat_name", "fit_val", "fit_err"])

    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()

    # workspace = rt.RooWorkspace("w", "w")
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,80,100)
    mass.setBins(nbins)

    # pick the preferred fit window
    mass.setRange("fitRange", 82, 100)
    mass.setRange("fullRange", 80,100)

    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    if roo_dataset.numEntries() == 0:
        logger.error(f"No entries in RooDataSet for category {cat_idx}. Skipping.")
        return df_fit

    frame = mass.frame(Title=f"ZCR Dimuon Mass BWxDCB + RooCMSShape calibration fit for category {cat_idx}")

    # BWxDCB --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    # bwmZ.setConstant(True) # Stated in HIG-19-006
    bwWidth.setConstant(True) # Stated in HIG-19-006

    # if (
    #     cat_idx == "30-45_BB_OB_EB"
    #     or cat_idx == "30-45_BO_OO_EO"
    #     or cat_idx == "30-45_BE_OE_EE"
    #     or cat_idx == "30-45_BO"
    #     # or cat_idx == "30-45_EE"
    #     # or cat_idx == "30-45_BB"
    # ):
    #     """
    #     FIXME: Added this condition for 2018 data, because the fit was not converging
    #     """
    #     bwWidth.setConstant(False)
    #     bwmZ.setConstant(False)
    # if (
    #     cat_idx == "30-45_EE"
    #     # or cat_idx == "30-45_BB"
    #     or cat_idx == "resBin1"
    # ):
    #     print("=====> Setting both bwmZ and bwWidth to constant for stability")
    #     bwmZ.setConstant(True)
    #     bwWidth.setConstant(True)

    model1_1 = rt.RooBreitWigner("bwz", "BWZ",mass, bwmZ, bwWidth)

    """
    Note from Jan: sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can
    put whatevere restrictions we want.
    """
    mean = rt.RooRealVar("mean" , "mean", 0, -10,10) # mean is mean relative to BW
    # mean = rt.RooRealVar("mean" , "mean", 100, 95,110) # test
    # NOTE: The lower bound on sigma was intentionally loosened from 0.1 to 0.001
    # to allow very narrow resolution values in some categories / years where the
    # fit would otherwise hit the boundary and fail to converge. This increases the
    # risk of unphysically small sigmas, so fitted results should be monitored and,
    # if needed, additional validation or tighter bounds should be applied downstream.
    sigma = rt.RooRealVar("sigma" , "sigma", 2, .001, 4.0)
    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
    n1 = rt.RooRealVar("n1" , "n1", 10, 0.01, 185)
    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
    n2 = rt.RooRealVar("n2" , "n2", 25, 0.01, 385)
    # n2 = rt.RooRealVar("n2" , "n2", 114, 0.01, 385) #test 114
    # n1.setConstant(True)
    # n2.setConstant(True)
    # if ("EE" in cat_idx
    #     or cat_idx == "resBin1"
    # ):
    #     n1.setConstant(False)
    #     n2.setConstant(False)
    #     alpha1.setRange(0.2, 5)  # don't fix it too low
    #     alpha2.setRange(0.2, 5)
    # mean.setRange(-2, 2)  # instead of full -10 to 10
    # # alpha1.setRange(0.1, 10)
    # alpha1.setVal(6.11551)
    # alpha1.setConstant(True)

    # # alpha2.setRange(0.1, 10)
    # alpha2.setVal(6.78)
    # alpha2.setConstant(True)

    model1_2 = rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)

    # merge BW with DCB via convolution
    model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB

    mass.setBins(10000,"cache") # This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache",50.5)
    mass.setMax("cache",130.5)

    # Add RooCMSShape Background --------------------------------------------------------------------------
    exp_alpha = rt.RooRealVar("exp_alpha", "#alpha", 101.0, 0.0, 300.0)
    exp_beta = rt.RooRealVar("exp_beta", "#beta", 0.15, 0.0, 2.0)
    exp_gamma = rt.RooRealVar("exp_gamma", "#gamma", 0.1, 0.0, 10.0)
    exp_peak = rt.RooRealVar("exp_peak", "peak", 91.1876,89.0, 93.0)  # 91.1876
    # if (cat_idx == "30-45_BO"
    #     or cat_idx == "30-45_EE"
    #     or cat_idx == "30-45_BB"
    #     ):
    #     exp_gamma.setRange(0.0, 5.0)

    # exp_peak.setConstant(True)
    exp_beta.setVal(0.45)
    # exp_beta.setConstant(True)

    exp_alpha.setVal(66.6)
    exp_alpha.setConstant(True)

    exp_gamma.setVal(0.45)
    exp_gamma.setConstant(True)

    model2 = rt.RooCMSShape("bkg", "bkg", mass, exp_alpha, exp_beta, exp_gamma, exp_peak)

    # # Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", 0.01, 0.00000001, 1)
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    # --------------------------------------------------

    # Landau Background --------------------------------------------------------------------------
    # mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", 90, 70, 200)
    # sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    # model2 = rt.RooLandau("bkg", "bkg", mass, mean_landau, sigma_landau) # generate Landau bkg
    # -----------------------------------------------------

    # neg Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", -0.01, -1,  -0.00000001)
    # shift = rt.RooRealVar("shift", "Offset", 70, 40, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    # --------------------------------------------------

    ## NEW VERSION
    # # # Reverse Landau Background test--------------------------------------------------------------------------
    # mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", -80,  -150, -70) # 80
    # mass_neg = rt.RooFormulaVar("mass_neg", "-@0", [mass])
    # sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    # model2 = rt.RooLandau("bkg", "bkg", mass_neg, mean_landau, sigma_landau) # generate Landau bkg
    # # #-----------------------------------------------------

    # Exp x Erf Background --------------------------------------------------------------------------
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "(@0 - @1)", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)

    # erf_center = rt.RooRealVar("erf_center" , "erf_center", 91.2, 75, 155)
    # erf_in = rt.RooFormulaVar("erf_in", "(@0 - @1)", rt.RooArgList(mass, erf_center))
    # model2_2a = rt.RooFit.bindFunction("erf", rt.TMath.Erf, erf_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erf","erf", model2_2a) # turn bound function to pdf
    # # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate Expxerf bkg

    # -----------------------------------------------------

    # # Exp x Erf Background V2--------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 100, 90, 150)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)
    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 100, 90, 150)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @1)", rt.RooArgList(mass, erfc_center))
    # # both bindPdf and RooGenericPdf work, but one may have better cuda integration over other, so leaving both options
    # # model2_2 = rt.RooFit.bindPdf("erfc", rt.TMath.Erfc, erfc_in)
    # model2_2 = rt.RooGenericPdf("erfc", "TMath::Erf(@0)+1", erfc_in)
    # model2 = rt.RooProdPdf("bkg", "bkg", rt.RooArgList(model2_1, model2_2))
    # -----------------------------------------------------

    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.95, 0.05, 0.99999999)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])
    # final_model = model1_2

    time_step = time.time()

    if ifbinned:
        # fitting directly to unbinned dataset is slow, so first make a histogram
        roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
        if roo_hist.numEntries() == 0:
            logger.error(f"No entries in RooDataHist for category {cat_idx}. Skipping.")
            return df_fit
    else:
        roo_hist = roo_dataset

    # do fitting
    rt.EnableImplicitMT()
    _ = final_model.fitTo(
        roo_hist,
        Save=True,
        # SumW2Error=True,
        EvalBackend="cpu",
        # Extended=True,
        Range="fitRange",
        # PrintLevel=-1,
    )
    # _ = final_model.fitTo(
    #     roo_hist,
    #     Save=True,
    #     SumW2Error=True,
    #     EvalBackend="cpu",
    #     # Extended=True,
    #     Range="fitRange",
    #     # PrintLevel=-1,
    # )
    # Fix all parameters of the signal model but the mean and  sigma of the DSCB
    # for param in rt.RooArgList(model1.getParameters(roo_hist)):
    #     # if param.GetName() != "sigma" and param.GetName() != "mean" and param.GetName() != "sigfrac":
    #     FixPars = ["bwz_Width", "bwz_mZ", "alpha1", "n1", "alpha2", "n2"]
    #     # if param.GetName() != "sigma" and param.GetName() != "mean" and param.GetName() != "bwz_Width" and param.GetName() != "bwz_mZ":
    #     if param.GetName() in FixPars:
    #         param.setConstant(True)
    #     else:
    #         logger.warning(f"Parameter '{param.GetName()}' is not fixed and will be optimized during the fit.")

    fit_result = final_model.fitTo(
        roo_hist,
        Save=True,
        # SumW2Error=True,
        EvalBackend="cpu",
        # Extended=True,  # For binned fit, Extended isn't always helpful
        # Minos=True,
        # Hesse=True,
        # Strategy=2,
        Range="fitRange",
        PrintLevel=-1,
        # NumCPU=25,
    )

    logger.info(f"Fit results for category {cat_idx}:")
    fit_result.Print("v")

    n_free_params = fit_result.floatParsFinal().getSize()
    logger.info(f"n_free_params: {n_free_params}")
    # logger.info("Fit status:", fit_result.status())
    # logger.info("CovQual:", fit_result.covQual())
    logger.info("------------------------------")

    # # Save model and variables into RooWorkspace
    # w = rt.RooWorkspace("w", "workspace")
    # getattr(w, 'import')(mass, rt.RooFit.RecycleConflictNodes())
    # getattr(w, 'import')(final_model, rt.RooFit.RecycleConflictNodes())
    # getattr(w, 'import')(fit_result, rt.RooFit.RecycleConflictNodes())

    # # Save to file
    # model_dir = f"{output_dir}/final_models"
    # os.makedirs(model_dir, exist_ok=True)
    # ws_output_path = f"{model_dir}/workspace_cat{cat_idx}.root"
    # w.writeToFile(ws_output_path)
    # logger.info(f"Workspace saved to {ws_output_path}")

    logger.info(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    # do plotting
    # NOTE: Remember to provide "Name" argument to plotOn so that legend and chi2 can find the correct objects
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # name is explicitly defined so chiSquare can find it
    # roo_hist.plotOn(frame, Name="data_hist") # name is explicitly defined so chiSquare can find it
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", Name="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", Name="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8],
                                # Label="Fit Result",
                                # Format="NEU", AutoPrecision=1
                                )
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    # NOTE: compute chi2 after all plotOn calls to ensure correct components are drawn
    # calculate chi2 and add to plot
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float("%.3g" % chi2)  # get up to 3 sig fig
    logger.info(f"chi2: {chi2}")
    print(f"===> output dir: {output_dir}/fit_params.json")
    # store the fit result in a json file
    save_fit_params_to_json(inputFilePath, ifbinned, fit_result, cat_idx, f"{output_dir}/fit_params.json", model_name="BWxDCB+RooCMSShape", chi2_val=chi2)

    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2}")

    # Add legend for components
    legend = rt.TLegend(0.1, 0.75, 0.45, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.AddEntry(frame.findObject("data_hist"), "Data", "lep")
    legend.AddEntry(frame.findObject("final_model"), "Total Fit", "l")
    legend.AddEntry(frame.findObject("signal"), "Signal (BWxDCB)", "l")
    legend.AddEntry(frame.findObject("bkg"), "Background (RooCMSShape)", "l")
    legend.Draw("same")

    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 5)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()
    # add referecne line at 0
    line = rt.TLine(75, 0, 105, 0)
    # line.SetNDC()
    line.SetLineColor(rt.kBlack)
    line.SetLineWidth(2)
    line.SetLineStyle(2)
    line.Draw("same")
    # add reference line at +/-2
    line2 = rt.TLine(75, 2, 105, 2)
    # line.SetNDC()
    line2.SetLineColor(rt.kBlack)
    line2.SetLineWidth(2)
    line2.SetLineStyle(2)
    line2.Draw("same")
    line3 = rt.TLine(75, -2, 105, -2)
    # line.SetNDC()
    line3.SetLineColor(rt.kBlack)
    line3.SetLineWidth(2)
    line3.SetLineStyle(2)
    line3.Draw("same")

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()

    # logger.info(f"mean_landau: {mean_landau.getVal()}")
    # logger.info(f"sigma_landau: {sigma_landau.getVal()}")
    logger.info(f"n1: {n1.getVal()}")
    logger.info(f"n2: {n2.getVal()}")
    logger.info(f"alpha1: {alpha1.getVal()}")
    logger.info(f"alpha2: {alpha2.getVal()}")
    logger.info(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save cat_idx and sigma value to a pandas dataframe
    if not df_fit.empty:
        new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
        df_fit = pd.concat([df_fit, new_row], ignore_index=True)
    else:
        df_fit = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])

    # Save the cat_idx and sigma value to a log file
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    full_path = f"{output_dir}/calibration_fitCat{cat_idx}.pdf"
    if pdfFile_ExtraText:
        full_path = full_path.replace(".pdf", f"_{pdfFile_ExtraText}.pdf")
    canvas.SaveAs(full_path)

    os.makedirs(f"{output_dir}/fits_root", exist_ok=True)
    canvas.SaveAs(f"{output_dir}/fits_root/calibration_fitCat{cat_idx}.root")

    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)
    return df_fit


def generateBWxDCB_plot_bkgErfxExp(mass_arr, cat_idx: int, nbins, df_fit = "", logfile="CalibrationLog.txt", output_dir=""):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    # if you want TCanvas to not crash, separate fitting and drawing
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx),800, 800) # giving a specific name for each canvas prevents segfault?
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",100,80,100)
    mass.setBins(nbins)
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass]) # associate numpy arr to RooRealVar
    # workspace.Import(mass)
    frame = mass.frame(Title=f"ZCR Dimuon Mass BWxDCB calibration fit for category {cat_idx}")

    # BWxDCB --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    # bwmZ.setConstant(True) # Stated in HIG-19-006
    bwWidth.setConstant(True) # Stated in HIG-19-006


    model1_1 = rt.RooBreitWigner("bwz", "BWZ",mass, bwmZ, bwWidth)

    """
    Note from Jan: sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can
    put whatevere restrictions we want.
    """
    mean = rt.RooRealVar("mean" , "mean", 0, -10,10) # mean is mean relative to BW
    # mean = rt.RooRealVar("mean" , "mean", 100, 95,110) # test
    sigma = rt.RooRealVar("sigma" , "sigma", 2, .1, 4.0)
    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
    n1 = rt.RooRealVar("n1" , "n1", 10, 0.01, 185)
    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
    n2 = rt.RooRealVar("n2" , "n2", 25, 0.01, 385)
    # n2 = rt.RooRealVar("n2" , "n2", 114, 0.01, 385) #test 114
    # n1.setConstant(True)
    # n2.setConstant(True)
    model1_2 = rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)

    # merge BW with DCB via convolution
    model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB


    mass.setBins(10000,"cache") # This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache",50.5)
    mass.setMax("cache",130.5)

    # # Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", 0.01, 0.00000001, 1)
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    #--------------------------------------------------

    # Landau Background --------------------------------------------------------------------------
    mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", 90, 70, 200)
    sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    model2 = rt.RooLandau("bkg", "bkg", mass, mean_landau, sigma_landau) # generate Landau bkg
    #-----------------------------------------------------

    # neg Exp Background --------------------------------------------------------------------------
    # coeff = rt.RooRealVar("coeff", "coeff", -0.01, -1,  -0.00000001)
    # shift = rt.RooRealVar("shift", "Offset", 70, 40, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    #--------------------------------------------------

    # # Reverse Landau Background test--------------------------------------------------------------------------
    # mean_landau = rt.RooRealVar("mean_landau" , "mean_landau", -80,  -150, -70) # 80
    # mass_neg = rt.RooFormulaVar("mass_neg", "-@0", [mass])
    # sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 7, 0.5, 8.5)
    # model2 = rt.RooLandau("bkg", "bkg", mass_neg, mean_landau, sigma_landau) # generate Landau bkg
    # #-----------------------------------------------------

    # Exp x Erf Background --------------------------------------------------------------------------
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "(@0 - @1)", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)

    # erf_center = rt.RooRealVar("erf_center" , "erf_center", 91.2, 75, 155)
    # erf_in = rt.RooFormulaVar("erf_in", "(@0 - @1)", rt.RooArgList(mass, erf_center))
    # model2_2a = rt.RooFit.bindFunction("erf", rt.TMath.Erf, erf_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erf","erf", model2_2a) # turn bound function to pdf
    # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate Expxerf bkg

    #-----------------------------------------------------

    # # Exp x Erf Background V2--------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want
    # shift = rt.RooRealVar("shift", "Offset", 100, 90, 150)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)
    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 100, 90, 150)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @1)", rt.RooArgList(mass, erfc_center))
    # # both bindPdf and RooGenericPdf work, but one may have better cuda integration over other, so leaving both options
    # # model2_2 = rt.RooFit.bindPdf("erfc", rt.TMath.Erfc, erfc_in)
    # model2_2 = rt.RooGenericPdf("erfc", "TMath::Erf(@0)+1", erfc_in)
    # model2 = rt.RooProdPdf("bkg", "bkg", rt.RooArgList(model2_1, model2_2))
    #-----------------------------------------------------



    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0.000001, 0.99999999)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])
    # final_model = model1_2



    time_step = time.time()

    #fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    _ = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    fit_result = final_model.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    logger.info(f"fitting elapsed time: {time.time() - time_step}")
    time.sleep(1) # rest a second for stability
    #do plotting
    roo_dataset.plotOn(frame, DataError="SumW2", Name="data_hist") # name is explicitly defined so chiSquare can find it
    # roo_hist.plotOn(frame, Name="data_hist") # name is explicitly defined so chiSquare can find it
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    model1.paramOn(frame, Parameters=[sigma], Layout=[0.55,0.94, 0.8])
    frame.GetYaxis().SetTitle("Events")
    frame.Draw()

    #calculate chi2 and add to plot
    n_free_params = fit_result.floatParsFinal().getSize()
    logger.info(f"n_free_params: {n_free_params}")
    chi2 = frame.chiSquare(final_model.GetName(), "data_hist", n_free_params)
    chi2 = float('%.3g' % chi2) # get upt to 3 sig fig
    logger.info(f"chi2: {chi2}")
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.7,0.8,f"#chi^2 = {chi2}")
    # canvas.Update()

    # obtain pull plot
    hpull = frame.pullHist("data_hist", "final_model")
    lower_pad.cd()
    frame2 = mass.frame(Title=" ")
    frame2.addPlotable(hpull, "P")
    frame2.GetYaxis().SetTitle("(Data-Fit)/ #sigma")
    frame2.GetYaxis().SetRangeUser(-5, 8)
    frame2.GetYaxis().SetTitleOffset(0.3)
    frame2.GetYaxis().SetTitleSize(0.08)
    frame2.GetYaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetLabelSize(0.08)
    frame2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    frame2.Draw()

    # canvas.Modified()
    canvas.Update()
    # canvas.Draw()


    # logger.info(f"mean_landau: {mean_landau.getVal()}")
    # logger.info(f"sigma_landau: {sigma_landau.getVal()}")
    logger.info(f"n1: {n1.getVal()}")
    logger.info(f"n2: {n2.getVal()}")
    logger.info(f"alpha1: {alpha1.getVal()}")
    logger.info(f"alpha2: {alpha2.getVal()}")
    logger.info(f"sigma result for cat {cat_idx}: {sigma.getVal()} +- {sigma.getError()}")

    # save cat_idx and sigma value to a pandas dataframe
    if not df_fit.empty:
        new_row = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])
        df_fit = pd.concat([df_fit, new_row], ignore_index=True)
    else:
        df_fit = pd.DataFrame([{"cat_name": cat_idx, "fit_val": sigma.getVal(), "fit_err": sigma.getError()}])


    # Save the cat_idx and sigma value to a log file
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write(f"{cat_idx} {sigma.getVal()} {sigma.getError()}\n")

    canvas.SaveAs(f"{output_dir}/calibration_fitCat{cat_idx}.pdf")
    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)
    return df_fit


def save_calibration_json(df_merged, json_filename="calibration_factors.json"):
    """
    Given a DataFrame (df_merged) with columns "cat_name" and "calibration_factor",
    write out a JSON file with the following multibinning structure.

    The calibration categories are assumed to be labeled as:
      "<pt_bin>_<eta1><eta2>"
    with:
      pt_bin in {"30-45", "45-52", "52-62", "62-200"}
      eta1, eta2 in {"B", "O", "E"}
    corresponding to the following bin edges:
      leading_mu_pt: [30.0, 45.0, 52.0, 62.0, 200.0]
      leading_mu_abseta: [0.0, 0.9, 1.8, 2.4]
      subleading_mu_abseta: [0.0, 0.9, 1.8, 2.4]

    The "content" field will be a flattened list of 36 calibration factors ordered as:
    For each pt_bin (in order: "30-45", "45-52", "52-62", "62-200"),
      for each leading muon eta bin (B, O, E),
      for each subleading muon eta bin (B, O, E),
      use the calibration factor from the corresponding category.
    If a category is missing, a default value of 1.0 is used.
    """
    # Define the bin edges and labels
    pt_bins = ["30-45", "45-52", "52-62", "62-200"]
    eta_bins = ["B", "O", "E"]

    calib_dict = dict(zip(df_merged["cat_name"], df_merged["calibration_factor"]))

    content = []
    # Loop over pt bins:
    for pt_bin in pt_bins:
        if pt_bin == "30-45_NOTUSED":
            # For pt bin "30-45", we have only three merged categories.
            # Loop over all 9 (leading, subleading) combinations but choose the factor based solely on subleading muon.
            for eta1 in eta_bins:
                for eta2 in eta_bins:
                    if eta2 == "B":
                        cat_name = "30-45_BB_OB_EB"
                    elif eta2 == "O":
                        cat_name = "30-45_BO_OO_EO"
                    elif eta2 == "E":
                        cat_name = "30-45_BE_OE_EE"
                    factor = calib_dict.get(cat_name, 1.0)
                    content.append(factor)
        else:
            # For other pt bins, there are 9 cells: loop over leading eta then subleading eta.
            for eta1 in eta_bins:
                for eta2 in eta_bins:
                    cat_name = f"{pt_bin}_{eta1}{eta2}"
                    content.append(calib_dict.get(cat_name, 1.0))

    # Build the JSON structure.
    json_dict = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "BS_ebe_mass_res_calibration",
                "description": "Dimuon Mass resolution calibration with BeamSpot Constraint correction applied",
                "version": 1,
                "inputs": [
                    {
                        "name": "leading_mu_pt",
                        "type": "real",
                        "description": "Transverse momentum of the leading muon (GeV)"
                    },
                    {
                        "name": "leading_mu_abseta",
                        "type": "real",
                        "description": "Absolute pseudorapidity of the leading muon"
                    },
                    {
                        "name": "subleading_mu_abseta",
                        "type": "real",
                        "description": "Absolute pseudorapidity of the subleading muon"
                    }
                ],
                "output": {
                    "name": "correction_factor",
                    "type": "real"
                },
                "data": {
                    "nodetype": "multibinning",
                    "inputs": [
                        "leading_mu_pt",
                        "leading_mu_abseta",
                        "subleading_mu_abseta"
                    ],
                    "edges": [
                        [30.0, 45.0, 52.0, 62.0, 200.0],
                        [0.0, 0.9, 1.8, 2.4],
                        [0.0, 0.9, 1.8, 2.4]
                    ],
                    "content": content,
                    "flow": "clamp"
                }
            }
        ]
    }

    with open(json_filename, "w") as f:
        json.dump(json_dict, f, indent=4)
    logger.info(f"Calibration JSON saved to {json_filename}")


def closure_test_from_df(df, additional_string, output_plot="closure_test_beforeCalibration.pdf"):
    """
    Given a DataFrame with columns:
         cat_name, fit_val, fit_err, median_val, calibration_factor,
    produce a closure test plot that compares the fitted resolution (fit_val)
    to the median predicted resolution (median_val) for each calibration category.

    A reference line y = x is drawn to indicate perfect agreement.

    Parameters:
      df         : Pandas DataFrame with the required columns.
      output_plot: Filename for the closure test plot.

    Returns:
      The input DataFrame (unchanged).
    """
    # Check that the necessary columns exist
    # required_cols = {"cat_name", "fit_val", "fit_err", "median_val", "calibration_factor"}
    required_cols = { "fit_val", "fit_err", "median_val"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Create the closure test plot.
    plt.figure(figsize=(8,6))
    plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"], fmt='o', label="Categories")

    # Plot the reference y = x line.
    x_min = df["median_val"].min()
    x_max = df["median_val"].max()
    x_vals = np.linspace(0.5, x_max*1.1, 100)
    plt.plot(x_vals, x_vals, "r--", label="y = x")

    # plot the 10% dotted line for reference
    y_10 = x_vals * 1.1
    plt.plot(x_vals, y_10, "g--", label="y = 1.1x")
    y_10 = x_vals * 0.9
    plt.plot(x_vals, y_10, "g--", label="y = 0.9x")



    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.title("Closure Test: Measured vs. Predicted Resolution")
    plt.legend()
    output_plot_dir = f"plots/{additional_string}/"
    os.makedirs(output_plot_dir, exist_ok=True)
    output_plot = os.path.join(output_plot_dir, output_plot)
    plt.savefig(output_plot)
    plt.close()

    logger.info(f"Closure test plot saved as {output_plot}")
    return df

def closure_test_from_df_BothBeforeAndAfter_OnSameCanvas(df, additional_string, output_plot="closure_test_combined.pdf"):
    """
    Generate a single closure test plot comparing:
      - fit_val vs median_val (unscaled prediction)
      - fit_val vs median_val * calibration_factor (scaled prediction)

    A y = x reference line and ±10% bands are also drawn.

    Parameters:
      df              : DataFrame with required columns.
      additional_string : Used for output directory naming.
      output_plot     : PDF filename.
    """
    required_cols = {"cat_name", "fit_val", "fit_err", "median_val", "calibration_factor"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    df["scaled_pred"] = df["median_val"] * df["calibration_factor"]

    x_min = min(df["median_val"].min(), df["scaled_pred"].min())
    x_max = max(df["median_val"].max(), df["scaled_pred"].max())
    x_vals = np.linspace(0.8 * x_min, 1.2 * x_max, 100)

    plt.figure(figsize=(8, 6))

    # Before calibration
    plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"],
                 fmt='o', label="Before Calibration", color='C0')

    # After calibration
    plt.errorbar(df["scaled_pred"], df["fit_val"], yerr=df["fit_err"],
                 fmt='s', label="After Calibration", color='C1')

    # Reference lines
    plt.plot(x_vals, x_vals, "k--", label="y = x")
    plt.plot(x_vals, 1.1 * x_vals, "g--", label="±10% band")
    plt.plot(x_vals, 0.9 * x_vals, "g--")

    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.title("Closure Test: Before and After Calibration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_plot_dir = f"plots/{additional_string}/"
    os.makedirs(output_plot_dir, exist_ok=True)
    full_path = os.path.join(output_plot_dir, output_plot)
    plt.savefig(full_path)
    plt.close()
    logger.info(f"Combined closure test plot saved as {full_path}")

    return df


def plot_closure_comparison_calibrated_uncalibrated(
    df,
    output_dir,
    output_plot="closure_test_comparison.pdf",
    pdfFile_ExtraText="",
    additional_string="",
):
    """
    Generate a closure test plot comparing:
      - fit_val vs median_val (after calibration), if available
      - fit_val vs median_val_NonCal (before calibration)

    A y = x reference line and ±10% bands are also drawn.

    Parameters:
      df                 : DataFrame with required columns.
      output_dir         : Output directory for saving the plot.
      output_plot        : PDF filename.
      pdfFile_ExtraText  : Extra string to append to output PDF filename.
    """
    required_cols = {"cat_name", "fit_val", "fit_err", "median_val_NonCal"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    has_median_val = "median_val" in df.columns

    # Determine x-axis limits
    x_min = df["median_val_NonCal"].min()
    x_max = df["median_val_NonCal"].max()
    if has_median_val:
        x_min = min(x_min, df["median_val"].min())
        x_max = max(x_max, df["median_val"].max())

    x_vals = np.linspace(0.8 * x_min, 1.2 * x_max, 100)

    plt.figure(figsize=(8, 6))

    # After calibration (only if available)
    if has_median_val:
        plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"],
                     fmt='o', label="After Calibration", color='C0')

    # Before calibration
    plt.errorbar(df["median_val_NonCal"], df["fit_val"], yerr=df["fit_err"],
                 fmt='s', label="Before Calibration", color='C1')

    # Reference lines
    plt.plot(x_vals, x_vals, "k--", label="y = x")
    plt.plot(x_vals, 1.1 * x_vals, "g--", label="±10% band")
    plt.plot(x_vals, 0.9 * x_vals, "g--")

    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]")
    plt.title("Closure Test: Before and After Calibration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_plot)
    if pdfFile_ExtraText:
        full_path = full_path.replace(".pdf", f"_{pdfFile_ExtraText}.pdf")
    plt.savefig(full_path)
    plt.close()
    logger.info(f"Combined closure test plot saved as {full_path}")


def closure_test_from_calibrated_df(df_fit, df_calibrated, additional_string, output_plot="closure_test.pdf"):
    df_merged = pd.merge(df_fit, df_calibrated, on="cat_name", how="inner")
    df = df_merged
    logger.info(df)
    required_cols = {"cat_name", "fit_val", "fit_err", "median_val"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Create the closure test plot.
    plt.figure(figsize=(8,6))
    plt.errorbar(df["median_val"], df["fit_val"], yerr=df["fit_err"], fmt='o', label="Categories")

    # Plot the reference y = x line.
    x_min = df["median_val"].min()
    x_max = df["median_val"].max()
    x_vals = np.linspace(0.5, x_max*1.1, 100)
    plt.plot(x_vals, x_vals, "r--", label="y = x")

    # plot the 10% dotted line for reference
    y_10 = x_vals * 1.1
    plt.plot(x_vals, y_10, "g--", label="y = 1.1x")
    y_10 = x_vals * 0.9
    plt.plot(x_vals, y_10, "g--", label="y = 0.9x")




    plt.xlabel("Predicted $\\sigma_{\\mu\\mu}$ [GeV]") # plt.xlabel("Median Predicted Resolution (GeV)")
    plt.ylabel("Measured $\\sigma_{\\mu\\mu}$ [GeV]") #plt.ylabel("Fitted Resolution (GeV)")
    plt.title("Closure Test: Measured vs. Predicted Resolution")
    plt.legend()
    output_plot = output_plot.replace(".pdf", f"_{additional_string}.pdf")
    # output_plot = f"plots/{additional_string}/" + output_plot
    plt.savefig(output_plot)
    # save image to png
    plt.savefig(output_plot.replace(".pdf", ".png"))
    plt.close()

    logger.info(f"Closure test plot saved as {output_plot}")
    # return df_merged
