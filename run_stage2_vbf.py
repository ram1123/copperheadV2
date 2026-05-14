import argparse
import copy
import glob
import itertools
import os
import pickle
import time
from functools import reduce
from pathlib import Path

import awkward as ak
import dask
import dask_awkward as dak
import hist.dask as hda
import numpy as np
import torch
from cli.common_argparser import build_common_parser
from coffea.ml_tools.torch_wrapper import torch_wrapper
from modules import selection
from modules.dask_utils import get_dask_client
from modules.utils import fillEventNans, logger
from modules.sample_config import get_bkg_sig_dicts
from tqdm import tqdm



def resolve_vbf_training_layout(model_path: Path, model_tag: str, nfolds: int):
    """
    Support both layouts:
    1) <model_path>/training_features.pkl + fold*/best_torchscript.pt
    2) <model_path>/training_features.pkl + <model_path>/<model_tag>/fold*/best_torchscript.pt
    3) <model_path>/<model_tag>/training_features.pkl + fold*/best_torchscript.pt
    """
    feature_roots = []
    for candidate in [model_path, model_path / model_tag]:
        if candidate not in feature_roots:
            feature_roots.append(candidate)

    model_roots = []
    for candidate in [model_path / model_tag, model_path]:
        if candidate not in model_roots:
            model_roots.append(candidate)

    missing_reports = []
    for feature_root in feature_roots:
        tf_path = feature_root / "training_features.pkl"
        scaler_paths = [feature_root / f"scalers_{fold}.npz" for fold in range(nfolds)]
        if not tf_path.is_file() or not all(p.is_file() for p in scaler_paths):
            continue

        for model_root in model_roots:
            model_paths = [
                model_root / f"fold{fold}" / "best_torchscript.pt"
                for fold in range(nfolds)
            ]
            if all(p.is_file() for p in model_paths):
                return feature_root, model_root
            missing_reports.extend([f"missing model: {p}" for p in model_paths if not p.is_file()])

        missing_reports.extend([f"missing training feature/scaler artifact under: {feature_root}"])

    details = "\n".join(missing_reports) if missing_reports else "No matching training layout found."
    raise FileNotFoundError(details)


def get_variation(wgt_variation, sys_variation):
    if "nominal" in wgt_variation:
        if "nominal" in sys_variation:
            return "nominal"
        else:
            return sys_variation
    else:
        if "nominal" in sys_variation:
            return wgt_variation
        else:
            return None


def get_compactedPath(stage1_path):
    """
    check if we have another directory, but with "compacted" in the name.
    if so, then return that instead
    NOTE: this is a lazy method that just looks if compacted directory exists.
    It doesn't check if all the necessary samples are in the directory.
    """
    compacted_stage1_path = str(stage1_path).replace("/f1_0", "/compacted")
    logger.debug(f"compacted_stage1_path: {compacted_stage1_path}")
    if os.path.isdir(compacted_stage1_path):
        return Path(compacted_stage1_path)
    elif os.path.isdir(stage1_path):
        return Path(stage1_path)
    else:
        logger.critical(f"Neither {compacted_stage1_path} nor {stage1_path} exists! Exiting!")
        raise FileNotFoundError(f"Neither {compacted_stage1_path} nor {stage1_path} exists! Exiting!")

def discover_jes_systs(fields, jet_prefixes=None):
    """
    Discover available JES/JER-like up/down suffixes from any jet-related variable.
    Returns a sorted list of strings like ['Absolute_2018_up', 'HF_down', ...].
    """
    if jet_prefixes is None:
        jet_prefixes = [
            "jet1_pt_",
            "jet2_pt_",
            "jj_mass_",
            "jj_dEta_",
            "njets_",
            "nBtagLoose_",
            "nBtagMedium_",
        ]
    suffixes = set()
    for f in fields:
        if not (f.endswith("_up") or f.endswith("_down")):
            continue
        for p in jet_prefixes:
            if f.startswith(p):
                suffixes.add(f[len(p):])
                break
    return sorted(suffixes)

def columns_for_selection(category, variation):
    # minimal columns for cuts; add here if your selection changes
    use_var = "nominal" if variation.startswith("wgt") else variation
    base = [
        "dimuon_mass",
        "event",
        f"njets_{use_var}",
        "gjj_mass",
        f"nBtagLoose_{use_var}",
        f"nBtagMedium_{use_var}",
        f"jj_mass_{use_var}",
        f"jj_dEta_{use_var}",
        f"jet1_pt_{use_var}",
        "nfatJets_drmuon",
        "MET_pt",
    ]
    return base

class DNNWrapper(torch_wrapper):
    def _create_model(self):
        logger.debug(f"Loading model from {self.torch_jit}")
        logger.debug(f"torch_jit type: {type(self.torch_jit)}")
        model = torch.jit.load(self.torch_jit)
        model.eval()
        return model
    def prepare_awkward(self, arr):
        # The input is any awkward array with matching dimension

        # Soln #1
        default_none_val = 0
        arr = ak.fill_none(arr, value=default_none_val) # apply "fill_none" to arr in order to remove "?" label of the awkward array


        # Soln #2
        # arr = ak.drop_none(arr)


        # Soln #3
        # arr = ak.to_packed(arr)

        # logger.info(f"arr: {arr.compute()}")
        return [
            ak.values_astype(arr, "float32"), #only modification we do is is force float32
        ], {}

def sigmoid_ak(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_ak(x, min_value, max_value):
    return ak.where(x < min_value, min_value, ak.where(x > max_value, max_value, x))


def prepare_features(events, features, variation="nominal"):
    features_var = []
    missing_features = []

    for feat in features:
        # Protect soft drop features (don't apply variation)
        variation_current = "nominal" if "soft" in feat else variation
        feat_name = None
        if f"{feat}_{variation_current}" in events.fields:
            feat_name = f"{feat}_{variation_current}"
        elif f"{feat}_nominal" in events.fields:
            feat_name = f"{feat}_nominal"
        elif feat in events.fields:
            feat_name = feat

        if feat_name in events.fields:
            features_var.append(feat_name)
        else:
            missing_features.append(feat_name)

    if missing_features:
        logger.warning(f"Missing features in events: {missing_features}")
        raise ValueError(f"Critical features missing: {missing_features}")

    if not features_var:
        logger.critical("No valid features found after filtering! Exiting.")
        raise ValueError("prepare_features: No features to use!")

    return features_var


def feature_name_for_variation(feat, variation, fields):
    # For weight-only variations, features must stay at nominal.
    # Also protect soft-drop features; fall back gracefully.
    if variation.startswith("wgt"):
        use_var = "nominal"
    else:
        use_var = "nominal" if "soft" in feat else variation
    candidates = [f"{feat}_{use_var}", f"{feat}_nominal", feat]
    for c in candidates:
        if c in fields:
            return c
    raise KeyError(f"Feature {feat} (var={variation}) not found in fields.")


def getFoldFilter(events, fold_vals, nfolds):
    fold_filter = ak.zeros_like(events.event, dtype="bool")
    # logger.info(f" eval_filter b4: {eval_filter.compute()}")
    for fold_value in fold_vals:
        fold_filter = fold_filter | ((events.event % nfolds) == fold_value)
    return fold_filter


def getStage1Samples(stage1_path, year, sample_config, data_samples=[], sig_samples=[], bkg_samples=[]):
    """
    sig samples: VBF, GGH
    bkg smaples: DY, TT, ST, VV, EWK
    """
    logger.info(f"stage1_path: {stage1_path}")
    data_l = []
    return_filelist_dict = {}
    for data_letter in data_samples:
        data_l.append(f"data_{data_letter.upper()}")

    data_filelist = []
    for sample in data_l:
        data_filelist += glob.glob(str(stage1_path / sample / "*" / "*.parquet"))

    if len(data_filelist) != 0:
        return_filelist_dict["data"] = data_filelist # keep data as one sample list for speedup

    # ------------------------------------
    # work on sig MC
    # ------------------------------------
    bkg_sample_dict, sig_sample_dict, combined_sample_dict = get_bkg_sig_dicts(
        yaml_path=sample_config,
        year=year,
    )

    sig_sample_l = []
    logger.info(f"sig_sample_dict: {sig_sample_dict}")
    logger.info(f"sig_samples: {sig_samples}")
    for sig_sample in sig_samples:
        logger.info(f"sig_sample: {sig_sample}")
        sig_sample = sig_sample.upper()
        if sig_sample in sig_sample_dict.keys():
            sig_sample_l += sig_sample_dict[sig_sample]
    logger.info(f"sig_sample_l: {sig_sample_l}")

    for sample in sig_sample_l:
        sample_filelist = glob.glob(str(stage1_path / sample / "*" / "*.parquet"))
        if len(sample_filelist) == 0:
            logger.warning(f"No {sample} files were found!")
            continue
        return_filelist_dict[sample] = sample_filelist

    # ------------------------------------
    # work on bkg MC
    # ------------------------------------
    bkg_sample_l = []
    for bkg_sample in bkg_samples:
        bkg_sample = bkg_sample.upper()
        if bkg_sample in bkg_sample_dict.keys():
            bkg_sample_l += bkg_sample_dict[bkg_sample]
    logger.info(f"bkg_sample_l: {bkg_sample_l}")

    for sample in bkg_sample_l:
        sample_filelist = glob.glob(str(stage1_path / sample / "*" / "*.parquet"))
        logger.info(f"sample: {sample}, number of files: {len(sample_filelist)}")
        logger.debug(f"sample_filelist: {sample_filelist}")
        if len(sample_filelist) == 0:
            logger.debug(f"No {sample} files were found!")
            continue
        return_filelist_dict[sample] = sample_filelist

    return return_filelist_dict


if __name__ == "__main__":
    t0 = time.perf_counter()
    parser = build_common_parser()
    parser.add_argument(
        "-m_tag",
        "--model_tag",
        dest="model_tag",
        default="test",
        action="store",
        help="Unique training label used to identify the model. Should match the label used in training.",
    )
    parser.add_argument(
        "-m_p",
        "--model_path",
        dest="model_path",
        default="test",
        action="store",
        help="path where model tag is saved on",
    )
    parser.add_argument(
        "-nv",
        "--no_variations",
        dest="no_variations",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, runs with all variations, otherwise only nominal",
    )
    parser.add_argument(
        "-nfolds",
        "--nfolds",
        dest="nfolds",
        default=4,
        type=int,
        action="store",
        help="Number of folds for cross-validation (default: 4)",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    t1 = time.perf_counter()
    logger.info(f"[timing] Argument parsing time: {t1 - t0:.2f} seconds")

    start_time = time.time()
    client = get_dask_client(args.use_gateway, cluster_index=args.cluster_index)

    t2 = time.perf_counter()
    logger.info(f"[timing] Dask client creation time: {t2 - t1:.2f} seconds")

    base_path = Path(args.input_path)

    bkg_samples = args.bkg_samples
    sig_samples = args.sig_samples
    data_samples = args.data_samples
    logger.info(f"data_samples: {data_samples}")

    stage1_path = base_path / "stage1_output" / args.year / "f1_0"
    stage1_path = get_compactedPath(stage1_path) # get compacted stage1 output if they exist
    logger.info(f"stage1 path: {stage1_path}")
    if not os.path.exists(stage1_path):
        logger.critical(f"Stage1 path {stage1_path} does not exist! Exiting!")
        raise FileNotFoundError(f"Stage1 path {stage1_path} does not exist! Run the compaction script first.")

    histDirName = f"score_{args.label}" if args.save_postfix == "" else f"score_{args.label}_{args.save_postfix}"
    if args.no_variations:
        histDirName = f"{histDirName}_NoSyst"

    hist_save_path = base_path / "stage2_histograms" / histDirName / args.year
    os.makedirs(hist_save_path, exist_ok=True)
    logger.info(f"Histograms will be saved to: {hist_save_path}")

    full_sample_dict = getStage1Samples(stage1_path, args.year, args.sample_config, data_samples=data_samples, sig_samples=sig_samples, bkg_samples=bkg_samples)

    logger.debug(f"full_sample_dict: {full_sample_dict}")
    logger.info(f"full_sample_dict: {full_sample_dict.keys()}")
    t3 = time.perf_counter()
    logger.info(f"[timing] sample dict processing time: {t3 - t2:.2f} seconds")


    if "dy_VBF_filter" in full_sample_dict.keys(): # if dy_VBF_filter is not available, don't apply the VBF filter cut on other DY samples
        do_vbf_filter_study = True
    else:
        do_vbf_filter_study = False
    
    nfolds = args.nfolds  # Define nfolds once for all samples
    for sample_type, sample_l in tqdm(full_sample_dict.items(), desc="Processing Samples"):
        t4 = time.perf_counter()

        # if output pkl file already exists, skip
        output_pkl_path = hist_save_path / f"{sample_type}_hist.pkl"
        if output_pkl_path.exists():
            print(f"Output pkl file {output_pkl_path} already exists. Skipping {sample_type}.")
            logger.warning(f"Output pkl file {output_pkl_path} already exists. Skipping {sample_type}.")
            continue

        logger.info(f"Processing sample type: {sample_type}, number of files: {len(sample_l)}")
        logger.info(f"Sample type: {sample_type}")
        logger.debug(f"Sample list: {sample_l}")
        if len(sample_l) == 0:
            logger.critical(f"No files for {sample_type} is found! Skipping!")
            continue

        events_schema = dak.from_parquet(sample_l)
        fields = set(events_schema.fields)
        logger.debug(f"fields: {fields}")

        # Auto-discover JES/JER-like systematic suffixes from jet-related columns
        jes_systs = discover_jes_systs(fields)
        # if "log_" exists remove that element from jes_systs, as these are not variations. This log belongs to the log of a particular variable.
        jes_systs = [sys for sys in jes_systs if not sys.startswith("log_")]
        logger.info(f"Discovered JES/JER variations: {jes_systs}")

        model_trained_path = Path(args.model_path).resolve()
        training_dir = (model_trained_path / args.model_tag).resolve()

        missing = []
        for fold in range(nfolds):
            p = training_dir / f"fold{fold}" / "best_torchscript.pt"
            s = model_trained_path / f"scalers_{fold}.npz"
            if not p.is_file():
                missing.append(f"missing model: {p}")
            if not s.is_file():
                missing.append(f"missing scaler: {s}")

        if missing:
            raise FileNotFoundError("\n".join(missing))

        # Load training features once per sample_type
        with open(model_trained_path / "training_features.pkl", "rb") as f:
            training_features = pickle.load(f)
        logger.debug(f"training_features: {training_features}")
        logger.info(f"len training_features: {len(training_features)}")

        NO_SCALE_FEATURES = {
            "year",
            "nsoftjets5_nominal",
        }

        # cache scalers
        scaler_cache = {}
        for fold in range(nfolds):
            sp = model_trained_path / f"scalers_{fold}.npz"
            with np.load(sp, allow_pickle=True) as f:
                scaler_cache[fold] = {
                    "mean": f["mean"].astype(np.float64),
                    "std":  f["std"].astype(np.float64),
                    "features": [str(x) for x in f["features"]],
                }

        # cache models (wrapper caches torchscript model lazily inside worker)
        model_cache = {}
        for fold in range(nfolds):
            mp = training_dir / f"fold{fold}" / "best_torchscript.pt"
            model_cache[fold] = DNNWrapper(mp.as_posix())
        # ------------------------------------------
        # Initialize sample histogram to save later
        # ------------------------------------------
        # logger.debug(f"fields: {events_stage1.fields}")
        if "data" in sample_type:
            wgt_variations = ["wgt_nominal"]
        else:
            # Collect nominal + _up/_down weight variations (exclude any 'separate' helpers)
            wgt_variations = ["wgt_nominal"] + sorted([
                w for w in fields
                if w.startswith("wgt_") and (w.endswith("_up") or w.endswith("_down")) and ("separate" not in w)
            ])

            #     wgt_variations = ["wgt_nominal",
            #                       'wgt_muIso_up', 'wgt_pu_wgt_up', 'wgt_muID_up', 'wgt_muTrig_up',
            #                       'wgt_muIso_down', 'wgt_pu_wgt_down', 'wgt_muID_down', 'wgt_muTrig_down',
            #                       'jet1_pt_jer6_up', 'jet1_pt_BBEC1_2018_up', 'jet1_pt_HF_up',
            #                       'jet1_pt_jer6_down', 'jet1_pt_BBEC1_2018_down', 'jet1_pt_HF_down',

            #                       ]  # FIXME: For debugging purpose.
            # wgt_variations = ["wgt_nominal", "wgt_muIso_up", "wgt_muIso_down"]
            logger.debug(f"wgt_variations: {wgt_variations}")
            logger.debug(f"length of wgt_variations: {len(wgt_variations)}")

            if args.no_variations:
                logger.warning(f"No weight variations found for {sample_type}, using nominal only.")
                wgt_variations = ["wgt_nominal"]

        t5 = time.perf_counter()
        logger.info(f"[timing] Weight variation processing time: {t5 - t4:.2f} seconds")
        logger.info(f"wgt_variations: {wgt_variations}")
        if args.no_variations:
            syst_variations = ["nominal"]
        else:
            syst_variations = ["nominal"] + jes_systs
        logger.info(f"syst_variations: {syst_variations}")
        variations = []
        for w in wgt_variations:
            for v in syst_variations:
                variation = get_variation(w, v)
                if variation:
                    variations.append(variation)
        logger.info(f"variations: {variations}")

        regions = ["h-peak", "h-sidebands"]  # full list of possible regions to loop over
        channels = ["vbf"]  # full list of possible channels to loop over
        score_hist = (
            hda.Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
        )
        # add axis for systematic variation
        score_hist = score_hist.StrCat(variations, name="variation")
        # add score category
        bins = selection.binning  # use the binning from selection module
        logger.info(f"bins: {bins}")
        score_name = f"score_{args.label}"
        score_hist = score_hist.Var(bins, name=score_name)

        score_hist_empty = score_hist.Double()

        # loop over configurations and fill the histogram
        loop_args_dict = {
            "region": regions,
            "wgt_variation": wgt_variations,
            "syst_variation": syst_variations,
            "channel": channels,
        }
        loop_args = [
            dict(zip(loop_args_dict.keys(), values))
            for values in itertools.product(*loop_args_dict.values())
        ]
        logger.debug(f"loop_args: {loop_args}")

        t6 = time.perf_counter()
        logger.info(f"[timing] Empty histogram time: {t6 - t5:.2f} seconds")

        score_hist_l = []
        iteration_counter = 0
        for count, loop_arg in enumerate(loop_args):
            score_hist = copy.deepcopy(score_hist_empty)

            region              = loop_arg["region"]
            category          = loop_arg["channel"]
            syst_variation = loop_arg["syst_variation"]
            wgt_variation  = loop_arg["wgt_variation"]
            variation          = get_variation(wgt_variation, syst_variation)
            logger.debug(f"variation: {variation}")
            if not variation:
                logger.debug(f"skipping variation {variation} from {wgt_variation} and {syst_variation}")
                continue

            sel_cols = columns_for_selection(category, variation)
            needed_cols = set(sel_cols + [wgt_variation])

            logger.debug(f"sel_cols: {sel_cols}")
            logger.debug(f"len(sel_cols): {len(sel_cols)}")

            # Never decorate feature columns with weight-only variation suffixes
            variation_for_features = "nominal" if variation.startswith("wgt") else variation

            # Map base training feature name -> actual source column for this variation
            feature_sources = {
                f: feature_name_for_variation(f, variation_for_features, fields)
                for f in training_features
            }
            logger.debug(f"feature_sources: {feature_sources}")
            for src_name in feature_sources.values():
                needed_cols.add(src_name)
            needed_cols = sorted(needed_cols)

            logger.debug(f"needed_cols: {needed_cols}")
            logger.debug(f"len(needed_cols): {len(needed_cols)}")

            # Re-open with column projection + row-group split
            events_stage1 = dak.from_parquet(
                sample_l,
                columns=needed_cols,
                # split_row_groups=True, # FIXME: This introduces some issue and number of entries does not remain same.
                # ).persist()  # Persist to memory for faster access
            )           
            logger.info(f"do_vbf_filter_study: {do_vbf_filter_study}")
            events = selection.applyRegionCatCuts(
                events_stage1,
                process=sample_type,
                category=category,
                region_name=region,
                do_vbf_filter_study=do_vbf_filter_study,
                variation=variation,
            )
            events = fillEventNans(events, category=category) # for vbf category, this may be unnecessary
            # As DNN is trained in the h-peak region, so while evaluating for the h-sideband region
            # we fix the dimuon mass to 125.0 GeV
            if region == "h-sidebands":
                try:
                    events["dimuon_mass"] = 125.0 * ak.ones_like(events.dimuon_mass)
                    logger.debug("[sidebands] Forced dimuon_mass=125.0 for DNN inputs")
                except Exception as _e:
                    logger.warning(f"[sidebands] Failed to fix dimuon_mass to 125.0: {_e}")

            nan_val = -99.0
            input_arr_dict = {
                feat: nan_val * ak.ones_like(events.event) for feat in training_features
            }
            logger.debug(f" input_arr_dict b4: {input_arr_dict}")
            for fold in range(nfolds):
                eval_folds = [(fold + f) % nfolds for f in [3]]
                logger.debug(f" eval_folds: {eval_folds}")
                eval_filter = getFoldFilter(events, eval_folds, nfolds)

                scaler_info = scaler_cache[fold]
                scalar_means = scaler_info["mean"].astype(np.float64)
                scalar_stds = scaler_info["std"].astype(np.float64)
                scaler_features = scaler_info["features"]

                scaler_map = {feat: (scalar_means[i], scalar_stds[i]) for i, feat in enumerate(scaler_features)}

                missing_in_scaler = [f for f in training_features if (f not in scaler_map and f not in NO_SCALE_FEATURES)]
                if missing_in_scaler:
                    raise ValueError(f"Features {missing_in_scaler} are missing in scaler and not in NO_SCALE_FEATURES!")

                for ix, base_feat in enumerate(training_features):
                    src_feat = feature_sources[base_feat]  # e.g. 'jet1_eta_jer6_down' for JES/JER
                    logger.debug(f"Processing feature: {base_feat}, source feature: {src_feat}")
                    input_arr_fold = input_arr_dict[base_feat]

                    # scale from the *source* column, but keep the base key + index
                    in_feat = events[src_feat]

                    # Features not present in scaler_features are passthrough by construction
                    # (in your case: nsoftjets5_nominal and year)
                    if base_feat in scaler_map and base_feat not in NO_SCALE_FEATURES:
                        mu, sigma = scaler_map[base_feat]
                        in_feat = (in_feat - mu) / sigma
                    else:
                        logger.debug(f"Not scaling feature {base_feat} (either NO_SCALE or not in scaler_map)")
                        in_feat = ak.values_astype(in_feat, "float32")
                    input_arr_fold = ak.where(eval_filter, in_feat, input_arr_fold)
                    input_arr_dict[base_feat] = input_arr_fold

            # ---------------------------------------------------
            # Now evaluate DNN score
            # ---------------------------------------------------
            input_arr = ak.concatenate(
                [
                    input_arr_dict[feat][:, np.newaxis] for feat in training_features
                ],  # np.newaxis is added so that we can concat on axis=1
                axis=1,
            )
            dnn_logit = nan_val * ak.ones_like(events.event)

            for fold in range(nfolds):
                eval_folds = [(fold + f) % nfolds for f in [3]]
                eval_filter = getFoldFilter(events, eval_folds, nfolds)
                dnnWrap = model_cache[fold]
                dnn_score_fold = dnnWrap(input_arr)
                dnn_score_fold = ak.flatten(dnn_score_fold, axis=None)

                dnn_logit = ak.where(eval_filter, dnn_score_fold, dnn_logit)

            valid_mask = dnn_logit != nan_val
            dnn_score = sigmoid_ak(dnn_logit)
            dnn_score = ak.where(valid_mask, dnn_score, nan_val)
            dnn_score = np.arctanh(clip_ak(dnn_score, 0.0, 0.999999))
            dnn_score = ak.where(valid_mask, dnn_score, nan_val)

            # logger.info(f"dnn_score: {dnn_score.compute()}")
            # logger.info(f"dnn_score: {ak.max(dnn_score).compute()}")
            # logger.info(f"dnn_score: {ak.min(dnn_score).compute()}")
            # raise ValueError
            # ---------------------------------------------------
            # Now onto converting DNN score as histograms
            # ---------------------------------------------------
            to_fill = {
                "region": region,
                "channel": "vbf",
                "variation": variation,
                score_name: dnn_score
            }
            weight = events[wgt_variation]

            to_fill_value = to_fill.copy()
            to_fill_value["val_sumw2"] = "value"
            score_hist.fill(**to_fill_value, weight=weight)

            to_fill_sumw2 = to_fill.copy()
            to_fill_sumw2["val_sumw2"] = "sumw2"
            score_hist.fill(**to_fill_sumw2, weight=weight * weight)

            logger.info(f"[{count:>4}]:score_hist is filled for {sample_type}, {region:>11}, {variation:<35} variation!")
            score_hist_l.append(score_hist)

        t7 = time.perf_counter()
        logger.info(f"[timing] Variation loop time: {t7 - t6:.2f} seconds")
        # ---------------------------------------------------
        # done with variation loop, compute hist
        # ---------------------------------------------------
        logger.info(f"loop_args len: {len(loop_args)}")
        logger.info(f"score_hist_l len: {len(score_hist_l)}")
        score_hist_l = dask.compute(score_hist_l)[0]
        logger.info(f"score_hist_l len after compute: {len(score_hist_l)}")
        if not score_hist_l:
            logger.warning(f"No histograms were filled for {sample_type}, skipping saving.")
            continue  # skip to next sample_type
        # Merge histograms from all loop configurations
        score_hist = reduce(lambda a, b: a + b, score_hist_l)
        logger.info("compute done!")
        t8 = time.perf_counter()
        logger.info(f"[timing] Dask compute time: {t8 - t7:.2f} seconds")

        # ---------------------------------------------------
        # Save Hist
        # ---------------------------------------------------
        with open(f"{output_pkl_path}", "wb") as file:
            pickle.dump(score_hist, file)
            logger.info(f"{sample_type} histogram on {output_pkl_path}")

        t9 = time.perf_counter()
        logger.info(f"[timing] Histogram saving time: {t9 - t8:.2f} seconds")

    logger.info("Success!")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time : {execution_time:.4f} seconds")
