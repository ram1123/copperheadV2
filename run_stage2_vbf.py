import argparse
import glob
import itertools
import os
import pickle
import time
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import torch
from cli.common_argparser import build_common_parser
from coffea import processor
from coffea.ml_tools.torch_wrapper import torch_wrapper
from coffea.nanoevents import BaseSchema
from modules import selection
from modules.dask_utils import get_dask_client
from modules.utils import get_compacted_path, logger, fillEventNans
from modules.sample_config import get_bkg_sig_dicts


DATASET_SEPARATOR = "::"


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


def discover_shape_systs(fields, prefixes=None):
    """
    Discover available shape-variation suffixes from shifted stage1 branches.
    This covers both jet/JEC-like variations and muon-momentum shape variations.
    """
    if prefixes is None:
        prefixes = [
            "dimuon_mass_",
            "dimuon_pt_",
            "dimuon_pt_log_",
            "dimuon_eta_",
            "dimuon_rapidity_",
            "dimuon_ebe_mass_res_",
            "dimuon_ebe_mass_res_rel_",
            "mu1_pt_",
            "mu2_pt_",
            "mu1_pt_over_mass_",
            "mu2_pt_over_mass_",
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
        # Use the longest (most specific) matching prefix, not the first one in
        # list order, so e.g. "mu1_pt_over_mass_" wins over "mu1_pt_" for a field
        # like "mu1_pt_over_mass_mu_roccor_up" regardless of how `prefixes` is
        # ordered.
        matching_prefixes = [p for p in prefixes if f.startswith(p)]
        if not matching_prefixes:
            continue
        best_prefix = max(matching_prefixes, key=len)
        suffixes.add(f[len(best_prefix):])
    return sorted(suffixes)

SHIFTED_SELECTION_VARIABLES = {
    "njets",
    "nBtagLoose",
    "nBtagMedium",
    "jj_mass",
    "jj_dEta",
    "jet1_pt",
}

NOMINAL_SELECTION_VARIABLES = {
    "dimuon_mass",
    "event",
    "gjj_mass",
    "nfatJets_drmuon",
    "MET_pt",
}


def resolve_variation_field(base, variation, fields):
    if base in NOMINAL_SELECTION_VARIABLES:
        if base in fields:
            logger.debug(
            # logger.warning(
                f"[stage2][field-resolve] kind=selection variation={variation} "
                f"var={base} resolved={base} fallback=False"
            )
            return base
        raise KeyError(f"Missing nominal selection field '{base}'")

    if base in SHIFTED_SELECTION_VARIABLES:
        use_var = "nominal" if variation == "nominal" or variation.startswith("wgt") else variation
        candidates = [f"{base}_{use_var}"]
        if use_var != "nominal":
            candidates.append(f"{base}_nominal")
        candidates.append(base)

        for idx, candidate in enumerate(candidates):
            if candidate in fields:
                if idx > 0:
                    logger.warning(
                        f"[stage2] Selection field '{base}_{use_var}' unavailable for "
                        f"variation '{variation}'; falling back to '{candidate}'."
                    )
                logger.debug(
                # logger.warning(
                    f"[stage2][field-resolve] kind=selection variation={variation} "
                    f"var={base} resolved={candidate} fallback={idx > 0}"
                )
                return candidate

        raise KeyError(
            f"Selection field '{base}_{use_var}' for variation '{variation}' is "
            f"unavailable (tried: {candidates})."
        )

    raise KeyError(f"Unknown selection variable '{base}'")


def columns_for_selection(category, variation, fields):
    # minimal columns for cuts; add here if your selection changes.
    # NOMINAL_SELECTION_VARIABLES are only used behind switch-gated cuts (e.g.
    # do_VH_veto) downstream, so they're kept optional here and left for the
    # caller's `needed_cols & set(events.fields)` intersection to drop if absent,
    # instead of being required via resolve_variation_field().
    cols = [
        "event",
        "dimuon_mass",
        "gjj_mass",
        "nfatJets_drmuon",
        "MET_pt",
    ]
    cols += [
        resolve_variation_field(name, variation, fields)
        for name in sorted(SHIFTED_SELECTION_VARIABLES)
    ]
    return cols


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


def feature_name_for_variation(
    feat,
    variation,
    fields,
    allow_nominal_fallback=False,
    nominal_only_features=None,
):
    """Resolve DNN inputs; default caller passes nominal to match sideHustle5."""
    # training_features often already carry a literal "_nominal" suffix
    # (e.g. "jet1_phi_nominal") rather than a bare base name ("jet1_phi").
    # Strip it so the shifted candidate below is built from the base name,
    # not double-suffixed into something like "jet1_phi_nominal_nominal".
    base = feat[: -len("_nominal")] if feat.endswith("_nominal") else feat

    if variation == "nominal" or variation.startswith("wgt"):
        use_var = "nominal"
    elif "soft" in feat:
        use_var = "nominal"
    else:
        use_var = variation

    candidates = [f"{base}_{use_var}", f"{base}_nominal", feat]
    for idx, c in enumerate(candidates):
        if c in fields:
            if idx > 0:
                logger.warning(
                    f"[stage2] DNN feature '{base}_{use_var}' unavailable for "
                    f"variation '{variation}'; falling back to '{c}'."
                )
            logger.debug(
            # logger.warning(
                f"[stage2][field-resolve] kind=dnn variation={variation} "
                f"var={feat} resolved={c} fallback={idx > 0}"
            )
            return c
    if allow_nominal_fallback:
        logger.warning(
            f"[stage2] DNN feature '{base}_{use_var}' (and nominal/base) unavailable "
            f"for variation '{variation}'; falling back to unverified '{feat}'."
        )
        logger.debug(
        # logger.warning(
            f"[stage2][field-resolve] kind=dnn variation={variation} "
            f"var={feat} resolved={feat} fallback=True"
        )
        return feat
    raise KeyError(
        f"Feature {feat} (var={variation}) not found in fields. "
        f"Tried: {candidates}. "
        "DNN inputs are resolved to nominal fields for all variations."
    )


def getFoldFilter(events, fold_vals, nfolds):
    fold_filter = ak.zeros_like(events.event, dtype="bool")
    # logger.info(f" eval_filter b4: {eval_filter.compute()}")
    for fold_value in fold_vals:
        fold_filter = fold_filter | ((events.event % nfolds) == fold_value)
    return fold_filter


def dy_vbf_filter_is_available(stage1_path, year, sample_config):
    """Return True when DY VBF-filter samples exist in config and on disk."""
    direct_files = glob.glob(str(stage1_path / "dy_VBF_filter" / "*" / "*.parquet"))
    if direct_files:
        logger.info(
            "Detected dy_VBF_filter with %d parquet files.",
            len(direct_files),
        )
        return True

    bkg_sample_dict, _, _ = get_bkg_sig_dicts(
        yaml_path=sample_config,
        year=year,
    )
    for sample in bkg_sample_dict.get("DYVBF", []):
        files = glob.glob(str(stage1_path / sample / "*" / "*.parquet"))
        if files:
            logger.info(
                "Detected DY VBF-filter sample %s with %d parquet files.",
                sample,
                len(files),
            )
            return True
    return False


def parse_years_option(year, years):
    """Return ordered years from -y/--year or --years."""
    raw_years = years if years != ["2018"] else [year]
    parsed_years = []
    for raw_year in raw_years:
        for candidate in str(raw_year).split(","):
            candidate = candidate.strip()
            if candidate and candidate not in parsed_years:
                parsed_years.append(candidate)
    if not parsed_years:
        raise ValueError("At least one year must be provided.")
    return tuple(parsed_years)


_MODEL_CACHE = {}


def load_torchscript_model(model_path):
    """Load each TorchScript model once per worker process."""
    if model_path not in _MODEL_CACHE:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        _MODEL_CACHE[model_path] = model
    return _MODEL_CACHE[model_path]


class CoffeaStage2VBFProcessor(processor.ProcessorABC):
    """Evaluate the trained VBF DNN and fill Stage-2-shaped histograms."""

    def __init__(
        self,
        training_features,
        scalers,
        model_paths,
        nfolds,
        score_name,
        no_variations=False,
        do_vbf_filter_study=False,
        allow_nominal_feature_fallback=True,
        use_nominal_dnn_features_for_systs=False,
    ):
        NO_SCALE_FEATURES = {
            "year",
            "nsoftjets5_nominal",
        }
        self.training_features = training_features
        self.scalers = scalers
        self.model_paths = model_paths
        self.nfolds = nfolds
        self.score_name = score_name
        self.no_variations = no_variations
        self.do_vbf_filter_study = do_vbf_filter_study
        self.allow_nominal_feature_fallback = allow_nominal_feature_fallback
        self.use_nominal_dnn_features_for_systs = use_nominal_dnn_features_for_systs
        self.no_scale_features = NO_SCALE_FEATURES

    def evaluate_scores(self, events, variation):
        fields = set(events.fields)
        event_numbers = ak.to_numpy(ak.materialize(events.event))
        n_events = len(events)
        columns = {}
        for feature in self.training_features:
            source = feature_name_for_variation(
                feature,
                variation,
                fields,
                allow_nominal_fallback=self.allow_nominal_feature_fallback,
                nominal_only_features=self.no_scale_features,
            )
            logger.debug(
                "DNN feature mapping: variation=%s, feature=%s -> source=%s",
                variation,
                feature,
                source,
            )
            values = ak.materialize(events[source])
            fill_value = -10.0 if "phi" in feature else 0.0
            values = ak.fill_none(values, fill_value)
            columns[feature] = ak.to_numpy(values)

        nan_val = -99.0
        input_columns = {
            feature: np.full(n_events, nan_val, dtype=np.float64)
            for feature in self.training_features
        }
        for fold in range(self.nfolds):
            fold_mask = (event_numbers % self.nfolds) == ((fold + 3) % self.nfolds)
            scaler = self.scalers[fold]
            missing_in_scaler = [
                feature
                for feature in self.training_features
                if feature not in scaler and feature not in self.no_scale_features
            ]
            if missing_in_scaler:
                raise ValueError(
                    "Features "
                    f"{missing_in_scaler} are missing in scaler and not in "
                    "NO_SCALE_FEATURES!"
                )

            for feature in self.training_features:
                values = columns[feature]
                if feature in scaler and feature not in self.no_scale_features:
                    mean, std = scaler[feature]
                    values = (values - mean) / std
                input_columns[feature][fold_mask] = values[fold_mask]

        inputs = np.column_stack(
            [input_columns[feature] for feature in self.training_features]
        ).astype(np.float32)

        logits = np.full(n_events, nan_val, dtype=np.float32)
        torch.set_num_threads(1)
        for fold in range(self.nfolds):
            fold_mask = (event_numbers % self.nfolds) == ((fold + 3) % self.nfolds)
            if not np.any(fold_mask):
                continue

            model = load_torchscript_model(self.model_paths[fold])
            with torch.inference_mode():
                fold_logits = model(torch.from_numpy(inputs)).reshape(-1).numpy()
            logits[fold_mask] = fold_logits[fold_mask]

            # has_non_finite = not np.isfinite(fold_logits).all()
            # if has_non_finite:
            #     nan_cols = np.isnan(inputs[~np.isfinite(fold_logits)]).any(axis=0)
            #     # raise ValueError("Found non-finite values in fold_logits. This may indicate a problem with the model or input data.")
            #     # raise ValueError(f"Found non-finite values in fold_logits for fold {fold}. with value {fold_logits[~np.isfinite(fold_logits)]} and input {inputs[~np.isfinite(fold_logits)]}.")
            #     # raise ValueError(f"Found non-finite values in fold_logits for fold {fold}. with value {fold_logits[~np.isfinite(fold_logits)]} and input {nan_cols.shape}, {self.training_features}.")
            #     from itertools import compress
            #     # raise ValueError(f"Found non-finite values in fold_logits for fold {fold}. with value {fold_logits[~np.isfinite(fold_logits)]} and input {list(compress(self.training_features, nan_cols))}.")
            #     jj_dEta_nan_arr = input_columns["jj_dEta_nominal"][~np.isfinite(fold_logits)]
            #     jj_dEta_variation_nan_arr = events[f"jj_dEta_{variation}"][~np.isfinite(fold_logits)]
            #     raise ValueError(f"Found non-finite values in fold_logits for fold {fold}. with value {fold_logits[~np.isfinite(fold_logits)]} and jj_dEta nominal {jj_dEta_nan_arr} and jj_dEta {variation} {jj_dEta_variation_nan_arr}.")


        # probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        has_non_finite = not np.isfinite(logits).all()
        # if has_non_finite:
        #     raise ValueError("Found non-finite values in logits. This may indicate a problem with the model or input data.")
        finite_mask = np.isfinite(logits)
        logits = np.where(finite_mask, logits, nan_val)
        dnn_score = sigmoid_ak(logits)
        # valid_mask = logits != nan_val
        # dnn_score = np.where(valid_mask, dnn_score, nan_val)
        # dnn_score = np.arctanh(clip_ak(dnn_score, 0.0, 0.999999))
        # dnn_score = np.where(valid_mask, dnn_score, nan_val)
        # # return np.arctanh(np.clip(probabilities, 0.0, 0.999999))
        # return dnn_score
        return np.arctanh(clip_ak(dnn_score, 0.0, 0.999999))


    def process(self, events):
        dataset_key = events.metadata["dataset"]
        sample_type = events.metadata.get("sample", dataset_key)
        year = events.metadata["year"]
        # events["MET_pt"] = events["PuppiMET_pt"]
        fields = set(events.fields)

        if "data" in sample_type:
            wgt_variations = ["wgt_nominal"]
        else:
            wgt_variations = ["wgt_nominal"] + sorted(
                w
                for w in fields
                if w.startswith("wgt_")
                and (w.endswith("_up") or w.endswith("_down"))
                and ("separate" not in w)
            )
            if self.no_variations:
                wgt_variations = ["wgt_nominal"]

        syst_variations = ["nominal"]
        if not self.no_variations:
            syst_variations += [
                syst
                for syst in discover_shape_systs(fields)
                if not syst.startswith("log_")
            ]

        variations = []
        for weight_variation, syst_variation in itertools.product(
            wgt_variations,
            syst_variations,
        ):
            variation = get_variation(weight_variation, syst_variation)
            if variation:
                variations.append(variation)

        score_hist = (
            hist.Hist.new.StrCat(["h-peak", "h-sidebands"], name="region")
            .StrCat(["vbf"], name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .StrCat(variations, name="variation", growth=True)
            .Var(selection.binning, name=self.score_name)
            .Double()
        )

        selected_events = 0
        for region, weight_variation, syst_variation in itertools.product(
            ["h-peak", "h-sidebands"],
            wgt_variations,
            syst_variations,
        ):
            variation = get_variation(weight_variation, syst_variation)
            if not variation:
                continue
            category= "vbf"
            sel_cols = columns_for_selection(category, variation, events.fields)
            needed_cols = set(sel_cols + [weight_variation])

            # ----------------------------------
            for feature in self.training_features:
                source = feature_name_for_variation(
                    feature,
                    variation,
                    events.fields,
                    allow_nominal_fallback=self.allow_nominal_feature_fallback,
                    nominal_only_features=self.no_scale_features,
                )
                needed_cols.add(source)

            needed_cols4print = sorted(needed_cols)
            needed_cols = needed_cols & set(events.fields) # merge with existing fields
            needed_cols = sorted(needed_cols)
            # ----------------------------------
            debug_variations = {
                "Absolute_2017_up",
                "Absolute_up",
                "BBEC1_2017_up",
                "BBEC1_up",
                "EC2_2017_up",
                "EC2_up",
                "FlavorQCD_up",
                "HF_up",
                "RelativeBal_down",
                "RelativeBal_up",
                "RelativeSample_2017_down",
                "RelativeSample_2017_up",
                "Total_up",
            }
            # if variation in debug_variations:
            feature_variation = (
                "nominal" if self.use_nominal_dnn_features_for_systs else variation
            )
            filtered_events = events[needed_cols]
            # if variation in debug_variations:
            #     raise ValueError(f"Needed columns for {variation}: {needed_cols4print} \n fields for {variation}: {filtered_events.fields}")
            region_events = selection.applyRegionCatCuts(
                filtered_events,
                process=sample_type,
                category=category,
                region_name=region,
                do_vbf_filter_study=self.do_vbf_filter_study,
                variation=variation,
                # year=year,
            )
            region_events = fillEventNans(region_events, category=category)
            if region == "h-sidebands":
                region_events = ak.with_field(
                    region_events,
                    125.0 * ak.ones_like(region_events.dimuon_mass),
                    "dimuon_mass",
                )
            if variation == "nominal":
                selected_events += len(region_events)

            scores = self.evaluate_scores(region_events, feature_variation)
            weights = ak.to_numpy(ak.materialize(region_events[weight_variation]))
            fill_common = {
                "region": region,
                "channel": "vbf",
                "variation": variation,
                self.score_name: scores,
            }
            score_hist.fill(**fill_common, val_sumw2="value", weight=weights)
            score_hist.fill(
                **fill_common,
                val_sumw2="sumw2",
                weight=weights * weights,
            )

        return {
            dataset_key: {
                "events": processor.value_accumulator(int, selected_events),
                "chunks": processor.value_accumulator(int, 1),
                "score_hist": score_hist,
            }
        }

    def postprocess(self, accumulator):
        return accumulator


def getStage1Samples(stage1_path, year, sample_config, data_samples=[], sig_samples=[], bkg_samples=[], do_vbf_filter_study=False):
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
    # logger.info(f"bkg_samples: {bkg_samples}")
    for bkg_sample in bkg_samples:
        bkg_sample = bkg_sample.upper()
        if bkg_sample in bkg_sample_dict.keys():
            bkg_sample_l += bkg_sample_dict[bkg_sample]
            if (
                do_vbf_filter_study
                and bkg_sample == "DY"
                and "DYVBF" in bkg_sample_dict
            ):
                logger.info(
                    "Adding DYVBF samples because --vbf_filter_study was passed."
                )
                bkg_sample_l += bkg_sample_dict["DYVBF"]
    logger.info(f"bkg_sample_l: {bkg_sample_l}")
    bkg_sample_l = sorted(list(set(bkg_sample_l)))# safeguard against repetitions of MC samples for whatever reason
    for sample in bkg_sample_l:
        if "dy_vbf_filter" in sample.lower() and not do_vbf_filter_study:
            logger.info(
                "Skipping sample %s because --vbf_filter_study was not passed.",
                sample,
            )
            continue
        sample_filelist = glob.glob(str(stage1_path / sample / "*" / "*.parquet"))
        logger.info(f"sample: {sample}, number of files: {len(sample_filelist)}")
        logger.info(f"bkg_sample_l: {bkg_sample_l}")
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
        required=True,
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
    parser.add_argument(
        "--allow_nominal_feature_fallback",
        dest="allow_nominal_feature_fallback",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Allow shape variations to use nominal DNN input features when the "
            "shifted feature branch is missing."
        ),
    )
    parser.add_argument(
        "--use_nominal_dnn_features_for_systs",
        dest="use_nominal_dnn_features_for_systs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Use nominal DNN input features for systematic variations instead "
            "of shifted ones, even though shifted event selection and weights "
            "are still applied. Disabled by default: DNN features use the "
            "shifted branch for the current variation, falling back to "
            "nominal/base when that branch doesn't exist. Pass "
            "--use_nominal_dnn_features_for_systs to pin DNN inputs to nominal "
            "for all variations instead."
        ),
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
    years = parse_years_option(args.year, args.years)
    logger.info(f"Years scheduled together: {years}")

    bkg_samples = args.bkg_samples
    sig_samples = args.sig_samples
    data_samples = args.data_samples
    logger.info(f"data_samples: {data_samples}")

    histDirName = f"score_{args.label}" if args.save_postfix == "" else f"score_{args.label}_{args.save_postfix}"
    if args.do_vbf_filter_study:
        histDirName = f"{histDirName}_vbf_filter_study"
    if args.no_variations:
        histDirName = f"{histDirName}_NoSyst"

    sample_dict_by_year = {}
    for year in years:
        stage1_path = base_path / "stage1_output" / year / "f1_0"
        stage1_path = get_compacted_path(stage1_path) # get compacted stage1 output if they exist
        logger.info(f"{year} stage1 path: {stage1_path}")
        if not os.path.exists(stage1_path):
            logger.critical(f"Stage1 path {stage1_path} does not exist! Exiting!")
            raise FileNotFoundError(f"Stage1 path {stage1_path} does not exist! Run the compaction script first.")

        if dy_vbf_filter_is_available(stage1_path, year, args.sample_config):
            if not args.do_vbf_filter_study:
                logger.info(
                    "Setting do_vbf_filter_study=True because dy_VBF_filter "
                    "samples are available."
                )
            args.do_vbf_filter_study = True

        hist_save_path = base_path / "stage2_histograms" / histDirName / year
        os.makedirs(hist_save_path, exist_ok=True)
        logger.info(f"{year} histograms will be saved to: {hist_save_path}")

        full_sample_dict = getStage1Samples(stage1_path, year, args.sample_config, data_samples=data_samples, sig_samples=sig_samples, bkg_samples=bkg_samples, do_vbf_filter_study=args.do_vbf_filter_study)

        sample_dict_by_year[year] = full_sample_dict
        logger.debug(f"{year} full_sample_dict: {full_sample_dict}")
        logger.info(f"{year} full_sample_dict: {full_sample_dict.keys()}")
    t3 = time.perf_counter()
    logger.info(f"[timing] sample dict processing time: {t3 - t2:.2f} seconds")

    model_trained_path = Path(args.model_path).resolve()
    training_dir = (model_trained_path / args.model_tag).resolve()
    feature_dir = model_trained_path
    nfolds = args.nfolds

    missing = []
    for fold in range(nfolds):
        model_path = training_dir / f"fold{fold}" / "best_torchscript.pt"
        scaler_path = model_trained_path / f"scalers_{fold}.npz"
        if not model_path.is_file():
            missing.append(f"missing model: {model_path}")
        if not scaler_path.is_file():
            missing.append(f"missing scaler: {scaler_path}")
    if missing:
        raise FileNotFoundError("\n".join(missing))

    with open(feature_dir / "training_features.pkl", "rb") as feature_file:
        training_features = pickle.load(feature_file)
    logger.info(f"len training_features: {len(training_features)}")

    scalers = []
    model_paths = []
    for fold in range(nfolds):
        scaler_path = model_trained_path / f"scalers_{fold}.npz"
        with np.load(scaler_path, allow_pickle=True) as scaler_file:
            features = [str(feature) for feature in scaler_file["features"]]
            means = scaler_file["mean"].astype(np.float64)
            stds = scaler_file["std"].astype(np.float64)
        scalers.append(dict(zip(features, zip(means, stds))))
        model_paths.append(
            (training_dir / f"fold{fold}" / "best_torchscript.pt").as_posix()
        )

    fileset = {}
    skipped_existing = []
    for year, full_sample_dict in sample_dict_by_year.items():
        hist_save_path = base_path / "stage2_histograms" / histDirName / year
        for sample_type, sample_l in full_sample_dict.items():
            output_pkl_path = hist_save_path / f"{sample_type}_hist.pkl"
            if output_pkl_path.exists():
                logger.warning(
                    f"Output pkl file {output_pkl_path} already exists. "
                    f"Skipping {year} {sample_type}."
                )
                skipped_existing.append(f"{year}{DATASET_SEPARATOR}{sample_type}")
                continue
            if len(sample_l) == 0:
                logger.critical(f"No files for {year} {sample_type} is found! Skipping!")
                continue
            dataset_key = f"{year}{DATASET_SEPARATOR}{sample_type}"
            fileset[dataset_key] = {
                "files": sample_l,
                "treename": "Events",
                "metadata": {"year": year, "sample": sample_type},
            }

    logger.info(f"Samples skipped because outputs exist: {skipped_existing}")
    if fileset:
        runner = processor.Runner(
            executor=processor.DaskExecutor(client=client),
            schema=BaseSchema,
            format="parquet",
            chunksize=50_000,
        )
        t4 = time.perf_counter()
        results = runner(
            fileset,
            processor_instance=CoffeaStage2VBFProcessor(
                training_features=training_features,
                scalers=scalers,
                model_paths=model_paths,
                nfolds=nfolds,
                score_name=f"score_{args.label}",
                no_variations=args.no_variations,
                do_vbf_filter_study=args.do_vbf_filter_study,
                allow_nominal_feature_fallback=args.allow_nominal_feature_fallback,
                use_nominal_dnn_features_for_systs=args.use_nominal_dnn_features_for_systs,
            ),
        )
        t5 = time.perf_counter()
        logger.info(f"[timing] Coffea Runner time: {t5 - t4:.2f} seconds")

        for dataset_key, output in results.items():
            year, sample_type = dataset_key.split(DATASET_SEPARATOR, maxsplit=1)
            hist_save_path = base_path / "stage2_histograms" / histDirName / year
            output_pkl_path = hist_save_path / f"{sample_type}_hist.pkl"
            with open(output_pkl_path, "wb") as file:
                pickle.dump(output["score_hist"], file)
            logger.info(
                f"{year} {sample_type} histogram on {output_pkl_path}; "
                f"chunks={output['chunks'].value}; "
                f"selected events={output['events'].value}"
            )
    else:
        logger.warning("No samples left to process after existing-output checks.")

    logger.info("Success!")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time : {execution_time:.4f} seconds")
    raise SystemExit(0)

    logger.info("Success!")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time : {execution_time:.4f} seconds")
