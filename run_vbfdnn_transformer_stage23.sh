#!/usr/bin/env bash
set -euo pipefail

# Manifest-driven, one-model-at-a-time VBF-DNN Stage-2/3 and Combine workflow.
# This intentionally does not train a model. GPU training remains owned by the
# existing Ray training scripts; Stage-2 TorchScript inference runs on CPU Dask
# Gateway workers.

usage() {
    cat <<'EOF'
Usage:
  run_vbfdnn_transformer_stage23.sh run2 MODEL_LABEL \
      <validate|dump|scan|stage23|cards|stats|impacts|all>
  run_vbfdnn_transformer_stage23.sh run3 MODEL_LABEL \
      <validate|dump|scan|stage23|cards|stats|impacts|all>
  run_vbfdnn_transformer_stage23.sh combine RUN2_MODEL RUN3_MODEL \
      <cards|stats|impacts|all>

Single-model defaults:
  TRAIN_LABEL=trained_ray4
  ANALYSIS_LABEL=<MODEL_LABEL>
  DUMP_POSTFIX=TransformerScoreDump
  FINAL_POSTFIX=TransformerStage23
  JJ_ETA_REGION=all
  BINARY_TRANSFORMER_SIDEBAND_MASS=pin125  (or raw)
  SIGNAL_XSEC_RESCALE=0                    (Run 3 only when set to 1)
  ENFORCE_MIN_PER_YEAR=0                   (set to 1 for per-year scan guards)
  DASK_CLUSTER_INDEX=0

  Run 2 uses NanoAODv15 for all four eras, defaulting to:
    configs/datasets/dataset_nanoAODv15_run2.yaml
  Run 3 is version-split without mixing inputs in one Stage-2 invocation:
    2022preEE,2022postEE,2023,2023BPix: NanoAODv12 with
      configs/datasets/dataset_nanoAODv12_run3.yaml
    2024,2025,2026: NanoAODv15 with
      configs/datasets/dataset_nanoAODv15_run3.yaml
  DATASET_CONFIG_V12 and DATASET_CONFIG_V15 override each group;
  DATASET_CONFIG remains a compatibility override for every active group.
  NANOAOD_VERSION_V12/NANOAOD_VERSION_V15 may override the corresponding -v.

  STRICT_SCORED_SOURCE_VERSION=1 (default) checks each scored year's sample
  _SUCCESS.json markers against its expected NanoAOD version. Set it to 0 only
  for an explicitly audited future marker format. SCORED_SOURCE_V12_REGEX and
  SCORED_SOURCE_V15_REGEX may override the default source-path regexes.

  SAVE_ROOT defaults to
    /work/projects/hmm/<user>/hmm_ntuples/copperheadV1clean
  MODEL_DIR, PREPROCESS_MANIFEST and SCAN_OUTPUT_DIR may be overridden
  explicitly. Fixed bins are written to configs/MVA/VBF/dnn_binning.yaml.
  Threshold, score column, direction and sentinels are
  never parsed from MODEL_LABEL and have no shell override; Stage 2 reads them
  from PREPROCESS_MANIFEST.

Cross-Run combination:
  Run-2 and Run-3 models, Stage-2 histograms, and per-year Stage-3 templates/
  SR/SB cards are always produced independently. Use the single-model `cards`
  step to build HMuMu_13TeV_Run2.txt and HMuMu_13TeV_Run3.txt first. The
  `combine` action combines only those two total cards; it never adds histograms
  or ROOT templates across runs.
  VBF_RUN2_CARD_PATH and VBF_RUN3_CARD_PATH may explicitly select the input
  cards. Otherwise they are derived from the two model labels and these vars:
  RUN{2,3}_SAVE_ROOT, RUN{2,3}_ANALYSIS_LABEL, RUN{2,3}_FINAL_POSTFIX,
  RUN{2,3}_JJ_ETA_REGION, and RUN3_SIGNAL_XSEC_RESCALE.
  The combined result is written separately using COMBINED_SAVE_ROOT,
  COMBINED_ANALYSIS_LABEL and COMBINED_FINAL_POSTFIX.

Safety:
  The runner never deletes output. A non-empty managed output without its
  matching completion marker is treated as ambiguous/partial and refused.
  Choose a new postfix (or move the old output aside) after inspecting it.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

print_cmd() {
    local arg
    printf '[command]'
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    "$@"
}

validate_bool() {
    local name="$1"
    local value="$2"
    [[ "${value}" == "0" || "${value}" == "1" ]] || \
        die "${name} must be 0 or 1, got '${value}'."
}

validate_token() {
    local name="$1"
    local value="$2"
    [[ -n "${value}" ]] || die "${name} cannot be empty."
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
        die "${name} contains unsafe path characters: '${value}'."
}

directory_nonempty() {
    local directory="$1"
    [[ -d "${directory}" ]] && \
        [[ -n "$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

require_nonempty_file() {
    local path="$1"
    [[ -s "${path}" ]] || die "Required non-empty file is missing: ${path}"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GLOBAL_PYTHON="/work/pixi/global/.pixi/envs/default/bin/python"
RUN_IN_PIXI="${REPO_ROOT}/run_in_pixi.sh"
ANALYSIS_PIPELINE="${REPO_ROOT}/run_analysis_pipeline.sh"
STATS_PIPELINE="${REPO_ROOT}/run_stats_pipeline_VBF.sh"
SCAN_SCRIPT="${REPO_ROOT}/MVA_training/VBF_run3/scan_bins_for_dnn.py"

[[ -x "${GLOBAL_PYTHON}" ]] || die "Global Python is missing: ${GLOBAL_PYTHON}"
[[ -x "${RUN_IN_PIXI}" ]] || die "Pixi runner is missing or not executable: ${RUN_IN_PIXI}"
[[ -f "${ANALYSIS_PIPELINE}" ]] || die "Analysis pipeline is missing: ${ANALYSIS_PIPELINE}"
[[ -f "${STATS_PIPELINE}" ]] || die "Stats pipeline is missing: ${STATS_PIPELINE}"
[[ -f "${SCAN_SCRIPT}" ]] || die "DNN bin scanner is missing: ${SCAN_SCRIPT}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

AF_USERNAME="${NB_USER:-$(whoami)}"
DEFAULT_SAVE_ROOT="/work/projects/hmm/${AF_USERNAME}/hmm_ntuples/copperheadV1clean"

configure_single() {
    RUN_KIND="$1"
    MODEL_LABEL="$2"
    validate_token "MODEL_LABEL" "${MODEL_LABEL}"

    case "${RUN_KIND}" in
        run2)
            YEARS_CSV="2016preVFP,2016postVFP,2017,2018"
            V12_YEARS_CSV=""
            V15_YEARS_CSV="${YEARS_CSV}"
            YEAR_SLUG="2016preVFP-2016postVFP-2017-2018"
            STATS_YEAR="Run2"
            DEFAULT_DATASET_CONFIG_V12=""
            DEFAULT_DATASET_CONFIG_V15="${REPO_ROOT}/configs/datasets/dataset_nanoAODv15_run2.yaml"
            DNN_CONFIG_PATH="${REPO_ROOT}/configs/dnn_run2_vbf.yaml"
            ;;
        run3)
            YEARS_CSV="2022preEE,2022postEE,2023,2023BPix,2024,2025,2026"
            V12_YEARS_CSV="2022preEE,2022postEE,2023,2023BPix"
            V15_YEARS_CSV="2024,2025,2026"
            YEAR_SLUG="2022preEE-2022postEE-2023-2023BPix-2024-2025-2026"
            STATS_YEAR="Run3"
            DEFAULT_DATASET_CONFIG_V12="${REPO_ROOT}/configs/datasets/dataset_nanoAODv12_run3.yaml"
            DEFAULT_DATASET_CONFIG_V15="${REPO_ROOT}/configs/datasets/dataset_nanoAODv15_run3.yaml"
            DNN_CONFIG_PATH="${REPO_ROOT}/configs/dnn_run3_vbf.yaml"
            ;;
        *) die "Run must be 'run2' or 'run3', got '${RUN_KIND}'." ;;
    esac
    IFS=',' read -r -a RUN_YEARS <<< "${YEARS_CSV}"

    TRAIN_LABEL="${TRAIN_LABEL:-trained_ray4}"
    JJ_ETA_REGION="${JJ_ETA_REGION:-all}"
    ANALYSIS_LABEL="${ANALYSIS_LABEL:-${MODEL_LABEL}}"
    DUMP_POSTFIX="${DUMP_POSTFIX:-TransformerScoreDump}"
    FINAL_POSTFIX="${FINAL_POSTFIX:-TransformerStage23}"
    SAVE_ROOT="$(realpath -m "${SAVE_ROOT:-${DEFAULT_SAVE_ROOT}}")"
    ANALYSIS_PATH="${SAVE_ROOT}/${ANALYSIS_LABEL}"
    if [[ -n "${V12_YEARS_CSV}" ]]; then
        DATASET_CONFIG_V12="$(realpath -m "${DATASET_CONFIG_V12:-${DATASET_CONFIG:-${DEFAULT_DATASET_CONFIG_V12}}}")"
    else
        DATASET_CONFIG_V12=""
    fi
    DATASET_CONFIG_V15="$(realpath -m "${DATASET_CONFIG_V15:-${DATASET_CONFIG:-${DEFAULT_DATASET_CONFIG_V15}}}")"
    NANOAOD_VERSION_V12="${NANOAOD_VERSION_V12:-${NANOAOD_VERSION:-12}}"
    NANOAOD_VERSION_V15="${NANOAOD_VERSION_V15:-${NANOAOD_VERSION:-15}}"
    # A single config/version is still needed by plotting and stats, which do
    # not read the scored input.  Use the v15 group; Stage 2/3 are routed below.
    DATASET_CONFIG="${DATASET_CONFIG_V15}"
    NANOAOD_VERSION="${NANOAOD_VERSION_V15}"
    MODEL_DIR="$(realpath -m "${MODEL_DIR:-${REPO_ROOT}/dnn/trained_models/${MODEL_LABEL}/${YEAR_SLUG}_h-peak_vbf_${JJ_ETA_REGION}}")"
    PREPROCESS_MANIFEST="$(realpath -m "${PREPROCESS_MANIFEST:-${MODEL_DIR}/preprocess_manifest.json}")"
    BINNING_CONFIG="${REPO_ROOT}/configs/MVA/VBF/dnn_binning.yaml"
    SCAN_OUTPUT_DIR="$(realpath -m "${SCAN_OUTPUT_DIR:-${REPO_ROOT}/validation/dnn_binning_scan/${MODEL_LABEL}_${DUMP_POSTFIX}_${JJ_ETA_REGION}}")"
    BINARY_TRANSFORMER_SIDEBAND_MASS="${BINARY_TRANSFORMER_SIDEBAND_MASS:-pin125}"
    SIGNAL_XSEC_RESCALE="${SIGNAL_XSEC_RESCALE:-0}"
    ENFORCE_MIN_PER_YEAR="${ENFORCE_MIN_PER_YEAR:-0}"
    DASK_CLUSTER_INDEX="${DASK_CLUSTER_INDEX:-0}"
    STRICT_SCORED_SOURCE_VERSION="${STRICT_SCORED_SOURCE_VERSION:-1}"
    SCORED_SOURCE_V12_REGEX="${SCORED_SOURCE_V12_REGEX:-(?i)nano(?:aod)?v?12}"
    SCORED_SOURCE_V15_REGEX="${SCORED_SOURCE_V15_REGEX:-(?i)nano(?:aod)?v?15}"

    validate_token "TRAIN_LABEL" "${TRAIN_LABEL}"
    validate_token "ANALYSIS_LABEL" "${ANALYSIS_LABEL}"
    validate_token "DUMP_POSTFIX" "${DUMP_POSTFIX}"
    validate_token "FINAL_POSTFIX" "${FINAL_POSTFIX}"
    validate_token "JJ_ETA_REGION" "${JJ_ETA_REGION}"
    [[ "${DUMP_POSTFIX}" != "${FINAL_POSTFIX}" ]] || \
        die "DUMP_POSTFIX and FINAL_POSTFIX must be different."
    [[ "${BINARY_TRANSFORMER_SIDEBAND_MASS}" == "pin125" || \
       "${BINARY_TRANSFORMER_SIDEBAND_MASS}" == "raw" ]] || \
        die "BINARY_TRANSFORMER_SIDEBAND_MASS must be pin125 or raw."
    validate_bool "SIGNAL_XSEC_RESCALE" "${SIGNAL_XSEC_RESCALE}"
    validate_bool "ENFORCE_MIN_PER_YEAR" "${ENFORCE_MIN_PER_YEAR}"
    validate_bool "STRICT_SCORED_SOURCE_VERSION" "${STRICT_SCORED_SOURCE_VERSION}"
    [[ "${NANOAOD_VERSION_V12}" =~ ^[0-9]+$ ]] || \
        die "NANOAOD_VERSION_V12 must be numeric."
    [[ "${NANOAOD_VERSION_V15}" =~ ^[0-9]+$ ]] || \
        die "NANOAOD_VERSION_V15 must be numeric."
    [[ "${SCORED_SOURCE_V12_REGEX}" != *$'\n'* && \
       "${SCORED_SOURCE_V15_REGEX}" != *$'\n'* ]] || \
        die "Scored-source regex overrides must be single-line values."
    [[ "${DASK_CLUSTER_INDEX}" =~ ^[0-9]+$ ]] || \
        die "DASK_CLUSTER_INDEX must be a non-negative integer."
    if [[ "${RUN_KIND}" == "run2" && "${SIGNAL_XSEC_RESCALE}" == "1" ]]; then
        die "SIGNAL_XSEC_RESCALE applies only to Run 3."
    fi
}

validate_single_contract() {
    run_cmd "${GLOBAL_PYTHON}" - \
        "${RUN_KIND}" "${MODEL_LABEL}" "${MODEL_DIR}" "${PREPROCESS_MANIFEST}" \
        "${TRAIN_LABEL}" "${YEARS_CSV}" "${JJ_ETA_REGION}" \
        "${DATASET_CONFIG_V12}" "${V12_YEARS_CSV}" "${NANOAOD_VERSION_V12}" \
        "${SCORED_SOURCE_V12_REGEX}" \
        "${DATASET_CONFIG_V15}" "${V15_YEARS_CSV}" "${NANOAOD_VERSION_V15}" \
        "${SCORED_SOURCE_V15_REGEX}" "${STRICT_SCORED_SOURCE_VERSION}" <<'PY'
import json
import math
import pickle
import re
import sys
from pathlib import Path

import yaml

(
    run_kind,
    model_label,
    model_dir_text,
    manifest_text,
    train_label,
    years_csv,
    jj_eta_region,
    dataset_v12_text,
    v12_years_csv,
    nanoaod_v12,
    source_v12_regex,
    dataset_v15_text,
    v15_years_csv,
    nanoaod_v15,
    source_v15_regex,
    strict_source_version_text,
) = sys.argv[1:]

model_dir = Path(model_dir_text)
manifest_path = Path(manifest_text)
expected_years = years_csv.split(",")
v12_years = [year for year in v12_years_csv.split(",") if year]
v15_years = [year for year in v15_years_csv.split(",") if year]
strict_source_version = strict_source_version_text == "1"

def fail(message):
    raise SystemExit(f"[contract] {message}")

if model_dir.parent.name != model_label:
    fail(
        f"model path parent '{model_dir.parent.name}' does not equal explicit "
        f"MODEL_LABEL '{model_label}'"
    )
for path, description in (
    (manifest_path, "preprocess manifest"),
    (model_dir / "training_features.pkl", "training feature list"),
    (model_dir / train_label / "manifest.json", "training manifest"),
):
    if not path.is_file():
        fail(f"missing {description}: {path}")

grouped_years = v12_years + v15_years
if grouped_years != expected_years or len(grouped_years) != len(set(grouped_years)):
    fail(
        f"NanoAOD version groups {grouped_years} do not exactly partition "
        f"the requested years {expected_years}"
    )
dataset_groups = (
    ("v12", dataset_v12_text, v12_years, nanoaod_v12, source_v12_regex),
    ("v15", dataset_v15_text, v15_years, nanoaod_v15, source_v15_regex),
)
for name, dataset_text, group_years, version, source_regex in dataset_groups:
    if not group_years:
        continue
    dataset_path = Path(dataset_text)
    if not dataset_path.is_file():
        fail(f"missing NanoAOD{name[1:]} dataset config: {dataset_path}")
    if not version.isdigit():
        fail(f"NanoAOD version for {name} must be numeric, got {version!r}")
    try:
        re.compile(source_regex)
    except re.error as error:
        fail(f"invalid scored-source regex for {name}: {error}")

with manifest_path.open() as stream:
    manifest = json.load(stream)
selection = manifest.get("selection") or {}
if selection.get("mode") != "transformer":
    fail("selection.mode must be 'transformer'")
if selection.get("vbf_score_operator") != "lt":
    fail("selection.vbf_score_operator must be 'lt'")
if not isinstance(selection.get("score_column"), str) or not selection["score_column"]:
    fail("selection.score_column is missing")
try:
    threshold = float(selection["score_threshold"])
except (KeyError, TypeError, ValueError):
    fail("selection.score_threshold is missing or non-numeric")
if not math.isfinite(threshold):
    fail("selection.score_threshold must be finite")
mass_window = selection.get("mass_window_gev") or {}
if mass_window.get("lower_inclusive") != 115.0 or mass_window.get("upper_exclusive") != 135.0:
    fail("selection.mass_window_gev must be [115, 135)")
if not selection.get("btag_veto"):
    fail("selection.btag_veto is missing")
if manifest.get("years") != expected_years:
    fail(f"manifest years {manifest.get('years')} != expected {expected_years}")
if int(manifest.get("n_folds", 0)) != 4:
    fail("manifest n_folds must be 4")
if manifest.get("jj_eta_region", "all") != jj_eta_region:
    fail(
        f"manifest jj_eta_region={manifest.get('jj_eta_region')} != "
        f"requested {jj_eta_region}"
    )
systematics = manifest.get("systematic_variations") or {}
if bool(systematics.get("enabled", False)):
    fail("binary Transformer production is nominal-only, but manifest enables variations")
preprocessing = manifest.get("feature_preprocessing") or {}
if preprocessing.get("sentinel_space") != "raw_before_scaling":
    fail("feature_preprocessing.sentinel_space must be raw_before_scaling")
if not isinstance(preprocessing.get("sentinels"), dict) or not preprocessing["sentinels"]:
    fail("feature_preprocessing.sentinels is missing")
features = manifest.get("training_features")
if not isinstance(features, list) or not features:
    fail("manifest training_features is missing")
with (model_dir / "training_features.pkl").open("rb") as stream:
    persisted_features = list(pickle.load(stream))
if persisted_features != features:
    fail("training_features.pkl does not match preprocess_manifest.json")

for fold in range(4):
    scaler = model_dir / f"scalers_{fold}.npz"
    checkpoint = model_dir / train_label / f"fold{fold}" / "best_torchscript.pt"
    if not scaler.is_file():
        fail(f"missing scaler: {scaler}")
    if not checkpoint.is_file():
        fail(f"missing TorchScript checkpoint: {checkpoint}")

base_path = Path(manifest.get("base_path", ""))
if not base_path.is_dir():
    fail(f"manifest base_path is not a directory: {base_path}")

year_contract = {}
dataset_summaries = []
for name, dataset_text, group_years, version, source_regex in dataset_groups:
    if not group_years:
        continue
    dataset_path = Path(dataset_text)
    with dataset_path.open() as stream:
        dataset = yaml.safe_load(stream) or {}
    dataset_years = set((dataset.get("years") or {}).keys())
    missing_dataset_years = [year for year in group_years if year not in dataset_years]
    if missing_dataset_years:
        fail(
            f"dataset config {dataset_path} is incompatible with {run_kind}; "
            f"missing {name} years {missing_dataset_years}"
        )
    dataset_summaries.append((name, version, dataset_path, group_years))
    for year in group_years:
        year_contract[year] = (version, source_regex)

def source_contract_issue(message):
    if strict_source_version:
        fail(message)
    print(f"[contract warning] {message}", file=sys.stderr)

marker_counts = {}
for year in expected_years:
    year_roots = [
        candidate / year
        for candidate in (base_path, base_path / "stage1_output")
        if (candidate / year).is_dir()
    ]
    if not year_roots:
        fail(f"no scored input year directory for {year} below {base_path}")
    if len(year_roots) > 1:
        fail(f"ambiguous scored input year directories for {year}: {year_roots}")
    year_root = year_roots[0]
    # Per-sample markers carry the actual `source`. Do not accept a top-level
    # _RAY_SUCCESS marker as evidence for an era's NanoAOD provenance.
    success_markers = sorted(year_root.rglob("_SUCCESS.json"))
    marker_counts[year] = len(success_markers)
    if not success_markers:
        source_contract_issue(
            f"no per-sample _SUCCESS.json markers found for {year} below {year_root}"
        )
        continue
    version, source_regex = year_contract[year]
    for marker_path in success_markers:
        try:
            marker = json.loads(marker_path.read_text())
        except Exception as error:
            source_contract_issue(f"cannot read scored marker {marker_path}: {error}")
            continue
        if str(marker.get("year", year)) != year:
            source_contract_issue(
                f"scored marker {marker_path} declares year={marker.get('year')!r}, "
                f"expected {year}"
            )
        source = marker.get("source")
        if not isinstance(source, str) or not source:
            source_contract_issue(f"scored marker has no source path: {marker_path}")
        elif re.search(source_regex, source) is None:
            source_contract_issue(
                f"scored marker {marker_path} source does not identify expected "
                f"NanoAODv{version}: {source}"
            )

print(f"Validated {run_kind} model: {model_label}")
print(f"  manifest: {manifest_path}")
print(f"  input:    {base_path}")
print(f"  years:    {','.join(expected_years)}")
print(f"  contract: {selection['score_column']} < {threshold}")
print(f"  model:    {model_dir / train_label}")
for name, version, dataset_path, group_years in dataset_summaries:
    counts = ", ".join(f"{year}:{marker_counts[year]}" for year in group_years)
    print(
        f"  NanoAODv{version}: {','.join(group_years)}; dataset={dataset_path}; "
        f"scored markers={counts}"
    )
PY
}

jj_suffix() {
    [[ "${JJ_ETA_REGION}" == "all" ]] || printf '_%s' "${JJ_ETA_REGION}"
}

stage2_hist_dir() {
    local postfix="$1"
    printf '%s/stage2_histograms/score_%s_%s_vbf_filter_study%s_NoSyst' \
        "${ANALYSIS_PATH}" "${ANALYSIS_LABEL}" "${postfix}" "$(jj_suffix)"
}

stage2_plot_dir() {
    printf '%s/validation/stage2_dnn_score/%s/%s_vbf_filter_study%s_NoSyst' \
        "${REPO_ROOT}" "${ANALYSIS_LABEL}" "${FINAL_POSTFIX}" "$(jj_suffix)"
}

stage3_suffix() {
    local suffix="${FINAL_POSTFIX}"
    if [[ "${SIGNAL_XSEC_RESCALE}" == "1" ]]; then
        suffix+="_SigXS"
    fi
    suffix+="_vbf_filter_study$(jj_suffix)"
    printf '%s' "${suffix}"
}

stats_postfix() {
    local suffix="${FINAL_POSTFIX}"
    [[ "${SIGNAL_XSEC_RESCALE}" == "0" ]] || suffix+="_SigXS"
    printf '%s' "${suffix}"
}

single_card_dir() {
    printf '%s/stage3_datacards_%s/score_%s' \
        "${ANALYSIS_PATH}" "$(stage3_suffix)" "${ANALYSIS_LABEL}"
}

single_template_dir() {
    local suffix
    suffix="$(stage3_suffix)"
    printf '%s/stage3_datacards_%s/stage3_templates_%s/score_%s' \
        "${ANALYSIS_PATH}" "${suffix}" "${suffix}" "${ANALYSIS_LABEL}"
}

binning_entry_hash() {
    "${GLOBAL_PYTHON}" - "${BINNING_CONFIG}" "${MODEL_LABEL}" "${JJ_ETA_REGION}" <<'PY'
import hashlib
import json
import sys

import yaml

with open(sys.argv[1]) as stream:
    payload = yaml.safe_load(stream) or {}
try:
    entry = payload["models"][sys.argv[2]][sys.argv[3]]
except KeyError as error:
    raise SystemExit(
        f"Missing models.{sys.argv[2]}.{sys.argv[3]} in {sys.argv[1]}"
    ) from error
canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(canonical).hexdigest())
PY
}

verify_binning_entry() {
    local ignored
    ignored="$(binning_entry_hash)" || die \
        "No fixed DNN binning for ${MODEL_LABEL}/${JJ_ETA_REGION}; run the scan step first."
}

file_sha256_or_none() {
    local path="$1"
    if [[ -n "${path}" ]]; then
        sha256sum "${path}" | awk '{print $1}'
    else
        printf 'none\n'
    fi
}

single_marker_payload() {
    local kind="$1"
    cat <<EOF
schema=2
kind=${kind}
run=${RUN_KIND}
model_label=${MODEL_LABEL}
model_dir=${MODEL_DIR}
manifest=${PREPROCESS_MANIFEST}
manifest_sha256=$(sha256sum "${PREPROCESS_MANIFEST}" | awk '{print $1}')
train_label=${TRAIN_LABEL}
years=${YEARS_CSV}
analysis_path=${ANALYSIS_PATH}
dump_postfix=${DUMP_POSTFIX}
final_postfix=${FINAL_POSTFIX}
jj_eta_region=${JJ_ETA_REGION}
sideband_mass=${BINARY_TRANSFORMER_SIDEBAND_MASS}
signal_xsec_rescale=${SIGNAL_XSEC_RESCALE}
nanoaod_v12_years=${V12_YEARS_CSV:-none}
nanoaod_v12_version=${NANOAOD_VERSION_V12}
dataset_config_v12=${DATASET_CONFIG_V12:-none}
dataset_config_v12_sha256=$(file_sha256_or_none "${DATASET_CONFIG_V12}")
nanoaod_v15_years=${V15_YEARS_CSV:-none}
nanoaod_v15_version=${NANOAOD_VERSION_V15}
dataset_config_v15=${DATASET_CONFIG_V15}
dataset_config_v15_sha256=$(file_sha256_or_none "${DATASET_CONFIG_V15}")
strict_scored_source_version=${STRICT_SCORED_SOURCE_VERSION}
scored_source_v12_regex=${SCORED_SOURCE_V12_REGEX}
scored_source_v15_regex=${SCORED_SOURCE_V15_REGEX}
dy_matched_jets=gen_2j
nominal_only=1
vbf_filter_study=1
EOF
    case "${kind}" in
        scan|stage2|stage23|cards|stats|impacts)
            printf 'binning_entry_sha256=%s\n' "$(binning_entry_hash)"
            ;;
    esac
    if [[ "${kind}" == "scan" ]]; then
        printf 'enforce_min_per_year=%s\n' "${ENFORCE_MIN_PER_YEAR}"
    fi
}

check_single_marker() {
    local marker="$1"
    local kind="$2"
    cmp -s <(single_marker_payload "${kind}") "${marker}" || \
        die "Completion marker does not match the requested contract: ${marker}"
}

write_single_marker() {
    local marker="$1"
    local kind="$2"
    mkdir -p "$(dirname "${marker}")"
    single_marker_payload "${kind}" > "${marker}"
    log "Wrote completion marker: ${marker}"
}

MANAGED_COMPLETE=0
guard_managed_outputs() {
    local marker="$1"
    local kind="$2"
    local description="$3"
    shift 3
    MANAGED_COMPLETE=0
    if [[ -f "${marker}" ]]; then
        check_single_marker "${marker}" "${kind}"
        MANAGED_COMPLETE=1
        return
    fi
    local directory
    for directory in "$@"; do
        if directory_nonempty "${directory}"; then
            die "${description} exists without a matching completion marker: ${directory}. Refusing to delete or mix it; inspect it and choose a new postfix."
        fi
    done
}

assert_no_newer_files() {
    local directory="$1"
    local marker="$2"
    local description="$3"
    local newer=""
    [[ -d "${directory}" ]] && \
        newer="$(find "${directory}" -type f ! -path "${marker}" -newer "${marker}" -print -quit 2>/dev/null || true)"
    [[ -z "${newer}" ]] || die "${description} changed after its completion marker: ${newer}"
}

verify_stage2_outputs() {
    local hist_dir="$1"
    local require_scores="$2"
    local year year_dir
    for year in "${RUN_YEARS[@]}"; do
        year_dir="${hist_dir}/${year}"
        [[ -d "${year_dir}" ]] || die "Missing Stage-2 year directory: ${year_dir}"
        [[ -n "$(find "${year_dir}" -maxdepth 1 -type f -name '*_hist.pkl' -size +0c -print -quit)" ]] || \
            die "No non-empty Stage-2 histograms found in ${year_dir}"
        require_nonempty_file "${year_dir}/dnn_binning.yaml"
        require_nonempty_file "${year_dir}/binary_transformer_contract.json"
        if [[ "${require_scores}" == "1" ]]; then
            [[ -n "$(find "${year_dir}/scores" -maxdepth 1 -type f -name '*_scores.parquet' -size +0c -print -quit 2>/dev/null)" ]] || \
                die "No non-empty per-event score dumps found in ${year_dir}/scores"
        fi
    done
}

verify_stage2_plots() {
    local plot_dir year filename
    plot_dir="$(stage2_plot_dir)"
    local -a expected_years=("${RUN_YEARS[@]}")
    [[ "${RUN_KIND}" != "run3" ]] || expected_years+=("Run3")
    for year in "${expected_years[@]}"; do
        for filename in \
            DNN_score_vbf_h-sidebands.pdf \
            DNN_score_vbf_h-sidebands_log.pdf \
            DNN_score_vbf_h-peak.pdf \
            DNN_score_vbf_h-peak_log.pdf; do
            require_nonempty_file "${plot_dir}/${year}/${filename}"
        done
    done
}

verify_stage3_outputs() {
    local card_dir template_dir year
    card_dir="$(single_card_dir)"
    template_dir="$(single_template_dir)"
    local -a expected_years=("${RUN_YEARS[@]}")
    [[ "${RUN_KIND}" != "run3" ]] || expected_years+=("2025_2026")
    for year in "${expected_years[@]}"; do
        require_nonempty_file "${card_dir}/datacard_vbf_SR_${year}.txt"
        require_nonempty_file "${card_dir}/datacard_vbf_SB_${year}.txt"
        require_nonempty_file "${template_dir}/vbf_h-peak_${year}.root"
        require_nonempty_file "${template_dir}/vbf_h-sidebands_${year}.root"
    done
}

run_analysis_mode() {
    local mode="$1"
    local years="$2"
    local postfix="$3"
    local dump_scores="$4"
    local allow_default_binning="$5"
    local nanoaod_version="${6:-${NANOAOD_VERSION}}"
    local dataset_config="${7:-${DATASET_CONFIG}}"
    local -a command=(
        env
        "WITH_VARIATIONS=0"
        "DNN_CONFIG=${DNN_CONFIG_PATH}"
        "MODEL_LABEL=${MODEL_LABEL}"
        "MODEL_YEARS=${YEARS_CSV}"
        "TRAIN_LABEL=${TRAIN_LABEL}"
        "DNN_MODEL_REGION=${JJ_ETA_REGION}"
        "JJ_ETA_REGION=${JJ_ETA_REGION}"
        "BINARY_TRANSFORMER_MANIFEST=${PREPROCESS_MANIFEST}"
        "BINARY_TRANSFORMER_SIDEBAND_MASS=${BINARY_TRANSFORMER_SIDEBAND_MASS}"
        "DUMP_SCORES=${dump_scores}"
        "ALLOW_DEFAULT_DNN_BINNING=${allow_default_binning}"
        "DY_MATCHED_JETS=gen_2j"
        "SIGNAL_XSEC_RESCALE=${SIGNAL_XSEC_RESCALE}"
        "${RUN_IN_PIXI}" default bash "${ANALYSIS_PIPELINE}"
        -m "${mode}"
        -y "${years}"
        -v "${nanoaod_version}"
        -c "${dataset_config}"
        -l "${ANALYSIS_LABEL}"
        -o "${postfix}"
        -S "${SAVE_ROOT}"
        -i "${DASK_CLUSTER_INDEX}"
        -k
        -V
    )
    run_cmd "${command[@]}"
}

run_analysis_mode_by_nanoaod() {
    local mode="$1"
    local postfix="$2"
    local dump_scores="$3"
    local allow_default_binning="$4"
    if [[ -n "${V12_YEARS_CSV}" ]]; then
        run_analysis_mode "${mode}" "${V12_YEARS_CSV}" "${postfix}" \
            "${dump_scores}" "${allow_default_binning}" \
            "${NANOAOD_VERSION_V12}" "${DATASET_CONFIG_V12}"
    fi
    if [[ -n "${V15_YEARS_CSV}" ]]; then
        run_analysis_mode "${mode}" "${V15_YEARS_CSV}" "${postfix}" \
            "${dump_scores}" "${allow_default_binning}" \
            "${NANOAOD_VERSION_V15}" "${DATASET_CONFIG_V15}"
    fi
}

run_dump() {
    local hist_dir marker
    hist_dir="$(stage2_hist_dir "${DUMP_POSTFIX}")"
    marker="${hist_dir}/.vbfdnn_dump_complete"
    guard_managed_outputs "${marker}" dump "Stage-2 score dump" "${hist_dir}"
    if [[ "${MANAGED_COMPLETE}" == "1" ]]; then
        verify_stage2_outputs "${hist_dir}" 1
        assert_no_newer_files "${hist_dir}" "${marker}" "Stage-2 score dump"
        log "Reusing validated score dump: ${hist_dir}"
        return
    fi
    run_analysis_mode_by_nanoaod 2 "${DUMP_POSTFIX}" 1 1
    verify_stage2_outputs "${hist_dir}" 1
    write_single_marker "${marker}" dump
}

run_scan() {
    local hist_dir dump_marker scan_marker
    hist_dir="$(stage2_hist_dir "${DUMP_POSTFIX}")"
    dump_marker="${hist_dir}/.vbfdnn_dump_complete"
    [[ -f "${dump_marker}" ]] || die "The scan step requires a runner-validated dump: ${dump_marker}"
    check_single_marker "${dump_marker}" dump
    verify_stage2_outputs "${hist_dir}" 1

    scan_marker="${SCAN_OUTPUT_DIR}/.vbfdnn_scan_complete"
    guard_managed_outputs "${scan_marker}" scan "DNN bin scan" "${SCAN_OUTPUT_DIR}"
    if [[ "${MANAGED_COMPLETE}" == "1" ]]; then
        verify_binning_entry
        require_nonempty_file "${SCAN_OUTPUT_DIR}/binning_comparison.txt"
        require_nonempty_file "${SCAN_OUTPUT_DIR}/dnn_binning_scanned.yaml"
        [[ -z "$(find "${hist_dir}" -type f -newer "${scan_marker}" -print -quit)" ]] || \
            die "Score dumps changed after the completed bin scan; use a new SCAN_OUTPUT_DIR."
        log "Reusing validated bin scan: ${SCAN_OUTPUT_DIR}"
        return
    fi

    local -a command=(
        "${RUN_IN_PIXI}" default python "${SCAN_SCRIPT}"
        --stage2-scores "${hist_dir}"
        --years "${YEARS_CSV}"
        --region h-peak
        --signal vbf,ggh
        --output-dir "${SCAN_OUTPUT_DIR}"
        --write-config
        --model-label "${MODEL_LABEL}"
        --jj-region "${JJ_ETA_REGION}"
    )
    if [[ "${ENFORCE_MIN_PER_YEAR}" == "1" ]]; then
        command+=(--enforce-min-per-year)
    fi
    run_cmd "${command[@]}"
    verify_binning_entry
    require_nonempty_file "${SCAN_OUTPUT_DIR}/binning_comparison.txt"
    require_nonempty_file "${SCAN_OUTPUT_DIR}/dnn_binning_scanned.yaml"
    write_single_marker "${scan_marker}" scan
}

run_stage23() {
    verify_binning_entry
    local hist_dir stage2_marker plot_dir card_dir template_dir stage23_marker
    hist_dir="$(stage2_hist_dir "${FINAL_POSTFIX}")"
    stage2_marker="${hist_dir}/.vbfdnn_stage2_complete"
    plot_dir="$(stage2_plot_dir)"
    card_dir="$(single_card_dir)"
    template_dir="$(single_template_dir)"
    stage23_marker="${card_dir}/.vbfdnn_stage23_complete"

    guard_managed_outputs "${stage2_marker}" stage2 "Final Stage-2 output" "${hist_dir}"
    local stage2_complete="${MANAGED_COMPLETE}"
    guard_managed_outputs "${stage23_marker}" stage23 "Stage-2 plots / Stage-3 output" \
        "${plot_dir}" "${card_dir}" "${template_dir}"
    local stage23_complete="${MANAGED_COMPLETE}"

    if [[ "${stage23_complete}" == "1" && "${stage2_complete}" != "1" ]]; then
        die "Stage-3 is marked complete but its final Stage-2 input is not; refusing an inconsistent rerun."
    fi
    if [[ "${stage2_complete}" != "1" ]]; then
        run_analysis_mode_by_nanoaod 2 "${FINAL_POSTFIX}" 0 0
        verify_stage2_outputs "${hist_dir}" 0
        write_single_marker "${stage2_marker}" stage2
    else
        verify_stage2_outputs "${hist_dir}" 0
        assert_no_newer_files "${hist_dir}" "${stage2_marker}" "Final Stage-2 output"
    fi

    if [[ "${stage23_complete}" == "1" ]]; then
        verify_stage2_plots
        verify_stage3_outputs
        [[ -z "$(find "${hist_dir}" -type f -newer "${stage23_marker}" -print -quit)" ]] || \
            die "Final Stage-2 inputs changed after Stage-3 completed; choose a new FINAL_POSTFIX."
        log "Reusing validated Stage-2 plots and Stage-3 output: ${card_dir}"
        return
    fi

    # Mode 2p runs both h-sidebands and h-peak plots for every year (and the
    # combined Run-3 plot). Keep all seven Run-3 eras in this single call so the
    # combined plot is produced exactly once from the full version-split set.
    run_analysis_mode 2p "${YEARS_CSV}" "${FINAL_POSTFIX}" 0 0
    verify_stage2_plots
    run_analysis_mode_by_nanoaod 3 "${FINAL_POSTFIX}" 0 0
    if [[ "${RUN_KIND}" == "run3" ]]; then
        run_analysis_mode 3 "2025_2026" "${FINAL_POSTFIX}" 0 0 \
            "${NANOAOD_VERSION_V15}" "${DATASET_CONFIG_V15}"
    fi
    verify_stage3_outputs
    write_single_marker "${stage23_marker}" stage23
}

configure_single_stats_context() {
    verify_binning_entry
    STATS_CONTEXT="single"
    STATS_CARD_DIR="$(single_card_dir)"
    STATS_LABEL="${ANALYSIS_LABEL}"
    STATS_SAVE_ROOT="${SAVE_ROOT}"
    STATS_POSTFIX="$(stats_postfix)"
    STATS_DATASET_CONFIG="${DATASET_CONFIG}"
    STATS_NANOAOD_VERSION="${NANOAOD_VERSION}"
    STATS_JJ_ETA_REGION="${JJ_ETA_REGION}"
    STATS_MODE_FOR_RESULTS="11"
}

stats_stem() {
    printf 'HMuMu_13TeV_%s' "${STATS_YEAR}"
}

stats_marker_payload() {
    local kind="$1"
    if [[ "${STATS_CONTEXT}" == "single" ]]; then
        single_marker_payload "${kind}"
        printf 'stats_year=%s\nstats_postfix=%s\n' "${STATS_YEAR}" "${STATS_POSTFIX}"
    else
        cat <<EOF
schema=1
kind=${kind}
context=combine
run2_model_label=${COMBINE_RUN2_MODEL}
run3_model_label=${COMBINE_RUN3_MODEL}
run2_card=${VBF_RUN2_CARD_PATH}
run2_card_sha256=$(sha256sum "${VBF_RUN2_CARD_PATH}" | awk '{print $1}')
run3_card=${VBF_RUN3_CARD_PATH}
run3_card_sha256=$(sha256sum "${VBF_RUN3_CARD_PATH}" | awk '{print $1}')
stats_year=${STATS_YEAR}
stats_label=${STATS_LABEL}
stats_save_root=${STATS_SAVE_ROOT}
stats_postfix=${STATS_POSTFIX}
jj_eta_region=${STATS_JJ_ETA_REGION}
vbf_filter_study=1
EOF
    fi
}

check_stats_marker() {
    local marker="$1"
    local kind="$2"
    cmp -s <(stats_marker_payload "${kind}") "${marker}" || \
        die "Stats completion marker does not match current inputs: ${marker}"
}

write_stats_marker() {
    local marker="$1"
    local kind="$2"
    mkdir -p "$(dirname "${marker}")"
    stats_marker_payload "${kind}" > "${marker}"
    log "Wrote completion marker: ${marker}"
}

verify_stats_outputs() {
    local stem
    stem="$(stats_stem)"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}.txt"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}.root"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}_prefitsignificance.log"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}_prefitsignificance_StatOnly.log"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}_expectedlimit.log"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}_expectedlimit_StatOnly.log"
    require_nonempty_file "${STATS_CARD_DIR}/vbf_significance_summary_${STATS_POSTFIX}.csv"
    require_nonempty_file "${STATS_CARD_DIR}/vbf_expected_limit_summary_${STATS_POSTFIX}.csv"
}

verify_card_outputs() {
    local stem
    stem="$(stats_stem)"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}.txt"
    require_nonempty_file "${STATS_CARD_DIR}/${stem}.root"
}

verify_impacts_outputs() {
    local tag
    for tag in r1 r0; do
        require_nonempty_file "${STATS_CARD_DIR}/impacts_${STATS_YEAR}_${STATS_POSTFIX}_${tag}.json"
        require_nonempty_file "${STATS_CARD_DIR}/impacts_${STATS_YEAR}_${STATS_POSTFIX}_${tag}.pdf"
    done
}

run_stats_command() {
    local mode="$1"
    local -a environment=(
        env
        "WITH_VARIATIONS=0"
        "JJ_ETA_REGION=${STATS_JJ_ETA_REGION}"
        # _SigXS is already part of STATS_POSTFIX when required. Keep this at
        # zero so future common-workflow plumbing cannot append it twice.
        "SIGNAL_XSEC_RESCALE=0"
    )
    if [[ "${STATS_CONTEXT}" == "combine" ]]; then
        environment+=(
            "VBF_RUN2_CARD_PATH=${VBF_RUN2_CARD_PATH}"
            "VBF_RUN3_CARD_PATH=${VBF_RUN3_CARD_PATH}"
            "VBF_RUN2_MODEL_LABEL=${COMBINE_RUN2_MODEL}"
            "VBF_RUN3_MODEL_LABEL=${COMBINE_RUN3_MODEL}"
        )
    fi
    local -a command=(
        "${RUN_IN_PIXI}" combine bash "${STATS_PIPELINE}"
        -m "${mode}"
        -y "${STATS_YEAR}"
        -v "${STATS_NANOAOD_VERSION}"
        -c "${STATS_DATASET_CONFIG}"
        -l "${STATS_LABEL}"
        -o "${STATS_POSTFIX}"
        -S "${STATS_SAVE_ROOT}"
        -V
    )
    run_cmd "${environment[@]}" "${command[@]}"
}

run_stats_results() {
    mkdir -p "${STATS_CARD_DIR}"
    local marker="${STATS_CARD_DIR}/.vbfdnn_stats_${STATS_YEAR}.complete"
    if [[ -f "${marker}" ]]; then
        check_stats_marker "${marker}" stats
        verify_stats_outputs
        [[ -z "$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f -name 'datacard_*.txt' -newer "${marker}" -print -quit)" ]] || \
            die "Stage-3 datacards changed after stats completed."
        log "Reusing validated stats output: ${STATS_CARD_DIR}"
        return
    fi
    local partial=""
    partial="$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f \
        \( -name '*_prefitsignificance*.log' -o -name '*_expectedlimit*.log' \
           -o -name 'vbf_significance_summary_*.csv' \
           -o -name 'vbf_expected_limit_summary_*.csv' \) -print -quit 2>/dev/null || true)"
    [[ -z "${partial}" ]] || die \
        "Unmarked/partial stats output exists (${partial}); refusing to overwrite it."
    run_stats_command "${STATS_MODE_FOR_RESULTS}"
    verify_stats_outputs
    write_stats_marker "${marker}" stats
}

run_card_results() {
    mkdir -p "${STATS_CARD_DIR}"
    local marker="${STATS_CARD_DIR}/.vbfdnn_cards_${STATS_YEAR}.complete"
    if [[ -f "${marker}" ]]; then
        check_stats_marker "${marker}" cards
        verify_card_outputs
        [[ -z "$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f -name 'datacard_*.txt' -newer "${marker}" -print -quit)" ]] || \
            die "Stage-3 datacards changed after the total card was built."
        log "Reusing validated total card/workspace: ${STATS_CARD_DIR}"
        return
    fi
    local stem partial=""
    stem="$(stats_stem)"
    partial="$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f \
        \( -name "${stem}.txt" -o -name "${stem}.root" \) \
        -print -quit 2>/dev/null || true)"
    [[ -z "${partial}" ]] || die \
        "Unmarked/partial total-card output exists (${partial}); refusing to overwrite it."
    run_stats_command 4
    verify_card_outputs
    write_stats_marker "${marker}" cards
}

run_impacts_results() {
    mkdir -p "${STATS_CARD_DIR}"
    local marker="${STATS_CARD_DIR}/.vbfdnn_impacts_${STATS_YEAR}.complete"
    if [[ -f "${marker}" ]]; then
        check_stats_marker "${marker}" impacts
        verify_impacts_outputs
        log "Reusing validated impacts output: ${STATS_CARD_DIR}"
        return
    fi
    local partial=""
    partial="$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f \
        \( -name "*impacts_${STATS_YEAR}_${STATS_POSTFIX}*" \
           -o -name "impacts_${STATS_YEAR}_${STATS_POSTFIX}_*.json" \
           -o -name "impacts_${STATS_YEAR}_${STATS_POSTFIX}_*.pdf" \) \
        -print -quit 2>/dev/null || true)"
    [[ -z "${partial}" ]] || die \
        "Unmarked/partial impacts output exists (${partial}); refusing to overwrite it."
    run_stats_command 7
    verify_impacts_outputs
    write_stats_marker "${marker}" impacts
}

run_single_stats() {
    verify_stage3_outputs
    configure_single_stats_context
    run_stats_results
}

run_single_cards() {
    verify_stage3_outputs
    configure_single_stats_context
    run_card_results
}

run_single_impacts() {
    verify_stage3_outputs
    configure_single_stats_context
    run_impacts_results
}

run_single_full_stats() {
    verify_stage3_outputs
    configure_single_stats_context
    run_full_stats_results
}

card_suffix_for() {
    local postfix="$1"
    local signal_rescale="$2"
    local jj_region="$3"
    local suffix="${postfix}"
    [[ "${signal_rescale}" == "0" ]] || suffix+="_SigXS"
    suffix+="_vbf_filter_study"
    [[ "${jj_region}" == "all" ]] || suffix+="_${jj_region}"
    printf '%s' "${suffix}"
}

configure_combine() {
    COMBINE_RUN2_MODEL="$1"
    COMBINE_RUN3_MODEL="$2"
    validate_token "RUN2_MODEL" "${COMBINE_RUN2_MODEL}"
    validate_token "RUN3_MODEL" "${COMBINE_RUN3_MODEL}"

    local base_final_postfix="${FINAL_POSTFIX:-TransformerStage23}"
    local base_jj_region="${JJ_ETA_REGION:-all}"
    local run2_root run3_root run2_label run3_label run2_postfix run3_postfix
    local run2_jj run3_jj run3_sig run2_suffix run3_suffix
    run2_root="$(realpath -m "${RUN2_SAVE_ROOT:-${SAVE_ROOT:-${DEFAULT_SAVE_ROOT}}}")"
    run3_root="$(realpath -m "${RUN3_SAVE_ROOT:-${SAVE_ROOT:-${DEFAULT_SAVE_ROOT}}}")"
    run2_label="${RUN2_ANALYSIS_LABEL:-${COMBINE_RUN2_MODEL}}"
    run3_label="${RUN3_ANALYSIS_LABEL:-${COMBINE_RUN3_MODEL}}"
    run2_postfix="${RUN2_FINAL_POSTFIX:-${base_final_postfix}}"
    run3_postfix="${RUN3_FINAL_POSTFIX:-${base_final_postfix}}"
    run2_jj="${RUN2_JJ_ETA_REGION:-${base_jj_region}}"
    run3_jj="${RUN3_JJ_ETA_REGION:-${base_jj_region}}"
    run3_sig="${RUN3_SIGNAL_XSEC_RESCALE:-${SIGNAL_XSEC_RESCALE:-0}}"
    validate_token "RUN2_ANALYSIS_LABEL" "${run2_label}"
    validate_token "RUN3_ANALYSIS_LABEL" "${run3_label}"
    validate_token "RUN2_FINAL_POSTFIX" "${run2_postfix}"
    validate_token "RUN3_FINAL_POSTFIX" "${run3_postfix}"
    validate_token "RUN2_JJ_ETA_REGION" "${run2_jj}"
    validate_token "RUN3_JJ_ETA_REGION" "${run3_jj}"
    validate_bool "RUN3_SIGNAL_XSEC_RESCALE" "${run3_sig}"
    [[ "${run2_jj}" == "${run3_jj}" ]] || die \
        "Run-2 and Run-3 cards must use the same JJ eta region (${run2_jj} != ${run3_jj})."

    run2_suffix="$(card_suffix_for "${run2_postfix}" 0 "${run2_jj}")"
    run3_suffix="$(card_suffix_for "${run3_postfix}" "${run3_sig}" "${run3_jj}")"
    VBF_RUN2_CARD_PATH="$(realpath -m "${VBF_RUN2_CARD_PATH:-${run2_root}/${run2_label}/stage3_datacards_${run2_suffix}/score_${run2_label}/HMuMu_13TeV_Run2.txt}")"
    VBF_RUN3_CARD_PATH="$(realpath -m "${VBF_RUN3_CARD_PATH:-${run3_root}/${run3_label}/stage3_datacards_${run3_suffix}/score_${run3_label}/HMuMu_13TeV_Run3.txt}")"
    require_nonempty_file "${VBF_RUN2_CARD_PATH}"
    require_nonempty_file "${VBF_RUN3_CARD_PATH}"

    STATS_CONTEXT="combine"
    STATS_YEAR="Run2Run3"
    STATS_LABEL="${COMBINED_ANALYSIS_LABEL:-Run2Run3_${COMBINE_RUN2_MODEL}__${COMBINE_RUN3_MODEL}}"
    STATS_POSTFIX="${COMBINED_FINAL_POSTFIX:-Run2Run3}"
    STATS_SAVE_ROOT="$(realpath -m "${COMBINED_SAVE_ROOT:-${SAVE_ROOT:-${DEFAULT_SAVE_ROOT}}}")"
    STATS_JJ_ETA_REGION="${COMBINED_JJ_ETA_REGION:-${run2_jj}}"
    STATS_DATASET_CONFIG="$(realpath -m "${COMBINED_DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/dataset_nanoAODv15_run3.yaml}")"
    STATS_NANOAOD_VERSION="15"
    # Keep the public step meanings separate: `stats` is mode 11, `impacts`
    # is mode 7, while a clean `all` invocation uses the new full mode 15.
    STATS_MODE_FOR_RESULTS="11"
    validate_token "COMBINED_ANALYSIS_LABEL" "${STATS_LABEL}"
    validate_token "COMBINED_FINAL_POSTFIX" "${STATS_POSTFIX}"
    validate_token "COMBINED_JJ_ETA_REGION" "${STATS_JJ_ETA_REGION}"
    [[ "${STATS_JJ_ETA_REGION}" == "${run2_jj}" ]] || die \
        "COMBINED_JJ_ETA_REGION must match the input cards."
    [[ -f "${STATS_DATASET_CONFIG}" ]] || die \
        "Combined stats dataset config is missing: ${STATS_DATASET_CONFIG}"
    local combined_suffix
    combined_suffix="$(card_suffix_for "${STATS_POSTFIX}" 0 "${STATS_JJ_ETA_REGION}")"
    STATS_CARD_DIR="${STATS_SAVE_ROOT}/${STATS_LABEL}/stage3_datacards_${combined_suffix}/score_${STATS_LABEL}"

    log "Cross-Run inputs:"
    log "  Run 2 model: ${COMBINE_RUN2_MODEL}"
    log "  Run 2 card:  ${VBF_RUN2_CARD_PATH}"
    log "  Run 3 model: ${COMBINE_RUN3_MODEL}"
    log "  Run 3 card:  ${VBF_RUN3_CARD_PATH}"
    log "  output:      ${STATS_CARD_DIR}"
}

run_full_stats_results() {
    local stats_marker="${STATS_CARD_DIR}/.vbfdnn_stats_${STATS_YEAR}.complete"
    local impacts_marker="${STATS_CARD_DIR}/.vbfdnn_impacts_${STATS_YEAR}.complete"

    # When either half was already run deliberately, preserve its marker and
    # execute only the missing half through the narrower legacy mode.
    if [[ -f "${stats_marker}" || -f "${impacts_marker}" ]]; then
        run_stats_results
        run_impacts_results
        return
    fi

    mkdir -p "${STATS_CARD_DIR}"
    local partial=""
    partial="$(find "${STATS_CARD_DIR}" -maxdepth 1 -type f \
        \( -name '*_prefitsignificance*.log' -o -name '*_expectedlimit*.log' \
           -o -name 'vbf_significance_summary_*.csv' \
           -o -name 'vbf_expected_limit_summary_*.csv' \
           -o -name "*impacts_${STATS_YEAR}_${STATS_POSTFIX}*" \
           -o -name "impacts_${STATS_YEAR}_${STATS_POSTFIX}_*.json" \
           -o -name "impacts_${STATS_YEAR}_${STATS_POSTFIX}_*.pdf" \) \
        -print -quit 2>/dev/null || true)"
    [[ -z "${partial}" ]] || die \
        "Unmarked/partial full-stats output exists (${partial}); refusing to overwrite it."

    run_stats_command 15
    verify_stats_outputs
    verify_impacts_outputs
    write_stats_marker "${stats_marker}" stats
    write_stats_marker "${impacts_marker}" impacts
}

main_single() {
    [[ "$#" -eq 3 ]] || { usage >&2; exit 2; }
    local run_kind="$1"
    local model_label="$2"
    local step="$3"
    case "${step}" in
        validate|dump|scan|stage23|cards|stats|impacts|all) ;;
        *) die "Invalid single-model step '${step}'." ;;
    esac
    configure_single "${run_kind}" "${model_label}"
    validate_single_contract
    log "Nominal-only ${run_kind} workflow for ${MODEL_LABEL}"
    log "  output: ${ANALYSIS_PATH}"
    log "  sideband dimuon mass: ${BINARY_TRANSFORMER_SIDEBAND_MASS}"
    log "  Dask Gateway cluster index: ${DASK_CLUSTER_INDEX}"
    if [[ -n "${V12_YEARS_CSV}" ]]; then
        log "  NanoAODv${NANOAOD_VERSION_V12}: ${V12_YEARS_CSV} (${DATASET_CONFIG_V12})"
    fi
    log "  NanoAODv${NANOAOD_VERSION_V15}: ${V15_YEARS_CSV} (${DATASET_CONFIG_V15})"

    case "${step}" in
        validate) ;;
        dump) run_dump ;;
        scan) run_scan ;;
        stage23) run_stage23 ;;
        cards) run_single_cards ;;
        stats) run_single_stats ;;
        impacts) run_single_impacts ;;
        all)
            run_dump
            run_scan
            run_stage23
            run_single_full_stats
            ;;
    esac
}

main_combine() {
    [[ "$#" -eq 4 ]] || { usage >&2; exit 2; }
    local run2_model="$2"
    local run3_model="$3"
    local step="$4"
    case "${step}" in
        cards|stats|impacts|all) ;;
        *) die "Combine step must be cards, stats, impacts, or all; got '${step}'." ;;
    esac
    configure_combine "${run2_model}" "${run3_model}"
    case "${step}" in
        cards) run_card_results ;;
        stats) run_stats_results ;;
        impacts) run_impacts_results ;;
        all) run_full_stats_results ;;
    esac
}

if [[ "${1:-}" == "combine" ]]; then
    main_combine "$@"
else
    main_single "$@"
fi
