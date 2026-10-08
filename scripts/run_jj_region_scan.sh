#!/usr/bin/env bash
# Runs the VBF chain separately in each dijet-eta region, then combines the regions.
# For every run label and every region it runs: stage-2 (default env, Dask Gateway), then
# stage-3, then the mandatory template validation (scripts/plot_template_systematics.py: one
# plot per nuisance + CSV under validation/stage3_templates/<label>/<postfix>_<region>/; it
# stops the chain on non-finite or negative template bins), then per-region significance +
# expected limit (combine env, stats mode 11). It then
# combines the regions into one multi-channel card per year and computes its significance and
# expected limit (stats mode 12 with JJ_COMBINE_REGIONS). Last, it runs blinded impacts
# (Asimov r=1 and r=0; modes 7/13) and a 1D likelihood scan (Asimov r=1; modes 8/14) on each
# region and on the combination, by default only for Run3 because impacts are expensive.
#
# The default regions jj_both_central, jj_one_fwd25_one_central and jj_both_fwd25 are an exact,
# non-overlapping split of the 2-jet VBF selection (|eta| <= 2.5 vs > 2.5). Do not combine
# jj_non_central with them: it is NOT(both central), so it overlaps both forward regions.
# Each region's stage-2 loads its own DNN model, ${MODEL_LABEL}/<MODEL_YEARS>_h-peak_vbf_<region>.
#
# How to run (repo root, from a plain AF terminal -- no pixi shell needed, each step is wrapped
# in ./run_in_pixi.sh). Prerequisites:
#   - stage-1 output exists for every label;
#   - the DNN models above exist for every region;
#   - voms_proxy.txt is valid for >= 12 h, otherwise run_in_pixi.sh tries an interactive
#     voms-proxy-init.
# Steps run one at a time, because only one Dask Gateway cluster per user is allowed.
#   bash scripts/run_jj_region_scan.sh [-l labels] [-r regions] [-o postfix] [-s steps] [-y years] [-Y years] [-n]
#     -l  comma-separated run labels (default: the LumiSplit and v12 Systematics labels below)
#     -r  comma-separated, mutually exclusive jj regions
#         (default: jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25), or just "all"
#         (inclusive VBF phase space: standalone, no combine step; never mixed with split regions,
#         which it overlaps)
#     -o  save postfix for the stage-2/3 outputs (default: Oct03_2026)
#     -s  comma-separated steps out of scoredump,binscan,stage2,stage3,validate,stats,combine,
#         impacts,lhscan (default: stage2,stage3,validate,stats,combine); use this to resume after
#         a failure without redoing finished steps. validate always runs after stage3 and before
#         stats, even when not listed, so no significance is computed from unchecked templates.
#         scoredump: nominal-only stage-2 with DUMP_SCORES=1 and the 'default' binning, under
#         postfix <-o>_ScoreDump; binscan: scan_bins_for_dnn.py on that dump, writing
#         models.<model label>.<region> into configs/MVA/VBF/dnn_binning.yaml (refuses to replace
#         an existing entry unless FORCE_BINSCAN=1). Both must precede stage2 for a new model.
#     -y  comma-separated years for stage-2/3/stats/binscan (default: all Run 3 years, with
#         2025_2026 added for stage-3 and 2025_2026,Run3 for stats); with -y exactly these years
#         are used everywhere, nothing is added
#     -Y  comma-separated years/pseudo-years for impacts + lhscan (default: Run3)
#     -n  dry run: print the commands without running them
# Env overrides: MODEL_LABEL (default: the v12 Aug30 Systematics label; "self" = each run label's
# own models), MODEL_YEARS (default: -y's years if given, else all Run 3 years),
# WITH_VARIATIONS (default 1 = with systematic variations; 0 = nominal-only _NoSyst histograms),
# HPO_LABEL / TRAIN_LABEL (which trained model stage-2 loads: <model dir>/trained_best_optuna_<HPO_LABEL>,
# default v1_multifold_050Trials -- export HPO_LABEL=v1_multifold_025Trials for 25-trial models).
# DNN_MODEL_REGION must be unset (each region loads its own model). scoredump refuses an existing
# dump dir (a partial dump would bias the bin scan); move it aside or use a new -o.
#
# Example (full chain for both labels, the 3-way split):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026
# Example (only impacts + likelihood scans, also per 2024):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026 -s impacts,lhscan -Y 2024,Run3
# Example (only redo the combination for one label):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026 -s combine \
#       -l Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_LumiSplit
# Example (2024 only, each label's own 2024 models, nominal only, new-model binning scan first;
# then the inclusive phase space as a separate, uncombined result):
#   L=Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics
#   MODEL_LABEL=self WITH_VARIATIONS=0 HPO_LABEL=v1_multifold_025Trials bash scripts/run_jj_region_scan.sh \
#       -l $L -y 2024 -Y 2024 -o Oct06_2026_DNN25_NoSyst \
#       -s scoredump,binscan,stage2,stage3,validate,stats,combine,lhscan -n
#   MODEL_LABEL=self WITH_VARIATIONS=0 HPO_LABEL=v1_multifold_025Trials bash scripts/run_jj_region_scan.sh \
#       -l $L -y 2024 -Y 2024 \
#       -o Oct06_2026_DNN25_NoSyst -r all -s scoredump,binscan,stage2,stage3,validate,stats,lhscan -n
set -euo pipefail

#labels="Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_LumiSplit,Run3_nanoAODv12_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics"
labels="Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit"
regions="jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25"
postfix="Oct04_2026"
#steps="stage2,stage3,validate,stats,combine,impacts,lhscan"
steps="stage2,stage3,validate,stats,combine"
scan_years="Run3"
years_override=""
dry_run=0

while getopts ":l:r:o:s:y:Y:nh" opt; do
    case "${opt}" in
        l) labels="${OPTARG}" ;;
        r) regions="${OPTARG}" ;;
        o) postfix="${OPTARG}" ;;
        s) steps="${OPTARG}" ;;
        y) years_override="${OPTARG}" ;;
        Y) scan_years="${OPTARG}" ;;
        n) dry_run=1 ;;
        h) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "Unknown option -${OPTARG}" >&2; exit 1 ;;
    esac
done

export MODEL_LABEL="${MODEL_LABEL:-Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics}"
export WITH_VARIATIONS="${WITH_VARIATIONS:-1}"

if [[ -n "${years_override}" ]]; then
    stage2_years="${years_override}"
    stage3_years="${years_override}"
    stats_years="${years_override}"
else
    stage2_years="2022preEE,2022postEE,2023,2023BPix,2024,2025,2026"
    # 2025_2026 is the merged stage-3 year (PC guideline); Run3 combines it with 2022-2024.
    stage3_years="${stage2_years},2025_2026"
    stats_years="${stage3_years},Run3"
fi
export MODEL_YEARS="${MODEL_YEARS:-${stage2_years}}"
model_label_mode="${MODEL_LABEL}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

IFS=',' read -r -a label_list <<< "${labels}"
IFS=',' read -r -a region_list <<< "${regions}"

if [[ " ${region_list[*]} " == *" jj_non_central "* && ${#region_list[@]} -gt 2 ]]; then
    echo "[ERROR] jj_non_central overlaps the forward regions; combining it would double-count events." >&2
    exit 1
fi
if [[ " ${region_list[*]} " == *" all "* && ${#region_list[@]} -gt 1 ]]; then
    echo "[ERROR] 'all' overlaps every split region; run it alone (-r all), never combined with them." >&2
    exit 1
fi
# Combining needs >= 2 mutually exclusive regions; 'all' is a standalone result.
do_combine=1
[[ ${#region_list[@]} -lt 2 ]] && do_combine=0

model_label_for() {
    if [[ "${model_label_mode}" == "self" ]]; then printf '%s' "$1"; else printf '%s' "${model_label_mode}"; fi
}
# stage-3 card dirs carry no region suffix for the inclusive phase space (stage3_output_postfix)
region_dir_suffix() {
    if [[ "$1" == "all" ]]; then printf ''; else printf '_%s' "$1"; fi
}
if [[ -n "${DNN_MODEL_REGION:-}" ]]; then
    echo "[ERROR] DNN_MODEL_REGION=${DNN_MODEL_REGION} would load one region's model for every region; unset it." >&2
    exit 1
fi
train_label="${TRAIN_LABEL:-trained_best_optuna_${HPO_LABEL:-v1_multifold_050Trials}}"
# Same files run_stage2_vbf.py loads; the model dir alone exists as soon as preprocessing ran.
for label in "${label_list[@]}"; do
    for r in "${region_list[@]}"; do
        model_dir="dnn/trained_models/$(model_label_for "${label}")/${MODEL_YEARS//,/-}_h-peak_vbf_${r}"
        missing=()
        [[ -f "${model_dir}/training_features.pkl" ]] || missing+=("${model_dir}/training_features.pkl")
        for f in 0 1 2 3; do
            [[ -f "${model_dir}/${train_label}/fold${f}/best_torchscript.pt" ]] || missing+=("${model_dir}/${train_label}/fold${f}/best_torchscript.pt")
            [[ -f "${model_dir}/scalers_${f}.npz" ]] || missing+=("${model_dir}/scalers_${f}.npz")
        done
        if [[ ${#missing[@]} -gt 0 ]]; then
            # -n still prints the plan (e.g. before the models are trained); a real run stops.
            echo "[ERROR] Incomplete DNN model for ${r}: ${missing[*]}" >&2
            [[ "${dry_run}" == "1" ]] || exit 1
        fi
    done
done

if [[ "${dry_run}" == "0" ]] && ! voms-proxy-info -file "${repo_root}/voms_proxy.txt" -exists -valid 12:00 >/dev/null 2>&1; then
    echo "[ERROR] ${repo_root}/voms_proxy.txt missing or valid < 12h; renew it interactively first." >&2
    exit 1
fi

save_root="/work/projects/hmm/${USER}/hmm_ntuples/copperheadV1clean"

has_step() { [[ ",${steps}," == *",$1,"* ]]; }

log_dir="logs/jj_region_scan_${postfix}"
mkdir -p "${log_dir}"

run() {
    local log_file="$1"; shift
    echo "[$(date '+%F %T')] JJ_ETA_REGION=${JJ_ETA_REGION:-} JJ_COMBINE_REGIONS=${JJ_COMBINE_REGIONS:-} MODEL_LABEL=${MODEL_LABEL} MODEL_YEARS=${MODEL_YEARS} HPO_LABEL=${HPO_LABEL:-} TRAIN_LABEL=${TRAIN_LABEL:-} WITH_VARIATIONS=${WITH_VARIATIONS} DUMP_SCORES=${DUMP_SCORES:-0} ALLOW_DEFAULT_DNN_BINNING=${ALLOW_DEFAULT_DNN_BINNING:-0} $*" | tee -a "${log_dir}/driver.log"
    [[ "${dry_run}" == "1" ]] && return 0
    "$@" 2>&1 | tee "${log_dir}/${log_file}"
}

for label in "${label_list[@]}"; do
    export MODEL_LABEL="$(model_label_for "${label}")"
    for region in "${region_list[@]}"; do
        tag="${label}__${region}"
        sfx="$(region_dir_suffix "${region}")"
        dump_dir="${save_root}/${label}/stage2_histograms/score_${label}_${postfix}_ScoreDump${sfx}_NoSyst"
        if has_step scoredump; then
            # stage-2 skips samples already done, so a reused partial dump would scan a sample subset.
            if [[ -e "${dump_dir}" && "${dry_run}" == "0" ]]; then
                echo "[ERROR] ${dump_dir} already exists; move it aside or use a new -o." >&2
                exit 1
            fi
            # Nominal-only: the bin scan reads the nominal per-event scores only.
            JJ_ETA_REGION="${region}" WITH_VARIATIONS=0 DUMP_SCORES=1 ALLOW_DEFAULT_DNN_BINNING=1 \
                run "scoredump_${tag}.log" ./run_in_pixi.sh default \
                bash run_analysis_pipeline.sh -m 2 -k -y "${stage2_years}" -l "${label}" -o "${postfix}_ScoreDump"
        fi
        if has_step binscan; then
            # Fail closed: 0 = entry exists, 2 = absent, anything else (no yaml, parse error) aborts.
            rc=0
            python3 -c '
import sys, yaml
cfg = yaml.safe_load(open("configs/MVA/VBF/dnn_binning.yaml")) or {}
sys.exit(0 if ((cfg.get("models") or {}).get(sys.argv[1]) or {}).get(sys.argv[2]) is not None else 2)
' "${MODEL_LABEL}" "${region}" || rc=$?
            if [[ "${rc}" == "0" && "${FORCE_BINSCAN:-0}" != "1" ]]; then
                echo "[ERROR] dnn_binning.yaml already has models/${MODEL_LABEL}/${region}; set FORCE_BINSCAN=1 to replace it." >&2
                exit 1
            elif [[ "${rc}" != "0" && "${rc}" != "2" ]]; then
                echo "[ERROR] could not read configs/MVA/VBF/dnn_binning.yaml (exit ${rc}); not scanning." >&2
                exit 1
            fi
            run "binscan_${tag}.log" ./run_in_pixi.sh default \
                python MVA_training/VBF_run3/scan_bins_for_dnn.py --stage2-scores "${dump_dir}" \
                --years "${stage2_years}" --model-label "${MODEL_LABEL}" --jj-region "${region}" \
                --output-dir "${repo_root}/validation/dnn_binning_scan/$(basename "${dump_dir}")" --write-config
            # The binning is keyed by label+region only, so confirm the entry is from these years.
            [[ "${dry_run}" == "1" ]] || python3 -c '
import sys, yaml
e = yaml.safe_load(open("configs/MVA/VBF/dnn_binning.yaml"))["models"][sys.argv[1]][sys.argv[2]]
got = sorted(str(y) for y in e["metadata"]["years"]); want = sorted(sys.argv[3].split(","))
sys.exit(0 if got == want else f"binning entry years {got} != {want}")
' "${MODEL_LABEL}" "${region}" "${stage2_years}"
        fi
        if has_step stage2; then
            JJ_ETA_REGION="${region}" run "stage2_${tag}.log" ./run_in_pixi.sh default \
                bash run_analysis_pipeline.sh -m 2 -k -y "${stage2_years}" -l "${label}" -o "${postfix}"
        fi
        if has_step stage3; then
            JJ_ETA_REGION="${region}" run "stage3_${tag}.log" ./run_in_pixi.sh default \
                bash run_analysis_pipeline.sh -m 3 -y "${stage3_years}" -l "${label}" -o "${postfix}"
        fi
        # Mandatory gate between stage-3 and the fits; a hard template error stops the chain.
        if has_step validate || has_step stage3 || has_step stats; then
            tmpl_dir="${save_root}/${label}/stage3_datacards_${postfix}${sfx}/stage3_templates_${postfix}${sfx}/score_${label}"
            val_dir="validation/stage3_templates/${label}/${postfix}_${region}"
            run "validate_${tag}.log" ./run_in_pixi.sh combine \
                python scripts/plot_template_systematics.py -i "${tmpl_dir}" -o "${val_dir}/template_systematics"
            run "validate_negbins_${tag}.log" ./run_in_pixi.sh combine \
                python scripts/plot_template_negative_bins.py -i "${tmpl_dir}" \
                -o "${val_dir}/templates_nominal.pdf" -t "${region}"
        fi
        if has_step stats; then
            JJ_ETA_REGION="${region}" run "stats_${tag}.log" ./run_in_pixi.sh combine \
                bash run_stats_pipeline_VBF.sh -m 11 -y "${stats_years}" -l "${label}" -o "${postfix}"
        fi
    done
    if has_step combine && [[ "${do_combine}" == "1" ]]; then
        JJ_COMBINE_REGIONS="${regions}" run "combine_${label}.log" ./run_in_pixi.sh combine \
            bash run_stats_pipeline_VBF.sh -m 12 -y "${stats_years}" -l "${label}" -o "${postfix}"
    fi
    # Needs the per-region / combined cards + workspaces built by the stats / combine steps.
    for region in "${region_list[@]}"; do
        tag="${label}__${region}"
        if has_step impacts; then
            JJ_ETA_REGION="${region}" run "impacts_${tag}.log" ./run_in_pixi.sh combine \
                bash run_stats_pipeline_VBF.sh -m 7 -y "${scan_years}" -l "${label}" -o "${postfix}"
        fi
        if has_step lhscan; then
            JJ_ETA_REGION="${region}" run "lhscan_${tag}.log" ./run_in_pixi.sh combine \
                bash run_stats_pipeline_VBF.sh -m 8 -y "${scan_years}" -l "${label}" -o "${postfix}"
        fi
    done
    if has_step impacts && [[ "${do_combine}" == "1" ]]; then
        JJ_COMBINE_REGIONS="${regions}" run "impacts_${label}__combined.log" ./run_in_pixi.sh combine \
            bash run_stats_pipeline_VBF.sh -m 13 -y "${scan_years}" -l "${label}" -o "${postfix}"
    fi
    if has_step lhscan && [[ "${do_combine}" == "1" ]]; then
        JJ_COMBINE_REGIONS="${regions}" run "lhscan_${label}__combined.log" ./run_in_pixi.sh combine \
            bash run_stats_pipeline_VBF.sh -m 14 -y "${scan_years}" -l "${label}" -o "${postfix}"
    fi
done

[[ "${dry_run}" == "1" ]] && exit 0

# Plots: <card dir>/impacts_<year>_<postfix>_r{1,0}.pdf and lh_scan_<year>_<postfix>.pdf

# Same naming as common_workflow.sh's jj_combined_name().
combined_name="jj_combined"
for r in "${region_list[@]}"; do combined_name+="_${r#jj_}"; done
summary_regions=("${region_list[@]}")
[[ "${do_combine}" == "1" ]] && summary_regions+=("${combined_name}")
for label in "${label_list[@]}"; do
    echo "===== ${label}"
    for region in "${summary_regions[@]}"; do
        d="${save_root}/${label}/stage3_datacards_${postfix}$(region_dir_suffix "${region}")/score_${label}"
        for csv in "${d}/vbf_significance_summary_${postfix}.csv" "${d}/vbf_expected_limit_summary_${postfix}.csv"; do
            if [[ -f "${csv}" ]]; then
                echo "--- ${region}: $(basename "${csv}")"
                cat "${csv}"
            fi
        done
    done
done
