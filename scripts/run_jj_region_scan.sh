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
#   bash scripts/run_jj_region_scan.sh [-l labels] [-r regions] [-o postfix] [-s steps] [-Y years] [-n]
#     -l  comma-separated run labels (default: the LumiSplit and v12 Systematics labels below)
#     -r  comma-separated, mutually exclusive jj regions
#         (default: jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25)
#     -o  save postfix for the stage-2/3 outputs (default: Oct03_2026)
#     -s  comma-separated steps out of stage2,stage3,validate,stats,combine,impacts,lhscan
#         (default: stage2,stage3,validate,stats,combine); use this to resume after a failure
#         without redoing finished steps. validate always runs after stage3 and before stats,
#         even when not listed, so no significance is computed from unchecked templates.
#     -Y  comma-separated years/pseudo-years for impacts + lhscan (default: Run3)
#     -n  dry run: print the commands without running them
# Env overrides: MODEL_LABEL, MODEL_YEARS, WITH_VARIATIONS (default 0 = _NoSyst histograms).
#
# Example (full chain for both labels, the 3-way split):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026
# Example (only impacts + likelihood scans, also per 2024):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026 -s impacts,lhscan -Y 2024,Run3
# Example (only redo the combination for one label):
#   bash scripts/run_jj_region_scan.sh -o Oct03_2026 -s combine \
#       -l Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_LumiSplit
set -euo pipefail

#labels="Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_LumiSplit,Run3_nanoAODv12_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics"
labels="Run3_nanoAODv15_FilterEvents_Sep22_tightPassLepVeto_OfficialRecomendation_Systematics_LumiSplit"
regions="jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25"
postfix="Oct04_2026"
#steps="stage2,stage3,validate,stats,combine,impacts,lhscan"
steps="stage2,stage3,validate,stats,combine"
scan_years="Run3"
dry_run=0

while getopts ":l:r:o:s:Y:nh" opt; do
    case "${opt}" in
        l) labels="${OPTARG}" ;;
        r) regions="${OPTARG}" ;;
        o) postfix="${OPTARG}" ;;
        s) steps="${OPTARG}" ;;
        Y) scan_years="${OPTARG}" ;;
        n) dry_run=1 ;;
        h) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "Unknown option -${OPTARG}" >&2; exit 1 ;;
    esac
done

export MODEL_LABEL="${MODEL_LABEL:-Run3_nanoAODv12_FilterEvents_Aug30_tightPassLepVeto_OfficialRecomendation_Systematics}"
export MODEL_YEARS="${MODEL_YEARS:-2022preEE,2022postEE,2023,2023BPix,2024,2025,2026}"
export WITH_VARIATIONS="${WITH_VARIATIONS:-1}"

stage2_years="2022preEE,2022postEE,2023,2023BPix,2024,2025,2026"
# 2025_2026 is the merged stage-3 year (PC guideline); Run3 combines it with 2022-2024.
stage3_years="${stage2_years},2025_2026"
stats_years="${stage3_years},Run3"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

IFS=',' read -r -a label_list <<< "${labels}"
IFS=',' read -r -a region_list <<< "${regions}"

if [[ " ${region_list[*]} " == *" jj_non_central "* && ${#region_list[@]} -gt 2 ]]; then
    echo "[ERROR] jj_non_central overlaps the forward regions; combining it would double-count events." >&2
    exit 1
fi
for r in "${region_list[@]}"; do
    model_dir="dnn/trained_models/${MODEL_LABEL}/${MODEL_YEARS//,/-}_h-peak_vbf_${r}"
    [[ -d "${model_dir}" ]] || { echo "[ERROR] Missing DNN model for ${r}: ${model_dir}" >&2; exit 1; }
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
    echo "[$(date '+%F %T')] JJ_ETA_REGION=${JJ_ETA_REGION:-} JJ_COMBINE_REGIONS=${JJ_COMBINE_REGIONS:-} $*" | tee -a "${log_dir}/driver.log"
    [[ "${dry_run}" == "1" ]] && return 0
    "$@" 2>&1 | tee "${log_dir}/${log_file}"
}

for label in "${label_list[@]}"; do
    for region in "${region_list[@]}"; do
        tag="${label}__${region}"
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
            tmpl_dir="${save_root}/${label}/stage3_datacards_${postfix}_${region}/stage3_templates_${postfix}_${region}/score_${label}"
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
    if has_step combine; then
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
    if has_step impacts; then
        JJ_COMBINE_REGIONS="${regions}" run "impacts_${label}__combined.log" ./run_in_pixi.sh combine \
            bash run_stats_pipeline_VBF.sh -m 13 -y "${scan_years}" -l "${label}" -o "${postfix}"
    fi
    if has_step lhscan; then
        JJ_COMBINE_REGIONS="${regions}" run "lhscan_${label}__combined.log" ./run_in_pixi.sh combine \
            bash run_stats_pipeline_VBF.sh -m 14 -y "${scan_years}" -l "${label}" -o "${postfix}"
    fi
done

[[ "${dry_run}" == "1" ]] && exit 0

# Plots: <card dir>/impacts_<year>_<postfix>_r{1,0}.pdf and lh_scan_<year>_<postfix>.pdf

# Same naming as common_workflow.sh's jj_combined_name().
combined_name="jj_combined"
for r in "${region_list[@]}"; do combined_name+="_${r#jj_}"; done
for label in "${label_list[@]}"; do
    echo "===== ${label}"
    for region in "${region_list[@]}" "${combined_name}"; do
        d="${save_root}/${label}/stage3_datacards_${postfix}_${region}/score_${label}"
        for csv in "${d}/vbf_significance_summary_${postfix}.csv" "${d}/vbf_expected_limit_summary_${postfix}.csv"; do
            if [[ -f "${csv}" ]]; then
                echo "--- ${region}: $(basename "${csv}")"
                cat "${csv}"
            fi
        done
    done
done
