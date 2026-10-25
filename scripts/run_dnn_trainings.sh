#!/usr/bin/env bash
# Trains VBF DNNs (preprocess + Optuna HPO + final training, run_analysis_pipeline.sh -m dnn) for
# several run labels x jj regions. One stream per label, streams run in parallel (at most -P at a
# time); inside a stream the regions train one after another. Every (label, region) writes
# "START|OK|FAIL <label> <region> y=<years> h=<HPO label>" lines to the status log (-S); a region
# already OK there for the same years and HPO label is skipped, so the script can be re-run after an
# interruption (the region's log is appended, never overwritten). OK requires the files run_stage2_vbf.py loads
# (training_features.pkl, scalers_<f>.npz, trained_best_optuna_<HPO label>/fold<f>/best_torchscript.pt).
# Consumed by scripts/run_scenario_chains.sh, which waits on the same status log.
#
# Gotchas (see the mva skill, references/vbf-dnn.md section 6):
# - Before re-running a region whose HPO was interrupted, move hpo_optuna/<HPO label> aside (never
#   delete): optuna resumes the stored study and would run <trials> MORE trials.
# - Stage-2 needs the same HPO label (HPO_LABEL) to find these models.
# - Parallel streams share one GPU (a T4 saturates at ~4-5 streams); OMP_NUM_THREADS caps the CPU
#   threads per stream (default: nproc / number of parallel streams).
# - Takes hours: launch it detached so it survives the session ending
#   (.claude/skills/run-copperheadV2-develop/detach_job.sh).
#
# How to run (repo root; GPU session; no Dask needed; voms_proxy.txt valid >= 12 h for run_in_pixi.sh):
#   bash scripts/run_dnn_trainings.sh -l <labels csv | @file> -y <years csv> -S <status log>
#       [-r regions csv] [-t trials] [-H hpo label] [-P max parallel streams] [-k] [-n]
#     -r  default jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25,all
#     -t  HPO trials to ADD to the study (default 50; the rule is >= 50 in total per jj region);
#         -H HPO label (default v1_multifold_0<t>Trials, e.g. ..._050Trials)
#     -P  default: one stream per label; -P 1 = fully sequential
#     -k  preprocess on the existing Dask Gateway cluster (run_analysis_pipeline.sh -k; scale it first)
#     -n  dry run: print what would be trained
#   HPO_BACKEND=ray: HPO trials + the 4 final-training folds run in parallel on the AF Ray GPU cluster
#   (MVA_training/VBF_run3/hpo_ray.py; RAY_MAX_WORKERS GPUs, default 15) instead of the local GPU;
#   needs a session without a GPU. Same output files, so the OK check and stage-2 are unchanged.
#
# Example (full Run 3, one region, preprocessing on Dask Gateway, HPO+training on 15 Ray GPUs, detached):
#   HPO_BACKEND=ray EXPECTED_H=6 bash .claude/skills/run-copperheadV2-develop/detach_job.sh dnn_ray_all \
#       bash scripts/run_dnn_trainings.sh -l Run3_nanoAODv15_FilterEvents_Oct23_tightPassLepVeto_OfficialRecomendation_IPAdditionalLepVeto_Systematics_LumiSplit \
#       -y 2022preEE,2022postEE,2023,2023BPix,2024,2025,2026 -S logs/oct23_dnn_ray/dnn_status.log -r all -t 50 -k
#
# Example (5 labels from a file, 2024, detached):
#   EXPECTED_H=12 bash .claude/skills/run-copperheadV2-develop/detach_job.sh dnn2024 \
#       bash scripts/run_dnn_trainings.sh -l @logs/scenario2024_dnn/labels.txt -y 2024 \
#       -S logs/scenario2024/dnn_status.log -t 50
set -uo pipefail

labels_arg=""; years=""; status=""; regions="jj_both_central,jj_one_fwd25_one_central,jj_both_fwd25,all"
trials=50; hpo_label=""; max_par=0; dry=0; gateway_args=()
while getopts ":l:y:S:r:t:H:P:knh" opt; do
    case "${opt}" in
        l) labels_arg="${OPTARG}" ;; y) years="${OPTARG}" ;; S) status="${OPTARG}" ;;
        r) regions="${OPTARG}" ;; t) trials="${OPTARG}" ;; H) hpo_label="${OPTARG}" ;;
        P) max_par="${OPTARG}" ;; k) gateway_args=(-k) ;; n) dry=1 ;;
        h) sed -n '2,/^set -uo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "Unknown option -${OPTARG}" >&2; exit 1 ;;
    esac
done
[[ -n "${labels_arg}" && -n "${years}" && -n "${status}" ]] || { echo "[ERROR] -l, -y and -S are required (-h for help)" >&2; exit 1; }
[[ "${trials}" =~ ^[0-9]+$ && "${trials}" -gt 0 && "${max_par}" =~ ^[0-9]+$ ]] || { echo "[ERROR] -t and -P must be whole numbers" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Labels: trimmed (CRLF/trailing spaces), blank lines dropped, duplicates removed (two streams on one
# label would train the same model dirs concurrently).
if [[ "${labels_arg}" == @* ]]; then
    [[ -r "${labels_arg#@}" ]] || { echo "[ERROR] cannot read ${labels_arg#@}" >&2; exit 1; }
    mapfile -t labels < <(sed 's/\r$//; s/^[[:space:]]*//; s/[[:space:]]*$//' "${labels_arg#@}" | grep -v '^$' | awk '!seen[$0]++')
else
    mapfile -t labels < <(tr ',' '\n' <<< "${labels_arg}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$' | awk '!seen[$0]++')
fi
IFS=',' read -r -a region_list <<< "${regions}"
[[ ${#labels[@]} -gt 0 ]] || { echo "[ERROR] no labels" >&2; exit 1; }
[[ ${#region_list[@]} -gt 0 ]] || { echo "[ERROR] no regions" >&2; exit 1; }
[[ "${max_par}" -gt 0 ]] || max_par=${#labels[@]}
hpo_label="${hpo_label:-v1_multifold_$(printf '%03d' "${trials}")Trials}"
mkdir -p "$(dirname "${status}")"
log_dir="$(dirname "${status}")"
years_slug="${years//,/-}"

unset MODEL_YEARS MODEL_LABEL DNN_MODEL_REGION TRAIN_LABEL JJ_COMBINE_REGIONS DUMP_SCORES ALLOW_DEFAULT_DNN_BINNING
export HPO_TRIALS="${trials}" HPO_LABEL="${hpo_label}"
ncpu="$(nproc)"; per=$(( ncpu / max_par )); [[ "${per}" -ge 1 ]] || per=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${per}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

stream() {  # $1 label: train its regions one after another
    local label="$1" r t0 s d f
    for r in "${region_list[@]}"; do
        grep -qF "] OK ${label} ${r} y=${years} h=${hpo_label} " "${status}" 2>/dev/null && continue
        if [[ "${dry}" == "1" ]]; then echo "would train ${label} ${r} (${years}, ${trials} trials, ${hpo_label})"; continue; fi
        t0=$(date +%s)
        echo "[$(date -u '+%F %T')] START ${label} ${r} y=${years} h=${hpo_label}" >> "${status}"
        echo "===== $(date -u '+%F %T') START y=${years} h=${hpo_label}" >> "${log_dir}/dnn_${label}__${r}.log"
        s=FAIL
        if JJ_ETA_REGION="${r}" ./run_in_pixi.sh default bash run_analysis_pipeline.sh -m dnn -y "${years}" -l "${label}" "${gateway_args[@]}" \
                >> "${log_dir}/dnn_${label}__${r}.log" 2>&1 < /dev/null; then
            d="dnn/trained_models/${label}/${years_slug}_h-peak_vbf_${r}"
            s=OK
            [[ -f "${d}/training_features.pkl" ]] || s=FAIL
            for f in 0 1 2 3; do
                [[ -f "${d}/trained_best_optuna_${hpo_label}/fold${f}/best_torchscript.pt" && -f "${d}/scalers_${f}.npz" ]] || s=FAIL
            done
        fi
        echo "[$(date -u '+%F %T')] ${s} ${label} ${r} y=${years} h=${hpo_label} wall=$(( $(date +%s) - t0 ))s" >> "${status}"
    done
}

for label in "${labels[@]}"; do
    if [[ "${dry}" == "1" ]]; then stream "${label}"; continue; fi
    while [[ $(jobs -rp | wc -l) -ge ${max_par} ]]; do wait -n; done
    stream "${label}" &
    sleep 5  # stagger: run_analysis_pipeline.sh log names have 1 s resolution and setup_logging moves others' logs
done
wait
[[ "${dry}" == "1" ]] || echo "[$(date -u '+%F %T')] ALLDONE" >> "${status}"
