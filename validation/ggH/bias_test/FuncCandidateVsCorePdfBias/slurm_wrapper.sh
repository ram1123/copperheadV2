#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-${SCRIPT_DIR}}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
RUN_MODE="${2:-${BIAS_RUN_MODE:-slurm}}"
MAX_LOCAL_JOBS="${3:-${BIAS_LOCAL_JOBS:-8}}"

# Note on the out index:
# out_index="0" # sumExp
# out_index="1" # BWZRedux
# out_index="2" # FEWZxBern

if [[ "${RUN_MODE}" == "local" ]]; then
    echo "Running bias jobs locally from: ${WORK_DIR}"
    echo "Max concurrent local jobs: ${MAX_LOCAL_JOBS}"
    job_id_base="$(date +%s)"
    running_jobs=0
    task_id=1

    for in_index in {0..7}; do # function candidate has 8 in_index and each are frozen for toy generation
        for out_index in {1..1}; do # set core pdf index to BWZ redux, but it is NOT frozen
            local_job_id="${job_id_base}${task_id}"
            pixi run -e combine --manifest-path /cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2_dev/pixi.toml \
                sh "${SCRIPT_DIR}/slurm_combine_sh.sh" "${task_id}" "${local_job_id}" "${in_index}" "${out_index}" "${WORK_DIR}" &
            running_jobs=$((running_jobs + 1))
            task_id=$((task_id + 1))

            if (( running_jobs >= MAX_LOCAL_JOBS )); then
                wait -n
                running_jobs=$((running_jobs - 1))
            fi
        done
    done

    wait
else
    # individual fit function ----------------------
    for in_index in {0..7}; do # function candidate has 8 in_index and each are frozen for toy generation
        for out_index in {1..1}; do # set core pdf index to BWZ redux, but it is NOT frozen
            sbatch "${SCRIPT_DIR}/slurm_setup.sub" $in_index $out_index "${WORK_DIR}" "${SCRIPT_DIR}"
        done
    done
    # individual fit function ----------------------
fi
