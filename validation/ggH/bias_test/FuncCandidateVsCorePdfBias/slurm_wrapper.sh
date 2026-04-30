#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-${SCRIPT_DIR}}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"

# Note on the out index:
# out_index="0" # sumExp
# out_index="1" # BWZRedux
# out_index="2" # FEWZxBern

# individual fit function ----------------------
for in_index in {0..7}; do # function candidate has 8 in_index and each are frozen for toy generation
    for out_index in {1..1}; do # set core pdf index to BWZ redux, but it is NOT frozen
        sbatch "${SCRIPT_DIR}/slurm_setup.sub" $in_index $out_index "${WORK_DIR}" "${SCRIPT_DIR}"
    done
done
# individual fit function ----------------------
