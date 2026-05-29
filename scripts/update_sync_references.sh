#!/bin/bash
set -euo pipefail

# """
# Refresh stage-1 sync reference txt files under test/reference/.

# This script follows the same basic flow as .github/workflows/sync-stage1.yml:
# 1. run the sync stage-1 sample for one or more years
# 2. dump sync text files for data, DY, and VBF
# 3. copy the dumped text files into test/reference/

# Example:
#     bash scripts/update_sync_references.sh "2017,2022preEE"
# """


years_csv="${1:-2017,2022preEE}"
dataset_yaml="configs/datasets/sync_dataset_nanoAODv12.yaml"
nanoaodv="12"
label="label_output"
output_root="test/output"
reference_dir="test/reference"

IFS=',' read -r -a years <<< "$years_csv"

rm -rf "${output_root:?}/${label}"
mkdir -p "$reference_dir"

for year in "${years[@]}"; do
    echo "Refreshing sync references for ${year}"

    bash stage1_loop_Improved.sh \
        -c "$dataset_yaml" \
        -v "$nanoaodv" \
        -l "$label" \
        -y "$year" \
        -m 1 \
        -z \
        -S "$output_root"

    year_root="${output_root}/${label}/stage1_output/${year}"
    f1_root="${year_root}/f1_0"

    python scripts/sync_parquet_dimuon.py \
        "${f1_root}"/data_*/ \
        -o "${year_root}/${year}_data_eventKinematics.txt"

    python scripts/sync_parquet_dimuon.py \
        "${f1_root}"/dy*/0/ \
        -o "${year_root}/${year}_dy_eventKinematics.txt"

    python scripts/sync_parquet_dimuon.py \
        "${f1_root}/vbf_powheg_dipole/0/" \
        -o "${year_root}/${year}_vbf_eventKinematics.txt"

    cp "${year_root}/${year}_data_eventKinematics.txt" "$reference_dir/"
    cp "${year_root}/${year}_dy_eventKinematics.txt" "$reference_dir/"
    cp "${year_root}/${year}_vbf_eventKinematics.txt" "$reference_dir/"
done
