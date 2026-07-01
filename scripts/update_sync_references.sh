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
#     bash scripts/update_sync_references.sh "2017,2022preEE" --use-reference-switches
# """


years_csv="${1:-2017,2022preEE}"
switch_mode="${2:-}"
dataset_yaml="configs/datasets/sync_dataset_nanoAODv12.yaml"
nanoaodv="12"
label="label_output"
output_root="test/output"
reference_dir="test/reference"
switches_file="configs/parameters/HMuMu/switches.yaml"
switches_backup=""
reference_switches_file="test/reference/switches.yaml"

IFS=',' read -r -a years <<< "$years_csv"

restore_switches() {
    if [[ -n "$switches_backup" && -f "$switches_backup" ]]; then
        mv "$switches_backup" "$switches_file"
    fi
}

if [[ "$switch_mode" == "--use-reference-switches" ]]; then
    switches_backup="$(mktemp "${TMPDIR:-/tmp}/switches.yaml.XXXXXX")"
    cp "$switches_file" "$switches_backup"
    trap restore_switches EXIT
    cp "$reference_switches_file" "$switches_file"
    echo "Using ${reference_switches_file} for this sync refresh run."
elif [[ -n "$switch_mode" ]]; then
    echo "Unsupported option: ${switch_mode}" >&2
    exit 1
fi

rm -rf "${output_root:?}/${label}"
mkdir -p "$reference_dir"

for year in "${years[@]}"; do
    echo "Refreshing sync references for ${year}"

    case "$year" in
        2017)
            data_sample="data_B"
            dy_sample="dy_M-50_aMCatNLO"
            ;;
        2022preEE)
            data_sample="data_D"
            dy_sample="dyTo2L_M-50_incl"
            ;;
        *)
            echo "Unsupported sync reference year for cutflow copy: ${year}" >&2
            exit 1
            ;;
    esac

    vbf_sample="vbf_powheg_dipole"

    bash run_analysis_pipeline.sh \
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
    cp "$(ls "${f1_root}/${data_sample}/0/cutflow_${data_sample}_"*.json | head -1)" \
        "${reference_dir}/${year}_cutflow_${data_sample}_0.json"
    cp "$(ls "${f1_root}/${dy_sample}/0/cutflow_${dy_sample}_"*.json | head -1)" \
        "${reference_dir}/${year}_cutflow_${dy_sample}_0.json"
    cp "$(ls "${f1_root}/${vbf_sample}/0/cutflow_${vbf_sample}_"*.json | head -1)" \
        "${reference_dir}/${year}_cutflow_${vbf_sample}_0.json"
done
