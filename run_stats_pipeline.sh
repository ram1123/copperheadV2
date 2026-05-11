#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_stats_pipeline.sh [options]

Modes:
  4|copy_datacards
  5|combine_vbf
  6|combine_vbf_significance
  7|combine_vbf_impacts
  8|combine_vbf_lhscan
  9|combine_vbf_all
  10|combine_vbf_summary
  11|vbf_limit
EOF
    exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common_workflow.sh"

common_defaults
parse_common_args "$@"
setup_logging
trap 'log "Program FAILED on $(date)"; exec 3>&-' ERR
log "Program started on $(date)"
require_workflow_root
load_year_maps
print_run_configuration

for year in "${years[@]}"; do
    log "Processing year: ${year}"
    case "${mode}" in
        4|copy_datacards)
            copy_stage3_datacards
            ;;
        5|combine_vbf)
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            ;;
        6|combine_vbf_significance)
            run_vbf_significance "${year}"
            ;;
        7|combine_vbf_impacts)
            run_vbf_impacts "${year}"
            ;;
        8|combine_vbf_lhscan)
            run_vbf_lhscan "${year}"
            ;;
        9|combine_vbf_all)
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            run_vbf_significance "${year}"
            collect_vbf_significance_summary
            ;;
        10|combine_vbf_summary)
            collect_vbf_significance_summary
            ;;
        11|vbf_limit)
            run_mode_from_nul < <(build_stage2_cmd "${year}")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-sidebands")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-peak")
            run_mode_from_nul < <(build_stage3_cmd "${year}")
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            run_vbf_significance "${year}"
            collect_vbf_significance_summary
            ;;
        *)
            die "Invalid stats mode '${mode}'."
            ;;
    esac
done

log "Program ended on $(date)"
exec 3>&-
