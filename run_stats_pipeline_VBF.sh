#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_stats_pipeline_VBF.sh [options]

Modes:
  4|combine_vbf              Build the VBF datacard + Combine workspace for -y (a single
                              year, or a pseudo-year like Run3/Run2/Run2Run3 to combine
                              already-built per-year cards into one).
  5|combine_vbf_significance Run signal / stat-only significance fits.
  6|combine_vbf_impacts      Run nuisance-parameter impacts. Blinded, so no observed
                              scenario: runs Asimov r=1 (signal injected) and r=0
                              (background-only), each as its own impacts_..._r1/_r0 plot.
  7|combine_vbf_lhscan       Run a 1D likelihood scan (Asimov, r=1 injected).
  8|combine_vbf_all          combine_vbf + combine_vbf_significance, then collect the
                              significance summary CSV.
  9|combine_vbf_summary      (Re)collect the significance summary CSV only.
  10|vbf_limit                Full chain: stage2 + stage2_plot + stage3 + combine_vbf_all.
                              NOTE: despite the name this runs significance, not
                              AsymptoticLimits — pre-existing naming leftover, kept
                              as-is here. Use mode 11 for an actual expected limit.
  11|combine_vbf_limit        Expected 95% CL limit via AsymptoticLimits --run blind
                              (Asimov background-only dataset, no unblinding), with-syst
                              and stat-only variants; collects
                              vbf_expected_limit_summary_<save_postfix>.csv.

Common options:
  -V    Enable --vbf_filter_study for the VBF stage-2/plot/stage-3 commands built by this wrapper.
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
        4|combine_vbf)
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            ;;
        5|combine_vbf_significance)
            run_vbf_significance "${year}"
            ;;
        6|combine_vbf_summary)
            collect_vbf_significance_summary
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
        10|combine_vbf_limit)
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            run_vbf_limit "${year}"
            collect_vbf_limit_summary
            ;;
        11|vbf_limit)
            ensure_vbf_card "${year}"
            ensure_vbf_workspace "${year}"
            run_vbf_significance "${year}"
            collect_vbf_significance_summary
            run_vbf_limit "${year}"
            collect_vbf_limit_summary
            ;;
        *)
            die "Invalid stats mode '${mode}'."
            ;;
    esac
done

log "Program ended on $(date)"
exec 3>&-
