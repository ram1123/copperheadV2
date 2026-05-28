#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: common_workflow.sh [options]

Modes:
  0|prestage
  1|stage1
  2|stage2
  2p|stage2_plot
  3|stage3
  all
  zpt_fit|zpt_fit0|zpt_fit1|zpt_fit2|zpt_fit12
  calib|calib_closure
  compact
  dnn|dnn_pre|dnn_train|dnn_var_rank

Options:
  -D    Add DNN score during the compact step. Default is off.
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

dnn_invoked="0"
for year in "${years[@]}"; do
    log "Processing year: ${year}"
    log "  Data: $(data_streams_for_year "${year}")"
    log "  Background: ${bkg_groups}"
    log "  Signal: ${sig_groups}"
    log "  Save path: ${save_path}"

    case "${mode}" in
        0|prestage)
            run_mode_from_nul < <(build_prestage_cmd "${year}" "$(data_streams_for_year "${year}")")
            ;;
        1|stage1)
            run_mode_from_nul < <(build_stage1_cmd "${year}")
            ;;
        1a|compact)
            run_mode_from_nul < <(build_compact_cmd "${year}")
            ;;            
        2|stage2)
            run_mode_from_nul < <(build_stage2_cmd "${year}")
            ;;
        2p|stage2_plot)
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-sidebands")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-peak")
            ;;
        3|stage3)
            run_mode_from_nul < <(build_stage3_cmd "${year}")
            ;;
        all)
            # run_mode_from_nul < <(build_stage1_cmd "${year}")
            # run_mode_from_nul < <(build_compact_cmd "${year}")
            run_mode_from_nul < <(build_stage2_cmd "${year}")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-sidebands")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-peak")
            run_mode_from_nul < <(build_stage3_cmd "${year}")
            ;;
        zpt_fit|zpt_fit0|zpt_fit1|zpt_fit2|zpt_fit12)
            run_zpt_fit "${year}"
            ;;
        calib)
            run_mode_from_nul < <(build_calib_cmd "${year}" "full")
            ;;
        calib_closure)
            run_mode_from_nul < <(build_calib_cmd "${year}" "closure")
            ;;
        dnn|dnn_pre|dnn_train|dnn_var_rank)
            if [[ "${dnn_invoked}" == "1" ]]; then
                log "DNN workflow already launched for years=${dnn_years_csv}; skipping duplicate invocation from year ${year}."
                continue
            fi
            dnn_invoked="1"
            run_dnn_workflow_once "${mode}"
            ;;
        *)
            die "Invalid analysis mode '${mode}'."
            ;;
    esac
done

log "Program ended on $(date)"
exec 3>&-
