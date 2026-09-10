#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_analysis_pipeline.sh [options]
Modes:
  0|prestage
  1|stage1
  cutflow_merge                Merges the per-chunk cutflow_*.npz shards from a stage-1
                                -z/--isCutflow run into one whole-dataset cutflow per sample
                                (scripts/merge_cutflow_npz_file.py), writing
                                <save_path>/stage1_output/<year>/f1_0/<sample>/cutflow_merged_<sample>.json.
                                Run after the stage1 -z run it merges, with the same -l/-y/-S.
  2|stage2
  2p|stage2_plot
  3|stage3
  23|stage23
  zpt_fit|zpt_fit0|zpt_fit1|zpt_fit2|zpt_fit12
  calib|calib_closure
  compact
  dnn|dnn_pre|dnn_hpo|dnn_train|dnn_var_rank
  pu_dnn_train                VBF-category DNN above is unrelated to this: pu_dnn_train
                               runs MVA_training/pileup_dnn/train_pu_dnn.py, the per-jet
                               HS-vs-PU classifier consumed inside stage1 via
                               do_use_pu_dnn_score. Runs once per year in -y (unlike
                               dnn_train, which combines years into one invocation).

Options:
  -D    Add DNN score during the compact step. Default is off.
  -V    Enable --vbf_filter_study for the VBF stage-2/plot/stage-3 workflow.

Env vars:
  MODEL_YEARS   Comma-separated years used to build the DNN model directory name
                (dnn/trained_models/<label>/<MODEL_YEARS>_<region>_<category>_<JJ_ETA_REGION>),
                independent of the years passed via -y. Defaults to -y's years.
                Use this to run stage2/stage3 for one year (-y) while loading a
                model trained on a different (e.g. combined) set of years.
  JJ_ETA_REGION dnn/dnn_pre/dnn_train modes only: OVERRIDE for the jet-eta
                topology the VBF DNN's dijet pair is restricted to. The default
                comes from analysis.jj_eta_region in the DNN config YAML
                (${DNN_CONFIG:-configs/dnn_run3_vbf.yaml}) -- edit that key to
                change it persistently; set this env var only for a one-off
                run. "all" (no restriction) or one of modules/selection.py's
                PAIR_JJ_ETA_REGIONS (jj_both_central, jj_non_central,
                jj_one_fwd25_one_central, jj_one_he_one_central,
                jj_one_fwd30_one_central, jj_both_fwd25, jj_both_he,
                jj_both_fwd30, jj_one_he_one_fwd30). The effective value is
                encoded into the DNN output directory name above, so different
                choices don't overwrite each other's output.

  pu_dnn_train mode (all optional, sensible defaults shown):
  PU_DNN_DY_GLOB       Compacted sample-name glob for the HS-jet proxy (default: dyTo2Mu_M-50_aMCatNLO)
  PU_DNN_TTBAR_GLOB    Compacted sample-name glob for a PU-jet proxy (default: ttjets_*)
  PU_DNN_EWK_GLOB      Compacted sample-name glob for a PU-jet proxy (default: ewk_*)
  PU_DNN_REGIONS       Space-separated region list (default: HEpos HEneg HFpos HFneg)
  PU_DNN_OUT_TAG       Output dir name under validation/pu_dnn/ (default: run<year>_dy_top_ewk_<date>)
  PU_DNN_EXTRA_ARGS    Extra args passed through as-is to train_pu_dnn.py (epochs, lr, pt-min, ...)
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
        cutflow_merge)
            run_cutflow_merge "${year}"
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
        23|stage23)
            run_mode_from_nul < <(build_stage2_cmd "${year}")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-sidebands")
            run_mode_from_nul < <(build_stage2_plot_cmd "${year}" "h-peak")
            run_mode_from_nul < <(build_stage3_cmd "${year}")
            ;;
        all)
            run_mode_from_nul < <(build_stage1_cmd "${year}")
            run_mode_from_nul < <(build_compact_cmd "${year}")
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
        dnn|dnn_pre|dnn_hpo|dnn_train|dnn_var_rank)
            if [[ "${dnn_invoked}" == "1" ]]; then
                log "DNN workflow already launched for years=${dnn_years_csv}; skipping duplicate invocation from year ${year}."
                continue
            fi
            dnn_invoked="1"
            run_dnn_workflow_once "${mode}"
            ;;
        pu_dnn_train)
            run_mode_from_nul_timed < <(build_pu_dnn_train_cmd "${year}")
            ;;
        *)
            die "Invalid analysis mode '${mode}'."
            ;;
    esac
done

log "Program ended on $(date)"
exec 3>&-
