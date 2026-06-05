#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_variant() {
  local label="$1"
  shift
  echo "[run_stage3_core_pdf_variants] Running variant: ${label}"
  python "${SCRIPT_DIR}/run_stage3.py" "$@"
}

COMMON_ARGS=("$@")

run_variant "all_core_pdfs" "${COMMON_ARGS[@]}"
run_variant "no_sumExp" "${COMMON_ARGS[@]}" --exclude-core-pdfs sumExp
run_variant "no_BWZRedux" "${COMMON_ARGS[@]}" --exclude-core-pdfs BWZRedux
run_variant "no_FEWZxBern" "${COMMON_ARGS[@]}" --exclude-core-pdfs FEWZxBern
