#!/usr/bin/env bash
set -euo pipefail

# Non-interactive sibling of enter_pixi.sh.
#
# enter_pixi.sh always ends with `exec bash`, so it's only useful from an
# interactive terminal - handed a command to run non-interactively (e.g. from
# an automated/background invocation), it just hangs waiting on stdin. This
# script does the same env setup (proxy reuse/creation, cmsset_default,
# PYTHONPATH) and then runs the given command, exiting with its status.
#
# Usage:
#   ./run_in_pixi.sh <env> <command> [args...]
#
# Examples:
#   ./run_in_pixi.sh default python3 src/copperhead/zpt_rewgt/derive/get_polyFit.py -l ... -y 2024 --njet 1 --save_postfix Sep14_2026
#   ./run_in_pixi.sh default bash scripts/zpt_loop.sh -m zpt_fit12 -y "2024 2025" -n "0 1 2" -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l my_label
#   ./run_in_pixi.sh combine bash run_stats_pipeline_VBF.sh -m 9 -y 2024 -l my_label

WORKDIR="$(pwd)"
PIXI_PROJECT="/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2"
PIXI_ENV="${1:-default}"
shift || true

if [[ $# -eq 0 ]]; then
    echo "[ERROR] Usage: $0 <env> <command> [args...]" >&2
    exit 1
fi

case "$PIXI_ENV" in
    combine|Combine) PIXI_ENV="combine" ;;
    default|Default) PIXI_ENV="default" ;;
    ci|CI) PIXI_ENV="ci" ;;
    *)
        echo "[ERROR] Unknown Pixi environment: $PIXI_ENV" >&2
        echo "Allowed: combine, default, ci" >&2
        exit 1
        ;;
esac

if [[ ! -d "$PIXI_PROJECT" ]]; then
    echo "[ERROR] Pixi project not found: $PIXI_PROJECT" >&2
    exit 1
fi

export WORKDIR

cd "$PIXI_PROJECT"
exec pixi run -e "$PIXI_ENV" bash -c '
set -euo pipefail
cd "$WORKDIR"

export X509_USER_PROXY="${WORKDIR}/voms_proxy.txt"
if [[ -f "$X509_USER_PROXY" ]] && voms-proxy-info -file "$X509_USER_PROXY" -exists -valid 12:00 >/dev/null 2>&1; then
    :
else
    echo "Setting up the proxy..." >&2
    voms-proxy-init -voms cms -rfc -valid 192:00 --out "$X509_USER_PROXY"
fi

export XRD_REQUESTTIMEOUT=300
if [[ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]]; then
    source /cvmfs/cms.cern.ch/cmsset_default.sh
fi
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

exec "$@"
' bash "$@"
