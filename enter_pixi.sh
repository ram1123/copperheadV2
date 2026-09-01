#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./enter_pixi.sh combine
#   ./enter_pixi.sh default
#   ./enter_pixi.sh ci

# ============================================================
# User working repo
# ============================================================
WORKDIR="$(pwd)"
PIXI_PROJECT="/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2"

# Default environment
PIXI_ENV="${1:-default}"

# Normalize common names
case "$PIXI_ENV" in
    combine|Combine)
        PIXI_ENV="combine"
        ;;
    default|Default)
        PIXI_ENV="default"
        ;;
    ci|CI)
        PIXI_ENV="ci"
        ;;
    *)
        echo "[ERROR] Unknown Pixi environment: $PIXI_ENV"
        echo "Allowed: combine, default, ci"
        exit 1
        ;;
esac

echo "WORKDIR      : $WORKDIR"
echo "PIXI_PROJECT : $PIXI_PROJECT"
echo "PIXI_ENV     : $PIXI_ENV"

if [[ ! -d "$PIXI_PROJECT" ]]; then
    echo "[ERROR] Pixi project not found: $PIXI_PROJECT"
    exit 1
fi

# ============================================================
# Enter CVMFS pixi environment, then return to WORKDIR
# ============================================================
cd "$PIXI_PROJECT"

pixi run -e "$PIXI_ENV" bash -lc "
set -euo pipefail

# ------------------------------------------------------------
# Return to analysis working directory
# ------------------------------------------------------------
cd '$WORKDIR'

export WORKDIR='$WORKDIR'

# ------------------------------------------------------------
# Proxy setup
# ------------------------------------------------------------
setup_proxy() {
    echo 'Setting up the proxy...'
    voms-proxy-init -voms cms -rfc -valid 192:00 --out \"\${WORKDIR}/voms_proxy.txt\"
    echo \"Your proxy is here: \${WORKDIR}/voms_proxy.txt\"
    export X509_USER_PROXY=\"\${WORKDIR}/voms_proxy.txt\"
}

# Reuse proxy if valid for at least 12 hours
export X509_USER_PROXY=\"\${WORKDIR}/voms_proxy.txt\"

if [[ -f \"\${X509_USER_PROXY}\" ]] && voms-proxy-info -file \"\${X509_USER_PROXY}\" -exists -valid 12:00; then
    echo \"Valid proxy found: \${X509_USER_PROXY}\"
else
    setup_proxy
fi

# ------------------------------------------------------------
# CMS defaults
# ------------------------------------------------------------
export XRD_REQUESTTIMEOUT=300

if [[ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]]; then
    source /cvmfs/cms.cern.ch/cmsset_default.sh
else
    echo '[WARN] /cvmfs/cms.cern.ch/cmsset_default.sh not found'
fi

# ------------------------------------------------------------
# Python path
# ------------------------------------------------------------
export PYTHONPATH=\"\${WORKDIR}:\${PYTHONPATH:-}\"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo '============================================================'
echo \"Pixi environment ready: $PIXI_ENV\"
echo \"WORKDIR=\${WORKDIR}\"
echo \"PWD=\$PWD\"
echo \"X509_USER_PROXY=\${X509_USER_PROXY}\"
echo \"PYTHONPATH=\${PYTHONPATH}\"
echo \"Python: \$(which python)\"
python -V
echo 'Proxy time left:'
voms-proxy-info -file \"\${X509_USER_PROXY}\" -timeleft || true
echo '============================================================'

exec bash
"
