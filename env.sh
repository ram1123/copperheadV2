#!/usr/bin/env bash

setup_proxy() {
  echo "Setting up the proxy..."
  voms-proxy-init -voms cms -rfc -valid 192:00 --out "$(pwd)/voms_proxy.txt"
  echo "Your proxy is here: $(pwd)/voms_proxy.txt"
  export X509_USER_PROXY="$(pwd)/voms_proxy.txt"
}

usage() {
  cat <<EOF
Usage:
  source setup_env.sh [--no-proxy]

Options:
  --no-proxy   -> skip voms-proxy-init
EOF
}

# -------------------------
# Main
# -------------------------
PROXY_FLAG="${1:-}"

# -------------------------
# Proxy (default ON)
# -------------------------
if [[ "${PROXY_FLAG}" != "--no-proxy" ]]; then
  setup_proxy
else
  echo "Skipping proxy setup (--no-proxy)"
fi

# -------------------------
# CMS defaults
# -------------------------
export WORKDIR=$PWD
export XRD_REQUESTTIMEOUT=300

# Setup CMSSW related environment
source /cvmfs/cms.cern.ch/cmsset_default.sh

# Load from the current working directory (copperheadV2)
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

echo "$CONDA_DEFAULT_ENV" > .conda_env_name.txt
