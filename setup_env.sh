#!/usr/bin/env bash

# -------------------------
# Config envs
# -------------------------
ENV_COFFEA="/depot/cms/kernels/coffea_latest"
ENV_YUN="/depot/cms/users/yun79/conda_envs/yun_coffea_latest"
ENV_PFN="/depot/cms/conda_envs/shar1172/pfn_env"
ENV_PY="/depot/cms/kernels/python3"

# -------------------------
# Helpers
# -------------------------
ensure_conda() {
  if ! command -v conda &>/dev/null; then
    echo "Conda could not be found, loading the conda module"
    source /etc/profile.d/modules.sh
    module --force purge
    module load anaconda/2024.06-1
  fi
}

setup_proxy() {
  echo "Setting up the proxy..."
  voms-proxy-init -voms cms -rfc -valid 192:00 --out "$(pwd)/voms_proxy.txt"
  echo "Your proxy is here: $(pwd)/voms_proxy.txt"
  export X509_USER_PROXY="$(pwd)/voms_proxy.txt"
}

usage() {
  cat <<EOF
Usage:
  source setup_env.sh [env] [--no-proxy]

env (optional, default: coffea):
  coffea   -> ${ENV_COFFEA}   (coffea_latest)
  yun      -> ${ENV_YUN}      (yun_coffea_latest)
  pfn      -> ${ENV_PFN}
  py       -> ${ENV_PY}

Options:
  --no-proxy   -> skip voms-proxy-init
EOF
}

# -------------------------
# Main
# -------------------------
ENV_CHOICE="${1:-coffea}"
PROXY_FLAG="${2:-}"

ensure_conda

case "${ENV_CHOICE}" in
  coffea)
    echo "Activating: coffea_latest (default)"
    conda activate "${ENV_COFFEA}"
    ;;
  yun)
    echo "Activating: yun_coffea_latest"
    conda activate "${ENV_YUN}"
    ;;
  pfn)
    echo "Activating: pfn_env"
    conda activate "${ENV_PFN}"
    ;;
  py)
    echo "Activating: python env"
    conda activate "${ENV_PY}"
    ;;
  -h|--help)
    usage
    return 0 2>/dev/null || exit 0
    ;;
  *)
    echo "Unknown env: ${ENV_CHOICE}"
    usage
    echo ""
    echo "Remember that first argument is the env choice and second argument is --no-proxy. For the option --no-proxy, you have to provide an env choice as the first argument."
    echo ""
    return 1 2>/dev/null || exit 1
    ;;
esac

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
