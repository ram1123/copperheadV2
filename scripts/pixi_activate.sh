#!/usr/bin/env bash


# Repository work directory
export WORKDIR="$(pwd)"

# ---- XRootD ----
export XRD_REQUESTTIMEOUT=2400

# ---- CMS env ----
source /cvmfs/cms.cern.ch/cmsset_default.sh

# ---- Append WORKDIR to PYTHONPATH ----
export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"

# ---- Proxy helper function ----
mkdir -p "$WORKDIR/bin"

cat > "$WORKDIR/bin/setup_proxy" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Use WORKDIR if present; else current directory
WORKDIR="${WORKDIR:-$(pwd)}"

echo "Setting up the proxy..."
voms-proxy-init -voms cms -rfc -valid 192:00 --out "${WORKDIR}/voms_proxy.txt"
export X509_USER_PROXY="${WORKDIR}/voms_proxy.txt"
echo "Your proxy is here: ${X509_USER_PROXY}"
EOF

chmod +x "$WORKDIR/bin/setup_proxy"
export PATH="${WORKDIR}/bin:${PATH}"
