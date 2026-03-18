#!/usr/bin/env bash
set -euo pipefail

# Copy selected JEC/JER JSON files from CVMFS to a local area,
# preserving the directory hierarchy, and verify with SHA256.
#
# Usage:
#   bash sync_jme_jsons.sh
#   bash sync_jme_jsons.sh /path/to/local_store
#
# Example:
#   bash sync_jme_jsons.sh ./data/POG

DEST_BASE="${1:-./data/POG}"

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "ERROR: sha256sum not found in PATH" >&2
  exit 1
fi

mkdir -p "${DEST_BASE}"

# -------------------------------------------------------------------
# Source file list
# -------------------------------------------------------------------
FILES=(
  # jerc_load_path
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jet_jerc.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jet_jerc.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jet_jerc.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jet_jerc.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jet_jerc.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jet_jerc.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jet_jerc.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/jet_jerc.json.gz"

  # jersmear_load_path
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/jer_smear.json.gz"

  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016preVFP-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2017-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2018-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetvetomaps.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-07-17/jetvetomaps.json.gz"

  "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22CDSep23-Summer22-NanoAODv12/2024-01-31/puWeights.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23/btagging.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23BPix/btagging.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-08-19/btagging.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016preVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016postVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2017-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2018-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-11-27/muon_Z.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetid.json.gz"
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02/jetid.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jmar.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jmar.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jmar.json.gz"
  "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jmar.json.gz"
)

# -------------------------------------------------------------------
# Deduplicate
# -------------------------------------------------------------------
declare -A seen
UNIQ_FILES=()
for f in "${FILES[@]}"; do
  if [[ -z "${seen[$f]+x}" ]]; then
    seen["$f"]=1
    UNIQ_FILES+=("$f")
  fi
done

echo "Destination base: ${DEST_BASE}"
echo "Unique files to sync: ${#UNIQ_FILES[@]}"
echo

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
CHECKSUM_FILE="${DEST_BASE}/checksums.sha256"
VERIFY_LOG="${DEST_BASE}/verify.log"
COPY_LOG="${DEST_BASE}/copy.log"

: > "${CHECKSUM_FILE}"
: > "${VERIFY_LOG}"
: > "${COPY_LOG}"

# -------------------------------------------------------------------
# Copy and checksum
# -------------------------------------------------------------------
copy_one() {
  local src="$1"

  if [[ ! -r "${src}" ]]; then
    echo "MISSING: ${src}" | tee -a "${COPY_LOG}" >&2
    return 1
  fi

  local rel="${src#/}"                  # remove leading slash
  local dest="${DEST_BASE}/${rel}"
  local dest_dir
  dest_dir="$(dirname "${dest}")"

  mkdir -p "${dest_dir}"

  echo "Copying: ${src}" | tee -a "${COPY_LOG}"
  cp -a "${src}" "${dest}"

  local src_sum dest_sum
  src_sum="$(sha256sum "${src}"  | awk '{print $1}')"
  dest_sum="$(sha256sum "${dest}" | awk '{print $1}')"

  if [[ "${src_sum}" != "${dest_sum}" ]]; then
    echo "CHECKSUM MISMATCH: ${src}" | tee -a "${COPY_LOG}" >&2
    echo "  src : ${src_sum}" | tee -a "${COPY_LOG}" >&2
    echo "  dest: ${dest_sum}" | tee -a "${COPY_LOG}" >&2
    return 2
  fi

  echo "${dest_sum}  ${dest}" >> "${CHECKSUM_FILE}"
  echo "OK: ${src} -> ${dest}" | tee -a "${COPY_LOG}"
}

status=0
for f in "${UNIQ_FILES[@]}"; do
  if ! copy_one "${f}"; then
    status=1
  fi
done

echo
echo "Running checksum verification on copied files..."
if ! sha256sum -c "${CHECKSUM_FILE}" > "${VERIFY_LOG}" 2>&1; then
  echo "Checksum verification failed. See: ${VERIFY_LOG}" >&2
  status=1
else
  echo "Checksum verification passed. See: ${VERIFY_LOG}"
fi

echo "Copy log:      ${COPY_LOG}"
echo "Checksum file: ${CHECKSUM_FILE}"
echo "Verify log:    ${VERIFY_LOG}"

exit "${status}"