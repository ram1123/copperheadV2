#!/bin/bash

HOST="eos.cms.rcac.purdue.edu"

# Unique files list
files=(
    "/store/mc/RunIII2024Summer24NanoAODv15/ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/f5f88078-86a5-42b4-8cc2-d648f029c728.root"
    "/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/2520000/a2e56afe-4257-422f-8867-741d8b9b7e8b.root"
    "/store/data/Run2024I/Muon1/NANOAOD/MINIv6NANOv15_v2-v1/90000/ccd541d2-f400-4080-97f9-ef285d848e9c.root"
    "/store/data/Run2024G/Muon1/NANOAOD/MINIv6NANOv15-v2/2520000/76304610-938e-4c3b-89d4-0a18e9d1a670.root"
    "/store/data/Run2024F/Muon1/NANOAOD/MINIv6NANOv15-v1/2540000/c42bf560-3457-4d55-9abc-e6c5d7409fe4.root"
)

# Header
printf "\n%s\n" "===================================================================================================="
printf "%-12s | %-12s | %-8s | %s\n" "xrdfs Query" "xrdadler32" "Status" "File Path"
printf "%s\n" "----------------------------------------------------------------------------------------------------"

for path in "${files[@]}"; do
    # 1. Fetch xrdfs checksum (extracts 8-character adler32 hash)
    xrdfs_raw=$(xrdfs "$HOST" query checksum "$path" 2>/dev/null)
    xrdfs_cksum=$(echo "$xrdfs_raw" | awk '/adler32/ {print $2}')
    [ -z "$xrdfs_cksum" ] && xrdfs_cksum="ERROR"

    # 2. Fetch xrdadler32 checksum (extracts first column)
    adler_raw=$(xrdadler32 "/eos/purdue$path" 2>/dev/null)
    adler_cksum=$(echo "$adler_raw" | awk '{print $1}')
    [ -z "$adler_cksum" ] && adler_cksum="ERROR"

    # 3. Check for match
    if [ "$xrdfs_cksum" != "ERROR" ] && [ "$xrdfs_cksum" == "$adler_cksum" ]; then
        status="MATCH"
    else
        status="MISMATCH"
    fi

    # Output formatted row
    printf "%-12s | %-12s | %-8s | %s\n" "$xrdfs_cksum" "$adler_cksum" "$status" "$path"
done

printf "%s\n\n" "===================================================================================================="

