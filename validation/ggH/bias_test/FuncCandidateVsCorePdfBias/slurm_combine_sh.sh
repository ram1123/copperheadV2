#!/bin/sh

# Command to run
# pixi run -e combine --manifest-path /cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2_dev/pixi.toml \
#   sh slurm_combine_sh.sh 1 1 0 1

# setup env -----------------------------------------------------------
ulimit -s unlimited
set -e
# cd /work/users/shar1172/copperheadV2_Feb2026
# source /work/users/shar1172/copperheadV2_Feb2026/enter_pixi.sh
#-----------------------------------------------------------


ntoys="50"
nGeneratedToys="50"
fitdiag_parallel="${BIAS_FITDIAG_PARALLEL:-30}"
expect_signal="${BIAS_EXPECT_SIGNAL:-0}"
# true_idx="1"
# out_idx="core"
true_idx=$3
out_idx=$4
minStrat="0"
# Use arithmetic expansion to perform the modulo operation
if (( $1 % 2 == 0 )); then
  random_seed="$1${2:0:4}"
  # random_seed="$1${2: -4}"
else
  # random_seed="${2:0:3}$1"
  random_seed="$1${2: -4}"
fi
# random_seed=$1${$2:0:2}

cat="all"

slurm_dir=slurmJobs/slurmJob_in${true_idx}_out${out_idx}_${1}_${2}
start_dir="${5:-$(cd "$(dirname "$0")" && pwd)}"
start_dir="$(cd "${start_dir}" && pwd)"
echo "start_dir: ${start_dir}"
slurm_path=${start_dir}/${slurm_dir}
datacard1_name="datacard_comb_sig_all_ggh_fitFuncCand.txt"
datacard2_name="datacard_comb_sig_all_ggh_corePdf.txt"
ws1_path="funcCandidate_workspace"
ws2_path="corePdf_workspace"

mkdir -p ${slurm_path}

cd ${start_dir}
cp -f ${datacard1_name} ${slurm_path}
cp -f ${datacard2_name} ${slurm_path}
cp -f -r ${ws1_path} ${slurm_path}
cp -f -r ${ws2_path} ${slurm_path}
echo "slurm_path: ${slurm_path}"
cd ${slurm_path}
text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh_fitFuncCand.txt
text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh_corePdf.txt
echo "random_seed: ${random_seed}"
echo "slurm_dir: ${slurm_dir}"

combineTool.py datacard_comb_sig_${cat}_ggh_fitFuncCand.root -M GenerateOnly -m 125 --setParameters pdf_index_ggh=${true_idx} -t ${nGeneratedToys}  --expectSignal ${expect_signal} --saveToys -m 125 --freezeParameters pdf_index_ggh --X-rtd MINIMIZER_MaxCalls=20000000000 -s ${random_seed}
time(combineTool.py datacard_comb_sig_${cat}_ggh_corePdf.root -M FitDiagnostics   -m 125 --toysFile higgsCombine.Test.GenerateOnly.mH125.${random_seed}.root   -t ${ntoys}  --expectSignal ${expect_signal} --cminRunAllDiscreteCombinations  --rMin -20 --rMax 20 --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${out_idx} --cminDefaultMinimizerStrategy=${minStrat}  --X-rtd MINIMIZER_MaxCalls=20000000000 -n bias_in${true_idx}_out${out_idx}_nToys${ntoys}_${cat}_asimovDataset --parallel ${fitdiag_parallel} )
