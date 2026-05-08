import argparse
from pysrpu.pipeline import (
    run_training,
    run_validation,
    run_rescan,
)

"""
python MVA_training/pileup_symbolic_regression/run_pysr.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2/stage1_output/2022postEE/f1_0/dyTo2L_M-50_incl/0/part135.parquet" \
    -o test_results_temp \
    --features-yaml MVA_training/pileup_symbolic_regression/configs/features.yaml \
    --niterations 3 \
    --population-size 400 \
    --maxsize 7 \
    --mode train

# multiple processes:
python MVA_training/pileup_symbolic_regression/run_pysr.py \
    -i \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/ttjets_dl/0/part*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/ttjets_sl/0/part*.parquet" \
    "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJetsHorn25GeV_Apr09_tightPassLepVeto_NoJER_v2/stage1_output/2022postEE/compacted/ewk_mmjj_mll_105_160/0/part*.parquet" \
    -o validation/pySR/run_multi_bkg_06May \
    --features-yaml MVA_training/pileup_symbolic_regression/configs/features.yaml \
    --mode train \
    --use-glob


python MVA_training/pileup_symbolic_regression/run_pysr.py \
  -i \
  "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_May06_pySRTraining/stage1_output/2022postEE/compacted/dyTo2L_M-50_incl/0/part*.parquet" \
  "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_May06_pySRTraining/stage1_output/2022postEE/compacted/ttjets_dl/0/part*.parquet" \
  "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_May06_pySRTraining/stage1_output/2022postEE/compacted/ttjets_sl/0/part*.parquet" \
  "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_FilterJets_May06_pySRTraining/stage1_output/2022postEE/compacted/ewk_mmjj_mll_105_160/0/part*.parquet" \
  -o validation/pySR/run_multi_bkg_07May_v2 \
  --features-yaml MVA_training/pileup_symbolic_regression/configs/features_2022_2023.yaml \
  --mode train \
  --use-glob \
  --balance-processes \
  --max-per-process-class 2000 \
  --maxsize 7

# 2024

time python MVA_training/pileup_symbolic_regression/run_pysr.py \
    -i "/work/projects/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_FilterJetsHorn25GeV_Apr14_tightPassLepVeto_NoJER_pySRTraining/stage1_output/2024/f1_0/dyTo2Mu_M-50_aMCatNLO/0/part*.parquet" \
    -o validation/pySR/run2024_14Apr_v2 \
    --features-yaml MVA_training/pileup_symbolic_regression/configs/features.yaml \
    --niterations 100 \
    --population-size 400 \
    --maxsize 7 \
    --mode train
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        help="One or more parquet files, glob patterns, or parquet dataset directories. "
             "Comma-separated values are also accepted inside each argument.",
    )
    parser.add_argument("--mode", choices=["train","validate","rescan"], default="train")
    parser.add_argument("--use-glob", action="store_true")
    parser.add_argument("--use-pyarrow", action="store_true")
    parser.add_argument("--niterations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=400)
    parser.add_argument("--maxsize", type=int, default=7)
    parser.add_argument("--hs-eff", type=float, default=0.8)
    parser.add_argument("-o","--output", required=True)
    parser.add_argument(
        "--features-yaml",
        required=True,
        help="YAML file containing feature list",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument(
        "--balance-processes",
        action="store_true",
        help="Balance HS/PU separately within each inferred process group (DY/Top/EWK/...).",
    )
    parser.add_argument(
        "--balance-pt-bins",
        action="store_true",
        help="When balancing processes, also stratify HS/PU sampling within pT bins.",
    )
    parser.add_argument(
        "--max-per-process-class",
        type=int,
        default=2000,
        help="Cap HS and PU examples per process group when --balance-processes is enabled.",
    )
    parser.add_argument(
        "--no-equalize-processes",
        action="store_true",
        help="Keep per-process caps without forcing all process groups to the same size.",
    )
    parser.add_argument("--pt-bins", nargs="+", type=float,
                        default=[25,27,30,32.5,35,37.5,40,42.5,45,47.5,50])
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument(
        "--hs-scan",
        nargs="+",
        type=float,
        default=None,
        help="List of HS efficiency targets to scan"
    )
    parser.add_argument("--pt-min", type=float, default=25.0)
    parser.add_argument("--pt-turnoff", type=float, default=50.0)    

    args = parser.parse_args()

    if args.maxsize < 7:
        parser.error("--maxsize must be at least 7 for the current PySR version.")

    if args.mode == "train":
        run_training(args)

    elif args.mode == "validate":
        run_validation(args)

    elif args.mode == "rescan":
        run_rescan(args)
        
if __name__ == "__main__":
    main()
