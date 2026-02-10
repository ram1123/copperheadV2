python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run3_vbf.yaml \
--base-path /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/stage1_output/ \
--tag Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV \
--year 2022preEE,2022postEE,2023,2023BPix,2024 \
--use-dask-gateway --cluster-index 0

time python MVA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run3_vbf.yaml \
--data-dir dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/ \
--out-dir dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/hpo_optuna/v2_108Trials \
--n-trials 108 --folds 0

time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run3_vbf.yaml \
--data-dir dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/  \
--out-dir dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/trained_best_optuna_03trail_v3 \
--optuna-best-json dnn/trained_models/Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/hpo_optuna/v1/optuna_best.json
