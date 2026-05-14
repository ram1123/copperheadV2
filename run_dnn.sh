# # python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run2_vbf.yaml \
# # --base-path /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output \
# # --tag Run2_NanoV12_forVBFChannel_Apr29_2026 \
# # --year 2017 \
# # --use-dask-gateway --cluster-index 0

# time python MVA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run2_vbf.yaml \
# --data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/ \
# --out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials \
# --n-trials 5 --folds 0
# # --n-trials 108 --folds 0


# time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run2_vbf.yaml \
# --data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/  \
# --out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/trained_best_optuna \
# --optuna-best-json dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials/optuna_best.json



# python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run2_vbf.yaml \
# --base-path /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage1_output \
# --tag Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc \
# --year 2018,2017,2016postVFP,2016preVFP \
# --use-dask-gateway --cluster-index 0

# time python MVA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run2_vbf.yaml \
# --data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/ \
# --out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/50Trials \
# --n-trials 50 --folds 0


# time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run2_vbf.yaml \
# --data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/  \
# --out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/trained_best_optuna \
# --optuna-best-json dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/50Trials/optuna_best.json




# python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run2_vbf.yaml \
# --base-path /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage1_output \
# --tag Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc \
# --year 2018,2017,2016postVFP,2016preVFP \
# --use-dask-gateway --cluster-index 0

# time python MVA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run2_vbf.yaml \
# --data-dir dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/ \
# --out-dir dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/01Trials \
# --n-trials 1 --folds 0
# # --out-dir dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/50Trials \
# # --n-trials 50 --folds 0


time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run2_vbf.yaml \
--data-dir dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/  \
--out-dir dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/trained_best_optuna \
--optuna-best-json dnn/trained_models/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/2018-2017-2016postVFP-2016preVFP_h-peak_vbf/hpo_optuna/50Trials/optuna_best.json

