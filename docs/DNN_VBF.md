title: DNN for VBF
---



# Input Features for DNN in VBF Analysis

The following features are used as input to the Deep Neural Network (DNN) for the Vector Boson Fusion (VBF) analysis:

- **Dimuon kinematics and angles**
  - `dimuon_mass`
  - `dimuon_pt`, `dimuon_pt_log`
  - `dimuon_rapidity`
  - `dimuon_cos_theta_cs`, `dimuon_phi_cs`

- **Event-by-event mass resolution**
  - `dimuon_ebe_mass_res`
  - `dimuon_ebe_mass_res_rel`

- **Jet kinematics**
  - Leading jet: `jet1_pt`, `jet1_eta`, `jet1_phi`
  - Subleading jet: `jet2_pt`, `jet2_eta`, `jet2_phi`

- **Jet flavor/shape information**
  - `jet1_qgl`, `jet2_qgl`

- **Dijet system**
  - `jj_mass`, `jj_mass_log`
  - `jj_dEta`

- **Additional event activity**
  - `htsoft2`
  - `nsoftjets5`

- **VBF topology variables**
  - `rpt`
  - `ll_zstar_log`
  - `mmj_min_dEta`
  - `pt_centrality`

- **Data-taking period**
  - `year`


# DNN Training

## Preprocessing Steps

### Preprocessing validations

1. Feature distributions before and after preprocessing
1. Correlation matrices before and after preprocessing
1. Plot all input features after preprocessing, from the output parquet files
    1. Add mean and stddev values to the plots

For this, there are two scripts available in the `plotter/` directory:

1. [plot_vbfdnn_input_features_compare.py](../plotter/plot_vbfdnn_input_features_compare.py): to compare feature distributions between signal and background samples before preprocessing.
1. [`dnn_preprocessing_validation.py`](../plotter/dnn_preprocessing_validation.py): to validate the preprocessing steps by plotting the feature distributions from the preprocessed parquet files. For these plots, the mean should be around 0 and the standard deviation around 1. **Note:** the variable `year`  and `nsoftjets5` are not standardized.

## DNN Training

### DNN preprocessing

NOTE: this step uses dask gateway for parallization. Example:
```bash
python MVA_training/VBF_run3/preprocess_dnn.py --config configs/dnn_run2_vbf.yaml \
--base-path /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output \
--tag Run2_NanoV12_forVBFChannel_Apr29_2026 \
--year 2017 \
--use-dask-gateway --cluster-index 0
```

### DNN Hyperparameter Optimization
Once preprocessing is done, do hyperparameter optimization (NOTE: using GPU session will greatly accelerate this).
example:
```bash
VA_training/VBF_run3/hpo_optuna.py --config configs/dnn_run2_vbf.yaml \
--data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/ \
--out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials \
--n-trials 5 --folds 0
```
### DNN Training Command

Example:
```bash
time python MVA_training/VBF_run3/train_dnn.py --config configs/dnn_run2_vbf.yaml \
--data-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/  \
--out-dir dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/trained_best_optuna \
--optuna-best-json dnn/trained_models/Run2_NanoV12_forVBFChannel_Apr29_2026/2017_h-peak_vbf/hpo_optuna/v2_108Trials/optuna_best.json
```


## VBF stage2 and stage3

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 2p -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 3 -k)
```


## DNN bin edge optimization

Change the `command_compact` in `stage1_loop_Improved.sh`, ie
```bash
command_compact="python scripts/compact_parquet_data.py -y $year --input_path $save_path -m $model_trained_path --model_tag $training_tag --add_dnn_score --fix_dimuon_mass --save_postfix $save_postfix "
```

then run compacted command again (no need to delete things), ie

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m compacted -k)

```

We re-bin the DNN score edgges. Configure the stage1 label, years and MC samples and then run the python

```bash
python MVA_training/VBF_run3/scan_bins_for_dnn.py
```


# combine step
first combine the cards:


```bash
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage3_datacards_May04_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026 2017
```

then generate the signficance
```bash
bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage3_datacards_May04_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026 2017
```

Long chain:
```
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2018
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2017
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ Run2

bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ Run2 

```
