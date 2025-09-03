# 02 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m dnn -d 1) 2>&1 | tee full_dnn_train_run_Scan_2Sep.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m dnn_train -d 1) 2>&1 | tee full_dnn_train_run_Scan_2Sep_train.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn ) 2>&1 | tee full_dnn_train_run_Scan_2Sep_FullRun2.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn ) 2>&1 | tee full_dnn_train_run_Scan_2Sep_FullRun2_train.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn ) 2>&1 | tee full_dnn_train_run_Scan_2Sep_FullRun2_21H27.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn_train ) 2>&1 | tee full_dnn_train_FullRun2_WithBestHP_3Sep.log

python bo_plot_root.py /depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/run2_h-peak_vbf_ScanHyperParamTest_ScanV1/bo_logs/bo_trials_live.jsonl /depot/cms/users/shar1172/copperheadV2_main/dnn/trained_models/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/run2_h-peak_vbf_ScanHyperParamTest_ScanV1/bo_logs/ --topN 45

```

# 28 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFixV3 -y "2018 2016postVFP" -k -m all -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFixV3 -y "2016postVFP" -k -m all)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -k -m dnn) 2>&1 | tee full_dnn_pre_train_run_1Sep.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -k -m dnn_train -d 1) 2>&1 | tee full_dnn_train_run_2Sep.log

```
# 26 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFixV2 -y "2017 2016preVFP 2016postVFP" -k -m all)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFixV2 -y "2017 2016preVFP 2016postVFP" -k -m 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFixV2 -y "2017" -k -m 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -k -m dnn_pre)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -k -m dnn_train)
```


# 23 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l haddtest_olddir -y "2018" -m all -d 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l haddtest_newhadddir -y "2018" -m all -d 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFix_hadd -y "2018" -m all -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFix -y "2018" -m compact -d 1 -k)
```

# 21 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l debug_softjet -y "2018" -m 0 -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_SoftJetBugFix -y "2018" -m all -d 1 -k)

python scripts/mergeNanoAODRootFiles.py -i /eos/purdue/store/user/rasharma/customNanoAOD/UL2018/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8 -o /depot/cms/users/shar1172/hmm/test_hadd -f GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8.root

python scripts/mergeNanoAODRootFiles.py -i  /eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9 -o /depot/cms/users/shar1172/hmm/test_hadd -f DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9.root


python scripts/mergeNanoAODRootFiles.py -i  /eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9 -o /store/user/rasharma/Run2_CustomNanoAODv12/hadded/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9  -f DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9.root

```

# 18 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2017 2016preVFP 2016postVFP" -m compact)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2016preVFP 2016postVFP" -m all)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p Latest -y "2018" -m 3)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p Latest -y "2017" -m 3 -d 1)
```


# 13 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m 2 -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2018" -m 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2018" -m compact)
```

# 11 August 2025

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt  -v 12 -y 2018 -m 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2018" -m compact)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2018" -m 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m 3)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2017 2016preVFP 2016postVFP" -m 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m compact)
```

# 06 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2018" -m 0 -d 1)
```

# 04 August 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2017 2016preVFP 2016postVFP" -m all -d 1)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt  -v 12 -y 2018 -m dnn_pre)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2017 2016preVFP 2016postVFP" -m compact -d 1)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt  -v 12 -y 2018 -m dnn)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt  -v 12 -y 2018 -m 2)
```

# 30 July 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Test -m all -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018 2017 2016preVFP 2016postVFP" -m all)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July  -v 12 -y 2018 -m 2)
```

# 28 July 2025

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2017" -m zpt_fit2 -n 1)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_17July -m all -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_17July -m all -y "2016preVFP 2016postVFP" -d 1)
```

# 24 July 2025

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July  -v 12 -y "2018 2017 2016preVFP 2016postVFP" -m compact)
```

## Z-pT reweighting

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y 2017 -m zpt_fit0)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2018 2017 2016preVFP 2016postVFP" -m zpt_fit0)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2018 2017 2016preVFP 2016postVFP" -m zpt_fit12 -n 0)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2018 2017 2016preVFP 2016postVFP" -m zpt_fit12 -n 1)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -y "2018 2017 2016preVFP 2016postVFP" -m zpt_fit12 -n 2)
```

# 21 July 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_17July -y 2017 -m all -d 1)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July  -v 12 -y 2018 -m dnn)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July  -v 12 -y 2018 -m compact)
```

# 18 July 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l DmitryServiceX_Check -y 2018 -d 1 -m all)
```

# 17 July 2025

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -m all -v 12)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -m all -v 12 -y 2018)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -m compact -v 12 -y 2018)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_UpdatedQGL_17July -m dnn -v 12 -y 2018)
```

# 11 July 2025

```bash
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_08June -y 2018 -m dnn_pre)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_08June -y 2018 -m dnn)
time(bash stage1_loop_Improved.sh -l Run2_nanoAODv12_08June -y 2018 -m compact)
```

# 08 July 2025

```bash
time(bash stage1_loop_Improved.sh  -v 12 -l Run2_nanoAODv12_08June -y 2018 -m 2)
```

# 03 July 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m zpt_fit0)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m zpt_fit12 -n 2)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l DmitryServiceX_Check -y 2018 -m 0 -d 1)
```

# 30 June 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2017 -m 0 -d 1 -s 1)
```


# 26 June 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2016postVFP -m all -d 1)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2016preVFP -m all -d 1)
```

# File list to be deleted

```bash
root://eos.cms.rcac.purdue.edu:1094//store/user/rasharma/customNanoAOD_Gautschi_2016/UL2016/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/E0B5655A-CAEB-9D45-BD47-00B4AECBB5FD.root
```

# 20 June 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2016postVFP -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2016preVFP -m all)

time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2016postVFP -m zpt_fit0)

```

# 17 June 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m 2)

time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2017 -m all)

time(python MVA_training/VBF/dnn_preprocessor.py --label "Run2_nanoAODv12_08June_MiNNLO" -cat "vbf" --year 2018)
time(python MVA_training/VBF/dnn_train.py --label "Run2_nanoAODv12_08June_MiNNLO")
```


# 16 June 2025

```bash
time(python MVA_training/VBF/dnn_preprocessor.py --label "Run2_nanoAODv12_08June" -cat "vbf" --year 2018)
time(python MVA_training/VBF/dnn_train.py --label "Run2_nanoAODv12_08June")
```

# 09 June 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m zpt_fit)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_08June -y 2018 -m 2)
```


# OLD

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l May28_NanoV12 -y 2018 -m all) Run2_nanoAODv12_08June
```

bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l test_test -y 2018 -m 1 -d 1

# 06 May 2025

```bash
time(python getCalibrationFactor_Improved.py --isMC  --years "2016preVFP")
time(python getCalibrationFactor_Improved.py --isMC --validate --years "2016preVFP")
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_UpdatedMassCalib -y 2018 -m all)
```

# 29 April 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_UpdatedMassCalibv2 -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2016preVFP -m all)
```

# 23 April 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff_TightPUID -y 2016postVFP -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff_TightPUID -y 2016preVFP -m all)
```
# 22 April 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_UpdatedMassCalib -y 2018 -m all)
# time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_RochOff -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JEROff -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_TightPUID -y 2018 -m all)
```

# 21 April 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JEROff -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff_TightPUID -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff_TightPUID -y 2017 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JECOff_TightPUID  -m all)

# JER OFF
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_JEROff_TightPUID -y 2018 -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2018 -m all)
```


# 20 April 2025

```bash
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2016postVFP -m all)
time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2016preVFP -m all)
```

# 18 April 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2018 -m all
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12 -y 2017 -m all
# bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April17_NanoV12 -y 2018 -m 0
# bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April17_NanoV12 -y 2018 -m 1
```

# 25 March 2025

## nanoAODv9 vs v12

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l Run2_nanoAODv9_25March -y 2018 -m 0 -d
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_25March -y 2018 -m all -d
```


# 25 March 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_March25 -y 2018 -m 0 -d
```

# 24 March 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12  -y 2018 -m 0
```

# 17 March 2025

## Get Z-pT reweight
```bash
bash stage1_loop_Improved.sh -v 9 -l WithPurdueZptWgt_DYWithoutLHECut_16Feb_AllYear -y 2018 -d -m zpt_fit
```

## Validation
```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m 0
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m 1
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m all
```



# 12 March 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn_UpdateMassCalib -y 2022preEE -d -m 0
```

## cross-check the mass calibration

### 2018

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_GeoFit -y 2018 -m 0 -d

python src/lib/ebeMassResCalibration/getCalibrationFactor_Improved.py
cp calibration_factors__2018C_12March.json data/res_calib/res_calib_BS_correction_2018UL.json

bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_BSC -y 2018 -m all -d
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_BSC -y 2022preEE -m all -d

bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_NoGeoNoBSC -y 2022preEE -m 0 -d

```

### 2018v9

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l Run2_nanoAODv9_12March_GeoFit -y 2018 -m all -d
```

# 10 March 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0 -f
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 1 -f


bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn -y 2022preEE -m 0
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn -y 2022preEE -m 1

bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0

```

---

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset.yaml -v 12 -l Run3_nanoAODv12_TEST -m 0
bash stage1_loop_Improved.sh  -c configs/datasets/dataset.yaml -v 12 -l Run3_nanoAODv12_TEST -m 0 -d

```

# 03 March 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_03March_BSOff -m 0 -d
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_03March_BSOff -m 1 -d
```

# 24 Feb 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset.yaml -v 9 -l Run2_nanoAODv9_24Feb_BSoff -m 0 -d
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_24Feb_BSoff -m all -d
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_24Feb_BSon -m 0 -d
```

# 23 Feb 2025

```bash
bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -y 2018 -v 12 -l Run2_nanoAODv12_24Feb -m 0
```
