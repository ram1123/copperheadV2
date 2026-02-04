# Label name scheme
- or Run3_nanoAODv<version>_<date>_<additional_info>


# 02 February 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_01Feb_JecJerFilterJets -y "2024" -m 1 -k 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJets -y "2022postEE" -m 1 -k 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -y "2022postEE 2022preEE 2023 2023BPix" -m 1 -k 2 -i 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_02Feb_FilterJetsHorn30GeV -y "2024" -m 1 -k 2 -i 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -y "2022postEE 2022preEE" -m compact -k 2 -i 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_02Feb_FilterJetsHorn30GeV -y "2024" -m compact -k 2 -i 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJets -y "2022postEE" -m 1 -k 2 -i 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -y "2022postEE" -m 1 -k 2 -i 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -y "2022postEE" -m compact -k 2 -i 2)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV -y "2022postEE" -m zpt_fit0 -k 2 -i 2)

```

# 31 January 2026

```bash
time python MVA_training/VBF_new/preprocess_dnn.py --config configs/dnn_run3_vbf.yaml --base-path /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output --tag Run3_01Feb_v1 --year 2022preEE,2022postEE,2023,2023BPix,2024 --use-dask-gateway  --cluster-index 0

time python MVA_training/VBF_new/hpo_optuna.py --config configs/dnn_run3_vbf.yaml --data-dir dnn/trained_models/Run3_01Feb_v1/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/ --out-dir dnn/trained_models/Run3_01Feb_v1/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/hpo_optuna/v1 --n-trials 108 --folds 0,1,2,3

python MVA_training/VBF_new/train_dnn.py --config configs/dnn_run3_vbf.yaml --data-dir dnn/trained_models/Run3_01Feb_v1/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/ --out-dir dnn/trained_models/Run3_01Feb_v1/2022preEE-2022postEE-2023-2023BPix-2024_h-peak_vbf/trained_best_optuna_51trail


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_01Feb_test -y "2024" -m 1 -k -d 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Feb_test -y "2022preEE 2022postEE 2023 2023BPix" -m 1 -k -d 2)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Feb_JecJer -y "2022preEE 2022postEE 2023 2023BPix" -m 1 -k 2 -i 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_01Feb_JecJer -y "2024" -m 1 -k 2 -i 1)
```



# 29 January 2026

```bash
python MVA_training/VBF_new/preprocess_dnn.py --config configs/dnn_run3_vbf.yaml --base-path /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output/2022postEE/compacted/ --tag kfold_shuffleTrue --year 2022postEE

time python MVA_training/VBF_new/hpo_optuna.py --config configs/dnn_run3_vbf.yaml --data-dir dnn/trained_models/kfold_shuffleTrue/2022postEE_h-peak_vbf/ --out-dir dnn/trained_models/kfold_shuffleTrue/2022postEE_h-peak_vbf/hpo_optuna/v1 --n-trials 51 --folds 0

python MVA_training/VBF_new/train_dnn.py --config configs/dnn_run3_vbf.yaml --data-dir dnn/trained_models/kfold_shuffleTrue/2022postEE_h-peak_vbf/ --out-dir dnn/trained_models/kfold_shuffleTrue/2022postEE_h-peak_vbf/trained_best_optuna_51trail
```

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "run3" -m dnn_pre -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterEvents -y "2022preEE 2022postEE 2023 2023BPix 2024" -m 1 -k)
```

# 23 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m 1 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -y "2022postEE" -m zpt_fit12 -n 0)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m zpt_fit0 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2022preEE" -m zpt_fit12 -n 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2022postEE" -m zpt_fit12 -n 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2023" -m zpt_fit12 -n 0)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2023BPix" -m zpt_fit12 -n 0)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m calib -k )

```


# 22 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m compact -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "run3" -m dnn_pre -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "run3" -m dnn_train -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "2022preEE" -m 2 -p "Jan22_test" )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "2022preEE 2022postEE" -m 2 -p "Jan22_test_Syst" )
```

# 20 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022postEE" -m 0 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022postEE" -m compact -k )

time python scripts/sync_parquet_dimuon.py /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_15Jan_NoJVM/stage1_output/2024/compacted/dyTo2Mu_M-50_aMCatNLO/0 -o dyTo2Mu_M-50_aMCatNLO_2024_sync_NoJVM.csv

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -y "2022postEE" -m calib -k -d 2)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m 0 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2023BPix" -m 0 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m 1 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_21Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m 1 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_21Jan_JVMFilterJets -y "2024" -m 1 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022preEE 2022postEE 2023 2023BPix" -m calib -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "run3" -m dnn_pre -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022preEE" -m dnn_pre -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_20Jan_JVMFilterJets -y "2022preEE" -m dnn_train -k)

ruff check --select I ./run_prestage.py
```


# 16 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -y "2022postEE" -m zpt_fit12 -n 0)
```


# 15 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_15Jan_NoJVM -y "2024" -m 1 -k -d 1)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_15Jan_JVMFilterJets -y "2022postEE" -m 0 -k )
```

# 12 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_16Dec_NoJVM -y "2024" -m calib -k -d 2)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -y "2022preEE 2022postEE 2023 2023BPix" -m zpt_fit )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv12_23October -y "2022preEE 2022postEE 2023 2023BPix" -m calib -k )


# sync dy 2024 hyeon
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3_sync.yaml -v 15 -l Run3_nanoAODv15_14Jan_SyncHyeon -y "2024" -m 1)
time python scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_14Jan_SyncHyeon//stage1_output/2024/f1_0/dyTo2Mu_M-50_aMCatNLO/0  /depot/cms/users/yun79/hmm/copperheadV1clean/Run3OJan14_2026_20224Test//stage1_output/2024/f1_0/dy_M-50_aMCatNLO/0

time python scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv15_14Jan_SyncHyeon//stage1_output/2024/f1_0/dyTo2Mu_M-50_aMCatNLO/0  /depot/cms/users/shar1172/hmm/copperheadV1clean/Run3OJan14_2026_20224Test//stage1_output/2024/f1_0/dy_M-50_aMCatNLO/0 -o hyeon_2024_sync_both_I_ran.csv
```

# 08 January 2026

```bash
python scripts/create_basic_info_stage1_files.py -p /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_DEPOT -w 64
python scripts/create_basic_info_stage1_files.py -p /eos/purdue/store/user/rasharma/hmm/reducedNtuples/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_EOS -w 32

time python scripts/create_basic_info_stage1_files.py -p /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_DEPOT -w 64
time python scripts/create_basic_info_stage1_files.py -p /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output -l Run2_nanoAODv12_AK8jets_DEPOT -w 64

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_16Dec_NoJVM -y "2022postEE" -m 0 -k )

```

# 05 January 2026

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_16Dec_NoJVM -y "2024" -m calib)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_16Dec_NoJVM -y "2022preEE" -m 1 -k )
```

# 21 December 2025

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_21Dec_JVMfilterjets -y "2022preEE" -m 1 -k )


# 18 December 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3_sync.yaml -v 12 -l Run3_nanoAODv12_Peking_sync -y "2022preEE" -m 1)
time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_Peking_sync/stage1_output/2022preEE/f1_0/data_C/0/ -o Purdue_event_info_2022Cv1.txt

time python ./scripts/sync_parquet_dimuon.py docs/sync/sync_Peking/peking.txt Purdue_event_info_2022Cv1.txt  -o compare_Peking_vs_Purduev1.txt
time python ./scripts/sync_parquet_dimuon.py docs/sync/sync_Peking/peking.txt Purdue_event_info_2022C.txt  -o compare_Peking_vs_Purdue.txt


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l Run2_nanoAODv12_HEMVetoFix_NoSyst -y "2018" -m compact  -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l Run2_nanoAODv12_HEMVetoFix_NoSyst -y "2018" -m zpt_fit0 )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l Run2_nanoAODv12_18Dec_HEMVetoFix -y "2018" -m 1  -k)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l Run2_nanoAODv15_18Dec_NoSyst -y "2017" -m 1  -k)

time(python src/lib/ebeMassResCalibration/getCalibrationFactor.py  --years "2018" --extraString "_HEMVetoFix")

```

# 17 December 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 12 -l Run2_nanoAODv15_25Nov_SwitchOffJec -y "2017" -m 1 -k -d 1)
time python ./scripts/sync_parquet_dimuon.py  sync_0_vs_3_dimuon_diff.txt docs/sync/sync_Peking/peking.txt
time python ./scripts/sync_parquet_dimuon.py Purdue_event_info_2022C.txt event_info_2022C_v2.txt  -o compare_old_vs_new.txt
time python ./scripts/sync_parquet_dimuon.py docs/sync/sync_Peking/peking.txt event_info_2022C_v2.txt  -o compare_Peking_vs_myNew.txt

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l Run2_nanoAODv12_HEMVetoFix_NoSyst -y "2018" -m 1  -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_16Dec_JVMjetFilter -y "2022preEE" -m 1 -k)
```

# 16 December 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_16Dec -y "2024" -m 1 -k -d 1)

# configs/datasets/dataset_nanoAODv12_run3_sync.yaml

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3_sync.yaml -v 15 -l Run3_nanoAODv12_Peking_sync -y "2022preEE" -m 1)
time python ./scripts/sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_Peking_sync/stage1_output/2022preEE/f1_0/data_C/0/

# Run2_nanoAODv12_16Dec_HEMVetoFix_NoSyst/stage1_output/2017/f1_0/data_C/0
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l Run2_nanoAODv12_16Dec_HEMVetoFix_NoSyst -y "2018" -m 1  -k)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_16Dec_NoJVM -y "2024" -m 1 -k )

```

# 15 December 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_SyncHyeon -y "2024" -m 0 -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l Run2_nanoAODv15_15Dec -y "2017" -m 0 -k  -d 1)
```

# 12 December 2025

```bash


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_Dec13_NoSyst -y "2017" -m 1  -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_Dec13_NoSyst -y "2017" -m compact  -k)



time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix -y "2018" -m 1  -k)
/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix/stage1_output/2018/f1_0/dy_M-50_aMCatNLO/0/part106.parquet

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2017" -m dnn_pre -k )
```

# 11 December 2025

```python
time python ./sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/vbf_powheg_dipole/0 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/vbf_powheg_dipole/0

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nan
oAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2017" -m 1  -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix -y "2018" -m 2 -k -p "Dec11_oldDNN17bins")
```

## DNN related

```bash

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "run2" -m dnn_pre -k)
# THe above command failed when I run them together so, I am running year wise and then will merge the preprocessor files

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2016preVFP 2016postVFP 2017 2018" -m dnn_pre -k -d 2)

python MVA_training/VBF/run2_legacyModel/merge_individual_preprocessor.py

```


# 10 December 2025

```bash
python ./sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/data_D/0/* /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/data_D/0/*
time python ./sync_parquet_dimuon.py  /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/f1_0/data_D/0 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2017/f1_0/data_D/0
```

# 09 December 2025

<!--
/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2016preVFP/compacted
/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2016postVFP/compacted
/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/compacted
/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2/stage1_output/2018/compacted -->


```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2018" -m dnn_pre -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l dnn_train_softlink_HEMVetoFix -y "run2" -m dnn_pre -k)
```



# 08 December 2025

## Done

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2018" -m dnn_pre  -k)
```

## To do
```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2016preVFP 2016postVFP 2017" -m 0  -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "2018" -m compact  -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2 -y "run2" -m 1  -k)
```

# 02 December 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_NoJVM -p HPScan_03Sep_17bins -y "2023" -m 0 -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_NoJVM -p HPScan_03Sep_17bins -y "2023" -m compact -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_JVM -p HPScan_03Sep_17bins -y "2023" -m compact -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_JVM_Horn50GeV -p HPScan_03Sep_17bins -y "2023" -m compact -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_JVMFilterJets -p HPScan_03Sep_17bins -y "2023" -m compact -k  -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_01Dec_JVMFilterJets_Horn50GeV -p HPScan_03Sep_17bins -y "2023" -m compact -k  -d 1)
```

# 01 December 2025

## To Rerun stage 1 with nanoAODv12 for 2018 after HEM veto fix

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix -y "2018" -m 1  -k)
```

## others

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run3.yaml -v 15 -l Run3_nanoAODv15_01Dec -p HPScan_03Sep_17bins -y "2024" -m 0 -k  -d 1)
```



# 28 November 2025

```bash
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l Run2_nanoAODv15_28Nov -y "2017" -m 0 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2_hadded_newFormat.yaml -v 12 -l Run2_nanoAODv12_28Nov_HEMVetoFix -y "2018" -m 0 -k)
```

# 20 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 12 -l Run2_nanoAODv15_25Nov_SwitchOffJec -y "2017" -m 1 -k -d 1)

```


# 18 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_18Nov_JetVetoFilterJetsOnly -p HPScan_03Sep_17bins -y "2023" -m 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_18Nov_JetVetoFilterJetsOnly_HornPt50GeV -p HPScan_03Sep_17bins -y "2023" -m 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_18Nov_JetVetoFilterJetsOnly_HornPt50GeV_TightMuon -p HPScan_03Sep_17bins -y "2023" -m 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_18Nov_JetVetoFilterJetsOnly_HornPt50GeV_JetIDTightLepVetoPass -p HPScan_03Sep_17bins -y "2023" -m 0 -k )
```

# 17 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_17Nov_JetVetoMap -p HPScan_03Sep_17bins -y "2023" -m 0 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_17Nov_JetVetoMap_HornpT50GeV -p HPScan_03Sep_17bins -y "2023" -m 0 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_17Nov_JetVetoFilterJetsOnly_HornpT50GeV -p HPScan_03Sep_17bins -y "2023" -m 0 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_17Nov_JetVetoFilterJetsOnly -p HPScan_03Sep_17bins -y "2023" -m 1 -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_17Nov_NoJetVetoMap -p HPScan_03Sep_17bins -y "2023" -m all -k )
```

# 07 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022postEE" -m 0 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run3.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2023 2023BPix" -m 0 -k -d 1)
```

# 05 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022preEE 2022postEE" -m 0 -k -d 1)
```

# 01 November 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022postEE" -m 0 -k)
```
# 31 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022preEE" -m 0 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022preEE" -m 1 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022preEE" -m compact -k)
```

# 21 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run3_nanoAODv12_23October -p HPScan_03Sep_17bins -y "2022preEE" -m 0 -k)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_WithVHVeto -y "2018 2017 2016preVFP 2016postVFP" -m 3)
```

# 17 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_NoVHVeto -y "2016preVFP" -m 0 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_NoVHVeto -y "2016preVFP" -m 2 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_NoVHVeto -y "2018 2017 2016preVFP 2016postVFP" -m 3)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_WithVHVeto -y "2018 2017 2016preVFP 2016postVFP" -m 2 -k )

```

# 15 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2016preVFP" -m 3)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2016postVFP" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins_NoVHVeto -y "2018 2017 2016preVFP 2016postVFP" -m 2 -k)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2017" -m all -k )
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2017" -m 2p )

```

# 09 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l CrossCheckCutFlow_BR_fatjet -p HPScan_03Sep_17bins -y "2018PR" -m 1 -k )

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2018" -m 1 -k ) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage1_2018_10Oct.log
time bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2018" -m 1 -k
time bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m dnn_pre -k
```


# 08 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k )
```

# 06 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2018" -m 0 -k -d 1) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage1_2018_MC.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2018" -m all -k) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage01_2018_Data.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2016postVFP" -m all -k) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage01_2016postVFP_Data.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m 3)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m 2p)

```

# 02 October 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m 0 -k) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage1_05Oct_data.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m all -k -d 1) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage1_05Oct_all.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_AK8jets -p HPScan_03Sep_17bins -y "2017" -m 2 -k ) 2>&1 | tee Run2_nanoAODv12_AK8jets_stage2_05Oct.log
```

# 23 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2018" -m 3)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins_NoDYVBF -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k -d 1)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2018" -m 3)

```

# 22 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13binSignificanceScan -y "2018" -m 2 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13binSignificanceScan -y "2017" -m 2 -k -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -p HPScan_03Sep_13binSignificanceScan -y "2018" -m compact -k -d 1)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_17bins -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k)

```


# 20 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13binSignificanceScan -y "2018" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13binSignificanceScan -y "2017" -m all -k -d 1)
```

# 19 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2016postVFP" -m 0 -d 1)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2016postVFP" -m 1 -k -d 1)

# Re-run for DY012 as yesterday it was with 13 bins
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2018" -m 2 -k -d 1)
```


# 18 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p Latest -y "2018" -m 3 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p Latest -y "2018" -m 2 -k)
```


```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep -y "2018" -m 3 -d 1 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p Latest -y "2018" -m 2 -k)
```

## New scan with 21 bins

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2018" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2018" -m 3 -k)


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13bins -y "2018" -m 2 -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_13bins -y "2018" -m 3 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_21bins -y "2017 2016preVFP 2016postVFP" -m 2 -k)
```

## Produce DYJ2 and DYJ01 samples

```bash
time python scripts/split_dy_by_gjj_mass.py --minnlo /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_M-50_MiNNLO/0/ /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_M-100To200_MiNNLO/0/  --vbff /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_VBF_filter/0/ --outdir-ge2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/DYJ2/0/ --outdir-lt2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/DYJ01/0/ --use_gateway

time python scripts/split_dy_by_gjj_mass.py --minnlo /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2017/compacted/dy_M-50_MiNNLO/0/ /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2017/compacted/dy_M-100To200_MiNNLO/0/  --vbff /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2017/compacted/dy_VBF_filter/0/ --outdir-ge2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2017/compacted/DYJ2/0/ --outdir-lt2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2017/compacted/DYJ01/0/ --use_gateway

time python scripts/split_dy_by_gjj_mass.py --minnlo /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016preVFP/compacted/dy_M-50_MiNNLO/0/ /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016preVFP/compacted/dy_M-100To200_MiNNLO/0/  --vbff /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016preVFP/compacted/dy_VBF_filter/0/ --outdir-ge2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016preVFP/compacted/DYJ2/0/ --outdir-lt2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016preVFP/compacted/DYJ01/0/ --use_gateway

time python scripts/split_dy_by_gjj_mass.py --minnlo /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016postVFP/compacted/dy_M-50_MiNNLO/0/ /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016postVFP/compacted/dy_M-100To200_MiNNLO/0/  --vbff /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016postVFP/compacted/dy_VBF_filter/0/ --outdir-ge2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016postVFP/compacted/DYJ2/0/ --outdir-lt2 /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2016postVFP/compacted/DYJ01/0/ --use_gateway

```


# 17 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_OldHiddenLayer -y "2018" -m 3 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -p HPScan_03Sep_NewSoftVars -y "2018" -m 2 -k) 2>&1 | tee Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_Stage2_17Sep.log

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -p HPScan_03Sep_NewSoftVars -y "2018" -m 3)

```


# 11 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn -d 1) 2>&1 | tee full_dnn_run_11Sep.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_OldHiddenLayer -y "2018" -m 2 -k) 2>&1 | tee Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Stage2_12Sep.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep_OldHiddenLayer -y "2018" -m 3 -k) 2>&1 | tee Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_Stage3_15Sep.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -y "run2" -m dnn_pre -d 1 -k) 2>&1 | tee Run2_nanoAODv12_07Sep2025_dnn_run_12Sep.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -y "run2" -m dnn_train -d 1) 2>&1 | tee Run2_nanoAODv12_07Sep2025_dnn_train_12Sep.log

```

# 07 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -p HPScan_03Sep -y "2018 2017 2016preVFP 2016postVFP" -m all -k)
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_hadded.yaml -v 12 -l Run2_nanoAODv12_07Sep2025 -p HPScan_03Sep -y "2018 2017 2016preVFP 2016postVFP" -m all -k -d 1) 2>&1 | tee Run2_nanoAODv12_07Sep2025_Run9Sep.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep -y "2018" -m 2 -d 1 -k) 2>&1 | tee stage2_07Sep_2018.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep -y "2018" -m 3 -d 1 -k) 2>&1 | tee stage3_07Sep_2018.log
```

# 04 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep -y "2016postVFP" -m all -d 1 -k)

time python split_dy_by_gjj_mass.py --in /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_M-50_MiNNLO/0/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_M-100To200_MiNNLO/0/ /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/dy_VBF_filter/0/ --outdir-ge2 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/DYJ2/0/ --outdir-lt2 /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar/stage1_output/2018/compacted/DYJ01/0/ --use_gateway
```


# 03 September 2025

```bash
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2018" -m compact -k ) 2>&1 | tee compact_dnn.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -y "2016preVFP 2016postVFP" -m all -d 1 -k)

time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "run2" -m dnn -d 1) 2>&1 | tee full_dnn_train_run_Scan_3Sep.log

# TO-DO
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt -y "2017 2016preVFP 2016postVFP" -m compact  ) 2>&1 | tee compact_dnn_allyear.log
time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep  -y "2018" -m 2 -k ) 2>&1 | tee stage2_03Sep.log


time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar -p HPScan_03Sep  -y "2017 2016preVFP 2016postVFP" -m 2 -k ) 2>&1 | tee stage2_3year_03Sep.log
```

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
