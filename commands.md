# 18 October 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn_UpdateMassCalib -y 2022preEE -m 1
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff_UpdateMassCalib -y 2022preEE -m 1
```

# 25 March 2025

## nanoAODv9 vs v12

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l Run2_nanoAODv9_25March -y 2018 -m 0 -d
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_25March -y 2018 -m all -d
```


# 25 March 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_March25 -y 2018 -m 0 -d
```

# 24 March 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12  -y 2018 -m 0
```

# 17 March 2025

## Get Z-pT reweight
```bash
bash stage1_loop.sh -v 9 -l WithPurdueZptWgt_DYWithoutLHECut_16Feb_AllYear -y 2018 -d -m zpt_fit
```

## Validation
```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m 0
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m 1
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l DYWithoutLHECut_16Feb_AllYear_UpdatedZptWgt -y 2018  -m all
```



# 12 March 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn_UpdateMassCalib -y 2022preEE -d -m 0
```

## cross-check the mass calibration

### 2018

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_GeoFit -y 2018 -m 0 -d

python src/lib/ebeMassResCalibration/getCalibrationFactor_Improved.py
cp calibration_factors__2018C_12March.json data/res_calib/res_calib_BS_correction_2018UL.json

bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_BSC -y 2018 -m all -d
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_BSC -y 2022preEE -m all -d

bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_12March_NoGeoNoBSC -y 2022preEE -m 0 -d

```

### 2018v9

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv9.yaml -v 9 -l Run2_nanoAODv9_12March_GeoFit -y 2018 -m all -d
```

# 10 March 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0 -f
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 1 -f


bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn -y 2022preEE -m 0
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOn -y 2022preEE -m 1

bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run3_nanoAODv12_BSOff -y 2022preEE -m 0

```

---

```bash
bash stage1_loop.sh  -c configs/datasets/dataset.yaml -v 12 -l Run3_nanoAODv12_TEST -m 0
bash stage1_loop.sh  -c configs/datasets/dataset.yaml -v 12 -l Run3_nanoAODv12_TEST -m 0 -d

```

# 03 March 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_03March_BSOff -m 0 -d
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_03March_BSOff -m 1 -d
```

# 24 Feb 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset.yaml -v 9 -l Run2_nanoAODv9_24Feb_BSoff -m 0 -d
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_24Feb_BSoff -m all -d
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l Run2_nanoAODv12_24Feb_BSon -m 0 -d
```

# 23 Feb 2025

```bash
bash stage1_loop.sh  -c configs/datasets/dataset_nanoAODv12.yaml -y 2018 -v 12 -l Run2_nanoAODv12_24Feb -m 0
```
