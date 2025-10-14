# Summary

- Run the script [`getCalibrationFactor.py`](../src/lib/ebeMassResCalibration/getCalibrationFactor.py) to get the calibration factor for the EBE mass resolution.
- The script will generate a JSON file with the calibration factors for different categories.
- Copy the generated JSON file to the path `data/res_calib/` and update the path and name of this JSON file in the config file: `configs/parameters/correction_filelist.yaml`.

# How to run

**NOTE:** Need to update the path of the input parquet files in the code [src/lib/ebeMassResCalibration/getCalibrationFactor.py](../src/lib/ebeMassResCalibration/getCalibrationFactor.py) before running.

```bash
cd src/lib/ebeMassResCalibration
python getCalibrationFactor.py --years "2018"
python getCalibrationFactor.py --years "2018" --fixCat "30-45_EE" --backup
# python getCalibrationFactor.py --years "2018" --isMC
# time(python getCalibrationFactor.py --isMC  --years "2016preVFP")
# time(python getCalibrationFactor.py --isMC --validate --years "2016preVFP")
# time(bash stage1_loop_Improved.sh  -c configs/datasets/dataset_nanoAODv12.yaml -v 12 -l April19_NanoV12_UpdatedMassCalib -y 2018 -m all)
```
