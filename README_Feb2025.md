# 🐍 Copperhead V2, - Columnar Parallel Pythonic framEwork for Run3 H&rarr;µµ Decay search

## setup

```bash
git clone https://github.com/green-cabbage/copperheadV2.git
cd copperheadV2
source setup_env.sh
# Run first two column of DaskGatewaySLURM.ipynb to start the DASK.
```

1. Run the pre-stage to get the dataset information.
   ```bash
   bash stage1_loop.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m 0
   ```
2. Run the stage1 to skim the data. It also saves the weight for Z-pT reweighting.
   ```bash
   bash stage1_loop.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m 1
   ```

## Per-event mass calibration

```bash
bash stage1_loop.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m "calib"
```

- To adjust the fitting one can change the parameters in the script `src/lib/ebeMassResCalibration/ebeMassResPlotter.py`

### Update

- New code: `src/lib/ebeMassResCalibration/getCalibrationFactor_Improved.py`
   - Just need to update the path of the input files and it should work.
   - Once we get the json file, copy it to the path `data/res_calib/` and update the path and name of this json file in the config file: `configs/parameters/correction_filelist.yaml`
   - Then re-run stage-1 to get the updated mass calibration. **REMEMBER TO SWITCH ON THE BSC OPTION**.
- For validation use the jupyter notebook: `src/lib/ebeMassResCalibration/closure_test.ipynb`

## Z-pT reweighting

```bash
bash stage1_loop.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m "zpt"
```

### Z-pT reweighting - validation

```bash
bash stage1_loop.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m "zpt_val"
```

## Step - 1:

***Summary*** :

### How to run

```bash
bash stage1_loop.sh <StageNo>
```

To run pre-stage the `StageNo` should be "0". For running `Stage1` the argument should be 1.

### Improvements

1. Remove "dummy" from the yaml file. Instead add the `/store` path. And add in the code that if the dataset name starts from `/store` then fetch all root files from that path instead of querying using `dasgoclient`.


## Step - 1: Skim, $Z_{p_T}$ correction


## Step - 2: Get Z-pT reweight

1. Get weight, data/DY, in the jet multiplicity bins
   * Code located in `data/Zpt_rewgt/fitting/do_fitting.py`
   * It extracts p_T(mumu) from data and DY in the Z-peak region. Also, the ratio of data/dy in nJet bins. Then save them as .root file
2. Fit the ratio data/dy: `do_f_test.py`
   * From here, get the polynomial that fits our data best as per f-test.
3. Run `get_goodnessofFit.py`: Fits and saves the polynomial info in the YAML file.
4. How to save the weight into the skimmed file:
   - Run: `work/users/shar1172/HMuMu/copperheadV2/src/copperhead_processor.py`
   - The function that saves weight in above script is `getZptWgts()`


# Run on the Hammer

## Setup to run on the Hammer

```bash
source /etc/profile.d/modules.sh
module --force purge
module load anaconda/2020.11
source setup_env.sh
```

## Run the script

```bash
python Scripts/Investigate_ParquetFile.py
```



# Improvements

- [ ] Update how the pre-stage JSON files are saved. It should be saved with year name, so that we don't need to run pre-stage everytime.
    - [ ] Also, if we already run for data and running for MC then it should append info to the JSON file.
- [ ]


