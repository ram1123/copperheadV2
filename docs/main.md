**Table of Contents**
=================

1. [Running the framework](#running-the-framework)
    1. [Framework setup](#framework-setup)
    2. [Running the code](#running-the-code)
        1. [Obtain the reduced ntuples](#obtain-the-reduced-ntuples)
        2. [Control Plots](#control-plots)
    3. [DY pt mismodelling correction](#dy-pt-mismodelling-correction)
        1. [Z-pT reweighting - validation](#z-pt-reweighting---validation)
    4. [Improving dimuon mass resolution](#improving-dimuon-mass-resolution)
        1. [Update](#update)




--------------

# Running the framework

## Framework setup

```bash
git clone https://github.com/ram1123/copperheadV2.git
cd copperheadV2
git checkout main
```

Everytime you open a new terminal session, run the following command to setup the environment variables:

```bash
source setup_env.sh
```

To create the dask client, open the jupyter notebook [DaskGatewaySLURM.ipynb](DaskGatewaySLURM.ipynb) and run cells upto section "Create the gateway" to create the dask client.

**NOTE**: Make sure when you don't need the dask client, you close it from the notebook to free up the resources. For this run the cells under section "Delete the gateway".

## Running the code

### Obtain the reduced ntuples

1. **Pre-Processing**: Just prepares the JSON file having all the root files belongs to a particular sample (DAS name), along with its metadata like total number of events, etc.
2. **Stage-1**: This step applies the pre-selection, corrections, scale-factors, etc.

Pre-stage reads the dataset information from the YAML file and saves the root files to read in next step with its metadata in a JSON file.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 0
```
where
- `-v`: nanoAOD version
- `-c`: path to the dataset YAML file that contains the list of samples to be processed
- `-l`: label for the output ntuple files
- `-y`: year of data-taking
- `-m`: run mode. `0` for pre-processing, `1` for stage-1 processing.


Run the stage1 to skim the data. It also saves the weight for Z-pT reweighting, and and all other necessary weights for the analysis.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 1
```

### Get the yields and compare with previous results


1. To get the yields from the reduced ntuples, use the script [get_yields.py](get_yields.py).

    ```bash
    python get_yields.py --years 2018
    ```
    Note that this file reads ntuples from the YAML file [configs/trials.yml](configs/trials.yml). Update the path under the key `current` to point the label with with the reduced ntuples are created.

2. To compare the yields with previous results, use the script [sync_parquet_dimuon.py](sync_parquet_dimuon.py).

    ```bash
    python sync_parquet_dimuon.py --years 2018 --ref_label previous_label --curr_label current_label
    ```

### Control Plots

Before running the below code make sure to update the input and output paths and several other parameters in the [run_plotter.py](run_plotter.py) file.

The main code for plotting is in the [plotter/validation_plotter_unified.py](plotter/validation_plotter_unified.py) file. In this file you may need to update the list of datasets to be considered for different processes. You can see them here [validation_plotter_unified.py#L28-L83](https://github.com/ram1123/copperheadV2/blob/2cdaf09321000a8eb4a5eb8faf06e22f5e9ec560/plotter/validation_plotter_unified.py#L28-L83)

```bash
python run_plotter.py
```


## DY pt mismodelling correction

Seveal steps are needed to get the Z-pT weights.

**Step-1**: Obtain the histograms of Z-pT in data and MC.
```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m "zpt_fit0" -n 0
```

**Step-2**: After several checks we decided that to fit the ratio of Z-pT distribution in data and MC using the three functions in low, medium and high pT range. So, in this step we fit the ratio histograms and find the polynomial degrees that fits best in each range.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m "zpt_fit1" -n 0
```

Where the option:
- `-n`: specifies which jet bin to consider for fitting. `0` for zero jet bin, `1` for one jet bin and `2` for greater than equal to two jet bin.


**Step-3**: Now we calculate the Z-pT weights using the fit functions obtained in the previous step.

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018  -m "zpt_fit2" -n 0
```

For details check the detailed documentation: [Z-pT reweighting](ZpT_reweight.md)


### Improving dimuon mass resolution


- Run the script [`getCalibrationFactor.py`](../src/lib/ebeMassResCalibration/getCalibrationFactor.py) to get the calibration factor for the EBE mass resolution.
- The script will generate a JSON file with the calibration factors for different categories.
- Copy the generated JSON file to the path `data/res_calib/` and update the path and name of this JSON file in the config file: `configs/parameters/correction_filelist.yaml`.

#### How to run

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

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -m "calib"
```

- To adjust the fitting one can change the parameters in the script `src/lib/ebeMassResCalibration/ebeMassResPlotter.py`

### Dec 2025 Updated commands

```bash
time(python src/lib/ebeMassResCalibration/getCalibrationFactor.py  --years "2018")
```

### Update

- New code: `src/lib/ebeMassResCalibration/getCalibrationFactor_Improved.py`
   - Just need to update the path of the input files and it should work.
   - Once we get the json file, copy it to the path `data/res_calib/` and update the path and name of this json file in the config file: `configs/parameters/correction_filelist.yaml`
   - Then re-run stage-1 to get the updated mass calibration. **REMEMBER TO SWITCH ON THE BSC OPTION**.
- For validation use the jupyter notebook: `src/lib/ebeMassResCalibration/closure_test.ipynb`

# Other details

1. [Custom nanoAOD production](NanoAODv12_v9.md)
1. [Check files for LZMA compression errors or corrupted files (from nanoAOD production)](FailedJobsCheck.md)
1. [Z-pT weight calculation](ZpT_reweight.md)
1. [Event by event mass resolution: Basic idea](eventbyeventMassResolution.md)
    1. [Technical details](EBE_technicalDetails.md)
1. [Plotting codes](PlottingCodes.md)
1. [Main framework - Basic idea](mainFramework.md)
    1. [Technical details](mainFramework_technicalDetails.md)
1. [Get yield](GetYields.md)
