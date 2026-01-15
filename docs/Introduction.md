---
title: Introduction
---


## Framework setup

```bash
git clone https://github.com/ram1123/copperheadV2.git
cd copperheadV2
git checkout main
```

Everytime you open a new terminal session, run the following command to setup the environment variables:

```bash
source setup_env.sh
# default conda environment is `coffea_latest`.
source setup_env.sh yun
# yun is for `yun_coffea_latest`
```

To create the dask client, open the jupyter notebook [DaskGatewaySLURM.ipynb](DaskGatewaySLURM.ipynb) and run cells upto section "Create the gateway" to create the dask client.

**NOTE**: *Make sure when you don't need the dask client, you close it from the notebook to free up the resources. For this run the cells under section "Delete the gateway".*

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


