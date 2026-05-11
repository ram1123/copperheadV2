---
title: Introduction
---

## Framework setup

```bash
git clone --recurse-submodules https://github.com/green-cabbage/copperheadV2.git
cd copperheadV2
git checkout main
# If already cloned the repo, then to update the submodules run:
git submodule update --remote --merge
```

Every time you open a new terminal session, enter one of the Pixi environments with:

```bash
./enter_pixi.sh default
```

Available environments are:

```bash
./enter_pixi.sh default
./enter_pixi.sh full
./enter_pixi.sh symbolic
./enter_pixi.sh combine
```

Use the environment that matches your task:

- `default`: standard analysis workflow
- `full`: larger analysis environment with extra packages
- `symbolic`: PySR / symbolic-regression work
- `combine`: statistical inference / Combine workflow

To create the dask client, open the jupyter notebook [DaskGatewaySLURM.ipynb](../DaskGatewaySLURM.ipynb) and run cells upto section "Create the gateway" to create the dask client.

**NOTE**: *Make sure when you don't need the dask client, you close it from the notebook to free up the resources. For this run the cells under section "Delete the gateway".*

## Running the code

### Obtain the reduced ntuples

1. **Pre-Processing**: Just prepares the JSON file having all the root files belongs to a particular sample (DAS name), along with its metadata like total number of events, etc.
2. **Stage-1**: This step applies the pre-selection, corrections, scale-factors, etc.

Pre-stage reads the dataset information from the YAML file and saves the root files to read in next step with its metadata in a JSON file.

```bash
bash run_analysis_pipeline.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 0
```
where
- `-v`: nanoAOD version
- `-c`: path to the dataset YAML file that contains the list of samples to be processed
- `-l`: label for the output ntuple files
- `-y`: year of data-taking
- `-m`: run mode. `0` for pre-processing, `1` for stage-1 processing.

Legacy note:

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 0
```


Run the stage1 to skim the data. It also saves the weight for Z-pT reweighting, and and all other necessary weights for the analysis.

```bash
bash run_analysis_pipeline.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 1
```

Legacy note:

```bash
bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 1
```

### Run the VBF stats pipeline

After `stage3` has produced the datacards, use the stats driver for VBF statistical workflows.

Typical modes include:

- `-m 4`: copy datacards
- `-m 5`: build combined VBF cards and workspaces
- `-m 6`: run significance
- `-m 7`: run impacts
- `-m 8`: run likelihood scan
- `-m 9`: run the full VBF combine chain
- `-m 10`: collect significance summaries
- `-m 11`: run the stage2/stage3/stats limit chain

Example:

```bash
./enter_pixi.sh combine
bash run_stats_pipeline.sh -m 9 -y Run3 -l label_for_ntuple
```

Legacy note:

```bash
bash stage1_loop_Improved.sh -m 9 -y Run3 -l label_for_ntuple
```

### Get the yields 


1. To get the yields from the reduced ntuples, use the script [get_yields.py](../get_yields.py).

    ```bash
    python scripts/get_yields.py --years 2018
    ```
    Note that this file reads ntuples from the YAML file [configs/trials.yml](../configs/trials.yml). Update the path under the key `current` to point the label with with the reduced ntuples are created.

### Sync/Compare with previous results

1. To compare the yields with previous results, use the script [sync_parquet_dimuon.py](../sync_parquet_dimuon.py).

    ```bash
    python scripts/sync_parquet_dimuon.py DIR1 DIR2 -o diff.csv
    ```

### Control Plots

Before running the below code make sure to update the input and output paths and several other parameters in the [run_plotter.py](../run_plotter.py) file.

The main code for plotting is in the [plotter/validation_plotter_unified.py](../plotter/validation_plotter_unified.py) file. In this file you may need to update the list of datasets to be considered for different processes. You can see them here [validation_plotter_unified.py#L28-L83](https://github.com/ram1123/copperheadV2/blob/2cdaf09321000a8eb4a5eb8faf06e22f5e9ec560/plotter/validation_plotter_unified.py#L28-L83)

```bash
python run_plotter.py
```
