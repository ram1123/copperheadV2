---
title: Introduction
---

## Framework setup

```bash
git clone --recurse-submodules https://github.com/ram1123/copperheadV2.git
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
./enter_pixi.sh combine
```

Use the environment that matches your task:

- `default`: standard analysis workflow, ROOT, workflow tools, ML, and symbolic regression
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

Name the label (`-l`) as

```
Run3_nanoAODv<version>_<date>_<additional_info>
```

for example `Run3_nanoAODv12_10Feb_FilterJetsHorn30GeV`. The label becomes the output directory
name, and every later stage (compact, stage-2/3, plots) finds the ntuples through it.

??? note "Legacy: `stage1_loop_Improved.sh`"
    ```bash
    bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 0
    ```


Run the stage1 to skim the data. It also saves the weight for Z-pT reweighting, and and all other necessary weights for the analysis.

```bash
bash run_analysis_pipeline.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 1
```

??? note "Legacy: `stage1_loop_Improved.sh`"
    ```bash
    bash stage1_loop_Improved.sh -v 12 -c configs/datasets/dataset_nanoAODv12.yaml -l label_for_ntuple -y 2018 -m 1
    ```

### Run the VBF stats pipeline

After `stage3` has produced the datacards, run the Combine chain with `run_stats_pipeline_VBF.sh` in
the `combine` environment. See [Statistics](Statistics.md) for the modes and examples.

### Get the yields

Use [get_yields.py](../scripts/get_yields.py) on the `stage1_output` directory of your label; see
[Get Yields](GetYields.md) for the command, options and how the categories are split.

### Sync/Compare with previous results

1. To compare the yields with previous results, use the script [sync_parquet_dimuon.py](../scripts/sync_parquet_dimuon.py).

    ```bash
    python scripts/sync_parquet_dimuon.py DIR1 DIR2 -o diff.csv
    ```

2. The sync txt schema is defined by `SYNCVARLIST` in
   [scripts/sync_parquet_dimuon.py](../scripts/sync_parquet_dimuon.py). Along with kinematics, it
   includes `wgt_nominal` and selected partial-weight branches such as
   `separate_wgt_btag`, which is useful when a nominal-weight mismatch comes from the b-tag
   scale factor.

3. To run the stage-1 sync check locally (the same steps the CI `sync-stage1` workflow runs, for
   `2017` or `2022preEE`):

    ```bash
    YEAR=2022preEE
    OUT=test/output/label_output/stage1_output

    # STEP-1: stage-1 on the sync sample, with the frozen reference switches CI uses
    bash run_analysis_pipeline.sh -c configs/datasets/sync_dataset_nanoAODv12.yaml -v 12 \
        -l label_output -y "${YEAR}" -m 1 -z -Z \
        -w test/reference/switches_official.yaml -S test/output

    # STEP-2: dump the event kinematics to sync text files
    python scripts/sync_parquet_dimuon.py ${OUT}/${YEAR}/f1_0/data_*/ -o ${OUT}/${YEAR}_data_eventKinematics.txt
    python scripts/sync_parquet_dimuon.py ${OUT}/${YEAR}/f1_0/dy*/0/ -o ${OUT}/${YEAR}_dy_eventKinematics.txt
    python scripts/sync_parquet_dimuon.py ${OUT}/${YEAR}/f1_0/vbf_powheg_dipole/0/ -o ${OUT}/${YEAR}_vbf_eventKinematics.txt

    # STEP-3: compare against the references
    for s in data dy vbf; do
        python scripts/sync_parquet_dimuon.py test/reference/${YEAR}_${s}_eventKinematics.txt \
            ${OUT}/${YEAR}_${s}_eventKinematics.txt -o diff_${s}_${YEAR}.csv
    done
    ```

    STEP-4: every `diff_*.csv` must be empty. A non-empty file means the change altered stage-1
    output: fix it, or, if the change is intentional, regenerate the references (next item).

4. To regenerate the sync reference txt files used by GitHub Actions, run:

    ```bash
    bash scripts/update_sync_references.sh "2017,2022preEE" --use-reference-switches
    ```

    For a single year:

    ```bash
    bash scripts/update_sync_references.sh 2017 --use-reference-switches
    ```

    This reruns the sync stage-1 samples, rebuilds the `*_eventKinematics.txt` files, and updates
    the copies in `test/reference`.

    **Always pass `--use-reference-switches`.** `.github/workflows/sync-stage1.yml` unconditionally
    overwrites `configs/switches/switches_official.yaml` with `test/reference/switches_official.yaml` before running
    stage-1 -- CI never runs the sync sample against your local/production switches. Omitting this
    flag runs against production `configs/switches/switches_official.yaml` instead, which can silently
    diverge from what CI actually produces (e.g. `do_remove_dy_M100to200`, `do_save_partial_weights`)
    and generate reference files that don't match what CI itself would compare against on the next
    run. `test/reference/switches_official.yaml` is a separately-maintained, intentionally-frozen snapshot
    (see `docs/known_issues.md`) -- if a code change adds a new required switch key, add it there too
    (matching whatever value reproduces the pre-change behavior) or this command will fail with a
    `KeyError`.

### Control Plots

Data/MC control plots are made with [validation_plotter_unified.py](../plotter/validation_plotter_unified.py),
either directly or through the `run_plotter.py` wrapper; see [Plotting Codes](PlottingCodes.md).
