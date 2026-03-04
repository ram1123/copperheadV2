---
title: ggH after step-1
---

# Train BDT

1. Go to path

    ```bash
    cd MVA_training/ggH_BDT/
    ```

2. Update the parameters in file [train.sh](train.sh). For example: `year`, `label` and `input path of stage-1`

    bash train.sh

3. Then go back to base dir, and follow further steps

4. Run the `stage2_sh.sh` script.

    ```bash
    cd $WORKDIR
    bash stage2_sh.sh
    ```

    Inside this script (`stage2_sh.sh`), run the python code `run_stage2.py`.

    NOTE: To run, we need to update several hardcoded paths in `stage2_sh.sh`, `run_stage2.py` and `MVA_functions.py`.

    This step applies the selection and add the branch having the BDT score in the output parquet file.

5. Obtain the target yield:

    In the script `run_script.sh`, update label, path, model name, and year then run.

    ```bash
    cd stage2/ggH/score_edge_generation/
    bash run_script.sh
    ```

    This will print the **target yield** on the terminal. Then manually write the target yield to file [../stage2/ggH/target_yields.yaml](../stage2/ggH/target_yields.yaml), in the ouput path.

    The target yield printed on the terminal should be like:

    `[0.16 0.47 0.94 0.99]`

    This is the cumulative. But we need to add 

    ```yaml
    target_yields:  # Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV_x_Run3PrelimResultsFeb09_2026_jecjer_ggh_perYr
    # [0.16 0.47 0.94 0.99]
    - 0.16
    - 0.31  # 47-16
    - 0.47  # 94-47
    - 0.05  # 99-94    
    ```


6. Obtain the BDT edges, based on target yield. For this go back to the script: `stage2_sh.sh`

    Keep uncommented line that belongs to `calculate_score_edges.py`. And other parameters should be same as you set it in step-4.

    ```bash
    bash stage2_sh.sh
    ```

    This updates the file BDT edges in the file [configs/MVA/ggH/BDT_edges.yaml](../configs/MVA/ggH/BDT_edges.yaml), for the year that we ran.

7. Now, run the script `run_stage2.py` again from `stage2_sh.sh`.

    Keep uncommented line that belongs to `run_stage2.py`, rest commented. And other parameters should be same as you set it in step-4.

8. Obtain the data/mc for BDT score run [validation/ggH/categorization/run_script.sh](../validation/ggH/categorization/run_script.sh)

    Inside this script uncommented line should belongs to code  `validation_plot.py`.

    ```bash
    cd validation/ggH/categorization/
    bash run_script.sh
    ```

    Note that the

    ```json
    "BDT_score": {
        "variable": "BDT_score",
        "binning_linspace": [-1, 1, 51],
        "density": false,
        "ylabel": "Events / 1 GeV",
        "xlabel": "BDT Score",
        "color": "royalblue"
    },   
    ```    


9. Obtain the workspaces:

    Use the script `stage2_sh.sh`. Only uncommented line should be `run_stage3.py`

    ```bash
    # use the conda env.
    source setup_env.sh
    bash stage2_sh.sh
    ```

    workspaces will be saved in `validation/stage3/2022postEE/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2_X_Run3_03March_test_perYr/workspaces`

10. Obtain the datacard

    ```bash
    cd stage2/ggH_datacard/
    bash run_script.sh
    ```
    Remove the JES mentions from the script and then it works.


11. Now obtain the repo for significance extraction: 

    ```bash
    git clone git@github.com:ram1123/HMuMuCombine.git
    cd HMuMuCombine
    ```

    Copy the datacard generated in step-10 (`stage2/ggH_datacard/datacards/all/*`), to the path `HMuMuCombine/Significance` inside the repo `HMuMuCombine`.


    ```bash
    mkdir my_workspace
    # copy workspaces to this dir
    # create dir my_workspace
    cp validation/stage3/2022postEE/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2_X_Run3_04March_test_perYr/workspaces/*.root  /work/users/shar1172/HMuMuCombine/Significance/Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2_X_Run3_04March_test_perYr/my_workspace/

    #Then run the script
    bash get_significance.sh
    ```


