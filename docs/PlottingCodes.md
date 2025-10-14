# Data/MC comparison plots

- Available code:  [validation_plotter_unified.py](../plotter/validation_plotter_unified.py)

- Few important options:
    - `--remove_zpt_weights`: Removes Z pT weights from the plots.
    - `-cat <CATEGORY>`: Category of the plot. Options include `nocat`, `ggh` and `vbf`.
    - `-reg <REGION>`: Region of the plot. Options include `z_peak`, `h_sideband`, `signal`.
    - `--njets <N_JETS>`: Number of jets to consider. Default is "inclusive" (no jet requirement). Available options are: 0, 1 and 2. If you use 2 then it will plot >= 2 jets.

command to run:
```bash
python validation_plotter_unified.py -y 2018 --save_path "./validation/figs/May28_NanoV12_June04/" --load "/depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/"   -cat "nocat" -reg "z_peak" --remove_zpt_weights --njets 0
```

# Compare inclusive DY with VBF filtered DY

- Available code:  [compare_inclusiveDY_DYvbfFilter.py](../plotter/compare_inclusiveDY_DYvbfFilter.py)

command to run:
```bash
time python compare_dy_parquet_v2.py         --dirs1 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-50_MiNNLO/ /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_M-100To200_MiNNLO/         --dirs2 /depot/cms/users/shar1172/hmm/copperheadV1clean/May28_NanoV12/stage1_output/2018/f1_0/dy_VBF_filter/             --nbins  110         --xmin 105         --xmax 160         --output compareDY_M50M100_110bins.pdf
```

# Plot the DNN score for VBF using the stage-2 output

- Available code:  [plot_dnn_score.py](../plotter/plot_dnn_score.py)

command to run:

```bash
bash run_script_dnn_score_plot.sh
```

***NOTE***: In the script currently there is hardcoded path: `load_path`.
