title: DNN for VBF
---



# Input Features for DNN in VBF Analysis

The following features are used as input to the Deep Neural Network (DNN) for the Vector Boson Fusion (VBF) analysis:

- **Dimuon kinematics and angles**
  - `dimuon_mass`
  - `dimuon_pt`, `dimuon_pt_log`
  - `dimuon_rapidity`
  - `dimuon_cos_theta_cs`, `dimuon_phi_cs`

- **Event-by-event mass resolution**
  - `dimuon_ebe_mass_res`
  - `dimuon_ebe_mass_res_rel`

- **Jet kinematics**
  - Leading jet: `jet1_pt`, `jet1_eta`, `jet1_phi`
  - Subleading jet: `jet2_pt`, `jet2_eta`, `jet2_phi`

- **Jet flavor/shape information**
  - `jet1_qgl`, `jet2_qgl`

- **Dijet system**
  - `jj_mass`, `jj_mass_log`
  - `jj_dEta`

- **Additional event activity**
  - `htsoft2`
  - `nsoftjets5`

- **VBF topology variables**
  - `rpt`
  - `ll_zstar_log`
  - `mmj_min_dEta`
  - `pt_centrality`

- **Data-taking period**
  - `year`


# DNN Training

## Preprocessing Steps

### Preprocessing validations

1. Feature distributions before and after preprocessing
1. Correlation matrices before and after preprocessing
1. Plot all input features after preprocessing, from the output parquet files
    1. Add mean and stddev values to the plots

For this, there are two scripts available in the `plotter/` directory:

1. [plot_vbfdnn_input_features_compare.py](../plotter/plot_vbfdnn_input_features_compare.py): to compare feature distributions between signal and background samples before preprocessing.
1. [`dnn_preprocessing_validation.py`](../plotter/dnn_preprocessing_validation.py): to validate the preprocessing steps by plotting the feature distributions from the preprocessed parquet files. For these plots, the mean should be around 0 and the standard deviation around 1. **Note:** the variable `year`  and `nsoftjets5` are not standardized.

## DNN Training

### DNN Hyperparameter Optimization

### DNN Training Command
