import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from dnn_helper import DIR_TAG

YEARS = ["2016preVFP", "2016postVFP", "2017", "2018"]
LABEL = "Run2_nanoAODv12_28Nov_HEMVetoFix_NoSyst_V2"
REGION = "h-peak"
CATEGORY = "vbf"
N_FOLDS = 4

in_dirs = [
    f"dnn/trained_models/{LABEL}/{year}_{REGION}_{CATEGORY}{DIR_TAG}"
    for year in YEARS
]
out_dir = f"dnn/trained_models/{LABEL}/run2_{REGION}_{CATEGORY}{DIR_TAG}"
Path(out_dir).mkdir(parents=True, exist_ok=True)

# use training_features from the first year
# with open(os.path.join(in_dirs[0], "training_features.pkl"), "rb") as f:
#     training_features = pickle.load(f)


training_features = [
    "dimuon_mass",
    "dimuon_ebe_mass_res",
    "dimuon_ebe_mass_res_rel",
    "jj_mass_nominal",
    "jj_mass_log_nominal",
    "rpt_nominal",
    "ll_zstar_log_nominal",
    "jj_dEta_nominal",
    "nsoftjets5_nominal",
    "mmj_min_dEta_nominal",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_rapidity",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jet1_qgl_nominal",
    "jet2_qgl_nominal",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "htsoft2_nominal",
    "pt_centrality_nominal",
    "year",
]

for i in range(N_FOLDS):
    dfs_train = []
    dfs_val = []
    dfs_eval = []
    for d in in_dirs:
        dfs_train.append(pd.read_parquet(os.path.join(d, f"data_df_train_{i}.parquet")))
        dfs_val.append(pd.read_parquet(os.path.join(d, f"data_df_validation_{i}.parquet")))
        dfs_eval.append(pd.read_parquet(os.path.join(d, f"data_df_evaluation_{i}.parquet")))

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_val   = pd.concat(dfs_val,   ignore_index=True)
    df_eval  = pd.concat(dfs_eval,  ignore_index=True)

    # recompute scalers on combined train, same logic as in preprocessor
    from dnn_preprocessor import weighted_std  # or reimplement here

    x_train = df_train[training_features].values
    wgt_train = df_train.wgt_nominal.values

    x_mean = np.average(x_train, axis=0, weights=wgt_train)
    x_std  = weighted_std(x_train, wgt_train)
    x_std = np.where(np.isclose(x_std, 0.0), 1.0, x_std)

    np.save(os.path.join(out_dir, f"scalers_{i}"), [x_mean, x_std])

    # rescale and save new combined parquet
    df_train[training_features] = (df_train[training_features] - x_mean) / x_std
    df_val[training_features]   = (df_val[training_features]   - x_mean) / x_std
    df_eval[training_features]  = (df_eval[training_features]  - x_mean) / x_std

    df_train.to_parquet(os.path.join(out_dir, f"data_df_train_{i}.parquet"))
    df_val.to_parquet(os.path.join(out_dir, f"data_df_validation_{i}.parquet"))
    df_eval.to_parquet(os.path.join(out_dir, f"data_df_evaluation_{i}.parquet"))

# also copy training_features.pkl
with open(os.path.join(out_dir, "training_features.pkl"), "wb") as f:
    pickle.dump(training_features, f)
