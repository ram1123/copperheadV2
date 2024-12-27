import dask_awkward as dak
import numpy as np
import awkward as ak
import glob
import pandas as pd


# def getParquetFiles(path):
    # return glob.glob(path)

def fillEventNans(events):
    """
    checked that this function is unnecssary for vbf category, but have it for robustness
    """
    for field in events.fields:
        if "phi" in field:
            events[field] = ak.fill_none(events[field], value=-10) # we're working on a DNN, so significant deviation may be warranted
        else: # for all other fields (this may need to be changed)
            events[field] = ak.fill_none(events[field], value=0)
    return events

# def replaceSidebandMass(events):
#     for field in events.fields:
#         if "phi" in field:
#             events[field] = ak.fill_none(events[field], value=-1)
#         else: # for all other fields (this may need to be changed)
#             events[field] = ak.fill_none(events[field], value=0)
#     return events

def applyCatAndFeatFilter(events, features: list, region="h-peak", category="vbf"):
    """
    
    """
    # apply category filter
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    
    if category.lower() == "vbf":
        btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
        cat_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
        cat_cut = cat_cut & (~btag_cut) # btag cut is for VH and ttH categories
    elif category.lower()== "ggh":
        btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
        cat_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5)
        cat_cut = cat_cut & (~btag_cut) # btag cut is for VH and ttH categories
    else: # no category cut is applied
        cat_cut = ak.ones_like(dimuon_mass, dtype="bool")
        
    cat_cut = ak.fill_none(cat_cut, value=False)
    cat_filter = (
        cat_cut & 
        region 
    )
    events = events[cat_filter] # apply the category filter
    # print(f"events dimuon_mass: {events.dimuon_mass.compute()}")
    # apply the feature filter (so the ak zip only contains features we are interested)
    print(f"features: {features}")
    events = ak.zip({field : events[field] for field in features}) 
    return events


def prepare_features(events, features, variation="nominal"):
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation
        
        if f"{trf}_{variation_current}" in events.fields:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in events.fields:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var

def preprocess_loop(events, features2load, region="h-peak", category="vbf", label=""):
    features2load = prepare_features(events, features2load) # add variation to features
    print(f"features2load: {features2load}")
    # features2load = training_features + ["event"]
    events = applyCatAndFeatFilter(events, features2load, region=region, category=category)
    events = fillEventNans(events)

    # turn to pandas df add label (signal=1, bkg=0)
    df = ak.to_dataframe(events.compute())
    if label== "signal":
        df["label"] = 1.0
    elif label== "background":
        df["label"] = 0.0
    else:
        print("Error: please define the label: signal or background")
        raise ValueError
    return df
    
# def scale_data(inputs, model_name: str, fold_idx: int):
#     x_mean = np.mean(x_train[inputs].values,axis=0)
#     x_std = np.std(x_train[inputs].values,axis=0)
#     training_data = (x_train[inputs]-x_mean)/x_std
#     validation_data = (x_val[inputs]-x_mean)/x_std
#     # np.save(f"output/trained_models/{model}/scalers_{fold_idx}", [x_mean, x_std])
#     np.save(f"dnn/trained_models/{model_name}/scalers_{fold_idx}", [x_mean, x_std])
#     return training_data, validation_data


def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    print(f"average.shape: {average.shape}")
    variance = np.average((values - average)**2, weights=weights, axis=0)
    print(f"variance.shape: {variance.shape}")
    return np.sqrt(variance)

def preprocess(base_path, region="h-peak", category="vbf"):
    # training_features = [
    #     "dimuon_mass",
    #     "dimuon_pt",
    #     "dimuon_pt_log",
    #     "dimuon_eta",
    #     # "dimuon_ebe_mass_res",
    #     # "dimuon_ebe_mass_res_rel",
    #     # "dimuon_cos_theta_cs",
    #     # "dimuon_phi_cs",
    #     "dimuon_pisa_mass_res",
    #     "dimuon_pisa_mass_res_rel",
    #     "dimuon_cos_theta_cs_pisa",
    #     "dimuon_phi_cs_pisa",
    #     "jet1_pt",
    #     "jet1_eta",
    #     "jet1_phi",
    #     "jet1_qgl",
    #     "jet2_pt",
    #     "jet2_eta",
    #     "jet2_phi",
    #     "jet2_qgl",
    #     "jj_mass",
    #     "jj_mass_log",
    #     "jj_dEta",
    #     "rpt",
    #     "ll_zstar_log",
    #     "mmj_min_dEta",
    #     "nsoftjets5",
        # "htsoft2",
        # "year",
    # ]
    training_features = [
        'dimuon_mass', 'dimuon_pt', 'dimuon_pt_log', 'dimuon_eta', \
         'dimuon_cos_theta_cs', 'dimuon_phi_cs',
         'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_qgl',\
         'jj_mass', 'jj_mass_log', 'jj_dEta', 'rpt', 'll_zstar_log', 'mmj_min_dEta', 'nsoftjets5', 'htsoft2'
    ]
    # TODO: add mixup
    # sig and bkg processes defined at line 1976 of AN-19-124. IDK why ggH is not included here
    sig_processes = ["vbf_powheg_dipole"]
    bkg_processes = ["dy_M-100To200", "ewk_lljj_mll105_160_ptj0","ttjets_dl","ttjets_sl"]
    
    filenames = []
    for process in sig_processes:
        filenames += glob.glob(f"{base_path}/{process}/*/*.parquet")
    # print(filenames)
    sig_events = dak.from_parquet(filenames)
    print(sig_events.fields)

    filenames = []
    for process in bkg_processes:
        filenames += glob.glob(f"{base_path}/{process}/*/*.parquet")
    # print(filenames)
    bkg_events = dak.from_parquet(filenames)

    
    
    training_features = prepare_features(sig_events, training_features) # add variation to features
    print(f"training_features: {training_features}")
    features2load = training_features + ["event","wgt_nominal"]

    loop_dict = {
        "signal" : sig_events,
        "background" : bkg_events,
    }
    df_l = []
    for label, events in loop_dict.items():
        df = preprocess_loop(events, features2load, region=region, category=category, label=label)
        # print(f"df: {df.head()}")
        print(f"df.label: {df.label.head()}")
        df_l.append(df)

    
    # merge sig and bkg dfs
    df_total = pd.concat(df_l)
    print(df_total)
    print(f"df_total.isnull().values.any(): {df_total.isnull().values.any()}")

    # divide our data into 4 folds
    nfolds = 4
    for i in range(nfolds):       
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Train classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")

        train_filter = df_total.event.mod(nfolds).isin(train_folds)
        val_filter = df_total.event.mod(nfolds).isin(val_folds)
        eval_filter = df_total.event.mod(nfolds).isin(eval_folds)

        # scale data, save the mean and std
        x_train = df_total[training_features].values[train_filter]
        wgt_train = df_total["wgt_nominal"].values[train_filter]
        x_mean = np.mean(x_train,axis=0)
        x_std = np.std(x_train,axis=0)
        model_name = "test"
        # np.save(f"output/trained_models/{model}/scalers_{fold_idx}", [x_mean, x_std])
        np.save(f"dnn/trained_models/{model_name}/scalers_{fold_idx}", [x_mean, x_std])

        # apply scaling to data, and save the data for training
        training_data = (x_train[inputs]-x_mean)/x_std
        
    
    
    # calculate the scale, save it
    # save the resulting df for training
    
    
if __name__ == "__main__":  
    from distributed import LocalCluster, Client
    cluster = LocalCluster(processes=True)
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    
    base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco/stage1_output/2018/f1_0/"
    
    preprocess(base_path)