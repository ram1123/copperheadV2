import dask_awkward as dak
import numpy as np
import awkward as ak
import glob
import pandas as pd
import itertools
import argparse

import os 
import copy
import pickle

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
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
        cat_cut = vbf_cut & (~btag_cut) # btag cut is for VH and ttH categories
    elif category.lower()== "ggh":
        btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5)
        cat_cut = (~vbf_cut) & (~btag_cut) # btag cut is for VH and ttH categories
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
    # events = applyCatAndFeatFilter(events, features2load, region=region, category="ggh")
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
    # print(f"average.shape: {average.shape}")
    variance = np.average((values - average)**2, weights=weights, axis=0)
    # print(f"variance.shape: {variance.shape}")
    return np.sqrt(variance)

# def mixup(x_train, label_train):
#     """
#     apply cartesian product on x_train then apply mixup
#     source: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-111.php
#     """
#     x=x_train
#     y=x_train
#     # Using np.tile and np.repeat to create a grid of repeated elements from 'x' and 'y'
#     # The grid is created by replicating 'x' along rows and 'y' along columns
#     cartesian_prod_x = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

#     # do the same for label
#     x=label_train
#     y=label_train
#     cartesian_prod_label = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
#     # print(cartesian_prod)
#     frac = 0.5
#     x_train_mixup = frac*cartesian_prod[:,0] + (1-frac)*cartesian_prod[:,1]
#     # print(x_train_mixup)
#     return x_train_mixup


# def applyMixup(x_train,label_train):
#     chunks = np.array_split(large_array, num_chunks)
#     # 

# def applyMixup(x_train):
#     """
#     apply cartesian product on x_train then apply mixup
#     source: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-111.php
#     """
#     # Compute all combinations of these arrays

#     combinations = list(itertools.product(x_train, x_train))
#     print("combination done")
#     result =np.array(combinations)
#     frac = 0.5
#     x_train_mixup = frac*result[:,0] + (1-frac)*result[:,1]
#         # print(x_train_mixup)
#     return x_train_mixup

"""mixup code start. credits to https://github.com/makeyourownmaker/mixupy """


import sys
import random
import numpy as np
import pandas as pd


def mixup(data, alpha=4, concat=False, batch_size=None, seed=1352):
    """
    Create convex combinations of pairs of examples and their labels
    for data augmentation and regularisation

    This function enlarges training sets using linear interpolations of
    features and associated labels as described in
    https://arxiv.org/abs/1710.09412.

    The data must be numeric.  Non-finite values are not permitted.
    Factors should be one-hot encoded.  Duplicate values will not
    be removed.

    For now, only binary classification is supported.  Meaning the y
    coloumn must contain only numeric 0 and 1 values.

    Alpha values must be greater than or equal to zero.  Alpha equal to
    zero specifies no interpolation.

    The mixup function returns a pandas dataframe containing interpolated
    x and y values.  Optionally, the original values can be concatenated
    with the new values.

    Parameters
    __________
    data : pandas dataframe
      Original features and labels
    alpha : float, optional
      Hyperparameter specifying strength of interpolation
    concat : bool, optional
      Concatenate mixup data with original data
    batch_size : int, optional
      How many mixup values to produce

    Returns
    _______
    A pandas dataframe containing interpolated x and y values and
    optionally the original values

    Examples
    ________
    >>> data_mix = mixup(data, 'y')

    See also
    ________
    https://github.com/makeyourownmaker/mixupy
    """
    random.seed(seed) 
    np.random.seed(seed)
    
    _check_data(data)
    _check_params(alpha, concat, batch_size)

    data_len = data.shape[0]

    if batch_size is None:
        batch_size = data_len

    # Used to shuffle data2
    if batch_size <= data_len:
        # no replacement
        # index = random.sample(range(0, data_len), batch_size)
        index1 = random.sample(range(0, data_len), batch_size)
        index2 = random.sample(range(0, data_len), batch_size)
        # print(f"mixup index with no replacement: {index1}")
        # print(f"mixup index with no replacement: {index2}")
    else:
        # with replacement
        # index = np.random.randint(0, data_len, size=batch_size)
        index1 = np.random.randint(0, data_len, size=batch_size)
        index2 = np.random.randint(0, data_len, size=batch_size)
        # print(f"mixup index with replacement: {index1}")
        # print(f"mixup index with replacement: {index2}")


    # data = data.sample(frac=1)
    data_orig = data

    # print(f"data_orig: {data_orig}")
    # Cut data into specified size
    # data1 = resize_data(data, batch_size).reset_index(drop=True)
    data1 = data_orig.iloc[index1]
    data1 = data1.reset_index(drop=True)

    # data2 = data1.loc[index]
    data2 = data_orig.iloc[index2]
    data2 = data2.reset_index(drop=True)

    # x <- lam * x1 + (1. - lam) * x2
    # y <- lam * y1 + (1. - lam) * y2
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1))
    # lam = 0.5
    data_mix = lam * data1 + (1.0 - lam) * data2
    if data_mix.isna().any().any():
        print("Error: NaN values encountered!")
        raise ValueError
    
    data_new = data_mix

    if concat is True:
        data_new = pd.concat([data_orig, data_mix])

    # print(f"data1: {data1.head()}")
    # print(f"data2: {data2.head()}")
    # print(f"data_mix: {data_mix.head()}")
    return data_new

def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def resize_data(data, batch_size):
    """Resize data by repeating/removing rows"""

    data_orig = data
    data_len = data.shape[0]

    if data_len < batch_size:
        rep_times = batch_size // data_len

        for _ in range(rep_times):
            data = pd.concat([data, data_orig])

        data = data.reset_index(drop=True)

    if data_len < batch_size:
        data = data.loc[: batch_size - 1, :]
    else:
        # data = data.loc[: int(batch_size), :]
        data = data.iloc[: int(batch_size)]
    return data


def printe(errmsg):
    """Print error message and exit"""

    print(errmsg)
    sys.exit(1)


def _check_data_is_numeric(data):
    """Check data is numeric (int or float)"""

    # numerics = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
    numerics = data.shape[1] == data.select_dtypes(include=np.number).shape[1]

    if numerics is False:
        errmsg = (
            "Values must be numeric in 'data':\n"
            + " non-numeric values found\n"
            + str(data.dtypes)
        )
        printe(errmsg)

    return 0


def _check_data_is_finite(data):
    """Check data is finite - no NAs and no infs"""

    errmsg = "Values must be finite in 'data':\n"
    nas = pd.isna(data).sum()

    if np.sum(nas) > 0:
        errmsg += " 'na's found at \n" + str(nas)
        printe(errmsg)

    # infs = np.isinf(data).sum()
    infs = np.isinf(data.select_dtypes(include=np.number)).sum()

    if np.sum(infs) > 0:
        errmsg += " 'inf's found at\n" + str(infs)
        printe(errmsg)

    return 0


def _check_data(data):

    if not isinstance(data, pd.DataFrame):
        errmsg = "'data' must be pandas dataframe.\n" + "  'data' is ", type(data), "\n"
        printe(errmsg)

    if data.shape[0] < 2:
        errmsg = (
            "'data' must have 2 or more rows.\n" + "  'data' has ",
            data.shape[0],
            " rows.\n",
        )
        printe(errmsg)

    if data.shape[1] < 2:
        errmsg = (
            "'data' must have 2 or more columns.\n" + "  'data' has ",
            data.shape[1],
            " columns.\n",
        )
        printe(errmsg)

    _check_data_is_numeric(data)
    _check_data_is_finite(data)

    return 0


def _check_params(alpha, concat, batch_size):

    if not isinstance(alpha, (int, float)):
        errmsg = "'alpha' must be integer or float\n" + "  'alpha' is ", alpha, "\n"
        printe(errmsg)

    if alpha < 0:
        errmsg = (
            "'alpha' must be greater than or equal to 0.\n" + "  'alpha' is ",
            alpha,
            "\n",
        )
        printe(errmsg)

    if not isinstance(concat, bool):
        errmsg = "'concat' must be True or False:\n" + "  'concat' is ", concat, "\n"
        printe(errmsg)

    if batch_size is not None and not isinstance(batch_size, int):
        errmsg = (
            "'batch_size' must be an integer\n" + "  'batch_size' is ",
            batch_size,
            "\n",
        )
        printe(errmsg)

    if batch_size is not None and batch_size <= 0:
        errmsg = (
            "'batch_size' must be greater than 0.\n" + "  'batch_size' is ",
            batch_size,
            "\n",
        )
        printe(errmsg)

    return 0

"""mixup code end """




def preprocess(base_path, region="h-peak", category="vbf", do_mixup=False, run_label="test"):
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
    # generate directory to save training_features
    save_path = f"dnn/trained_models/{run_label}"
    if not os.path.exists(save_path): 
        os.makedirs(save_path) 

    

    # Create a list
    my_list = ['apple', 'banana', 'cherry']
    
    # Pickle the list into a file
    with open(f'{save_path}/training_features.pkl', 'wb') as f:
        pickle.dump(training_features, f)
    
    
    # TODO: add mixup
    # sig and bkg processes defined at line 1976 of AN-19-124. IDK why ggH is not included here
    sig_processes = ["vbf_powheg_dipole", "ggh_powhegPS"]
    bkg_processes = ["dy_M-100To200", "ewk_lljj_mll105_160_ptj0","ttjets_dl","ttjets_sl"]
    # sig_processes = ["ggh_powhegPS"] # testing
    # bkg_processes = ["ewk_lljj_mll105_160_ptj0"] # testing

    sig_events_dict = {}
    for process in sig_processes:
        filenames = glob.glob(f"{base_path}/{process}/*/*.parquet")
        sig_events = dak.from_parquet(filenames)
        sig_events_dict[process] = sig_events
    
    bkg_events_dict = {}
    for process in bkg_processes:
        filenames = glob.glob(f"{base_path}/{process}/*/*.parquet")
        bkg_events = dak.from_parquet(filenames)
        bkg_events_dict[process] = bkg_events

    
    
    training_features = prepare_features(sig_events, training_features) # add variation to features
    # print(f"training_features: {training_features}")
    print(f"len training_features: {len(training_features)}")
    features2load = training_features + ["event","wgt_nominal"]

    loop_dict = {
        "signal" : sig_events_dict,
        "background" : bkg_events_dict,
    }
    df_l = []
    for label, events_dict in loop_dict.items():
        print(f"{label} events dict: {events_dict}")
        for process, events in events_dict.items(): # lopp through each process's events
            df = preprocess_loop(events, features2load, region=region, category=category, label=label)
            if "dy_" in process.lower():
                df["process"] = "dy" # add in process type
            elif "ttjet" in process.lower():
                df["process"] = "top" # add in process type
            elif "ewk" in process.lower():
                df["process"] = "ewk" # add in process type
            elif "vbf" in process.lower():
                df["process"] = "vbf" # add in process type
            elif "ggh" in process.lower():
                df["process"] = "ggh" # add in process type
            # print(f"df: {df.head()}")
            print(f"df.label: {df.label}")
            print(f"df.process: {df.process}")
            df_l.append(df)

    
    # merge sig and bkg dfs
    df_total = pd.concat(df_l)
    print(df_total)
    print(f"df_total.isnull().values.any(): {df_total.isnull().values.any()}")
    # sanity check
    print(f"signal weight sum: {np.sum(df_total.wgt_nominal[df_total.label==1])}")
    print(f"bkg weight sum: {np.sum(df_total.wgt_nominal[df_total.label==0])}")

    # divide our data into 4 folds
    nfolds = 4
    for i in range(nfolds):       
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")

        train_filter = df_total.event.mod(nfolds).isin(train_folds)
        val_filter = df_total.event.mod(nfolds).isin(val_folds)
        eval_filter = df_total.event.mod(nfolds).isin(eval_folds)

        df_train = df_total[train_filter]
        df_val = df_total[val_filter]
        df_eval = df_total[eval_filter]

        
        
        # scale data, save the mean and std. This has to be done b4 mixup
        x_train = df_train[training_features].values
        print(f"x_train shape b4 mixup: {x_train.shape}")
        label_train = df_train.label.values
        wgt_train = df_train.wgt_nominal.values
        x_mean = np.average(x_train,axis=0, weights=wgt_train)
        x_std = weighted_std(x_train, wgt_train)
        print(f"x_mean: {x_mean}")
        print(f"x_std: {x_std}")
        # np.save(f"output/trained_models/{model}/scalers_{fold_idx}", [x_mean, x_std])
        
        np.save(f"{save_path}/scalers_{i}", [x_mean, x_std])


        # print(f"df_train b4 mixup: {df_train}")
        do_mixup = False
        if do_mixup:
            addToOriginalData = True
            print(f"df_train b4: {df_train.process}")
            df_mixup = copy.deepcopy(df_train)
            processes2keep = ["ggh", "vbf"]
            proc_filter = np.full(len(df_mixup), False, dtype=bool)
            for process in processes2keep:
                proc_filter = proc_filter | (df_mixup.process == process)
            df_mixup = df_mixup[proc_filter]
            print(f"df_mixup process: {df_mixup.process}")
            print(f"df_mixup label: {np.all(df_mixup.label==1)}")

            # drop process column. can't have non-numeric value for mixup, We don't need it for training anyways
            df_mixup = df_mixup.drop("process", axis=1)
            df_train = df_train.drop("process", axis=1)

            
            multiplier = 1
            

            # df_mixup = mixup(df_train, batch_size = int(len(df_train)*multiplier)) # batch size is subject to change ofc
            df_mixup = mixup(df_mixup, batch_size = int(len(df_mixup)*multiplier)) # batch size is subject to change ofc

            # print("non zero mixup labels: ",np.sum((df_mixup.label == 1) |(df_mixup.label == 0)))
            print(f"df_mixup label after mixup: {np.all(df_mixup.label==1)}")

            if addToOriginalData:
                df_train = pd.concat([df_train, df_mixup])
            else:
                df_train = df_mixup

            
            print(f"df_train after mixup: {df_train}")
            # once mixup is done, recalculate the x, label and wgt for train
            x_train = df_train[training_features].values
            label_train = df_train.label.values
            wgt_train = df_train.wgt_nominal.values # idk if this is needed
            print(f"x_train shape after mixup: {x_train.shape}")
            

        # apply scaling to data, and save the data for training
        x_train = (x_train-x_mean)/x_std

        x_val = df_val[training_features].values
        x_val = (x_val-x_mean)/x_std
        label_val = df_val.label.values
        x_eval = df_eval[training_features].values
        x_eval = (x_eval-x_mean)/x_std
        label_eval = df_eval.label.values

        # old way start --------------------------------------
        # data_dict = {
        #     "train": [x_train, label_train],
        #     "validation" : [x_val, label_val],
        #     "evaluation" : [x_eval, label_eval],
        # }
        # for mode, data in data_dict.items():
        #     np.save(f"{save_path}/data_input_{mode}_{i}", data[0])
        #     np.save(f"{save_path}/data_label_{mode}_{i}", data[1])
        # old way end --------------------------------------

        # new way start --------------------------------------
        # update the values on df and save that bc we need "process" column for analysis
        # print(f"df_train b4 scale: {df_train}")
        # print(f"df_val b4 scale: {df_val}")
        # print(f"df_eval b4 scale: {df_eval}")
        df_train[training_features] = x_train
        df_val[training_features] = x_val
        df_eval[training_features] = x_eval
        # print(f"df_train after scale: {df_train}")
        # print(f"df_val after scale: {df_val}")
        # print(f"df_eval after scale: {df_eval}")
        data_dict = {
            "train": df_train,
            "validation" : df_val,
            "evaluation" : df_eval,
        }
        for mode, data_df in data_dict.items():
            data_df.to_parquet(f"{save_path}/data_df_{mode}_{i}")
        # new way end --------------------------------------
        
    
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="vbf",
    action="store",
    help="production mode category. Options: vbf or ggh",
)
args = parser.parse_args()
    
if __name__ == "__main__":  
    from distributed import LocalCluster, Client
    cluster = LocalCluster(processes=True)
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    
    base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec22_HEMVetoOnZptOn_RerecoBtagSF_XS_Rereco/stage1_output/2018/f1_0/"
    preprocess(base_path, run_label=args.label, category=args.category)
    print("Success!")