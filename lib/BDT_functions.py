from typing import Tuple, List, Dict
import ROOT as rt
import numpy as np
import pickle
import awkward as ak

# functions for MVA related stuff start --------------------------------------------
def prepare_features(df, training_features, variation="nominal", add_year=False):
    #global training_features
    if add_year:
        features = training_features + ["year"]
    else:
        features = training_features
    features_var = []
    #print(features)
    for trf in features:
        if f"{trf}_{variation}" in df.fields:
            features_var.append(f"{trf}_{variation}")
        elif trf in df.fields:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var

    

def evaluate_bdt(df: ak.Record, variation, model, training_features: List[str], parameters) -> ak.Record :
    """
    This also filters in only h_peak and h_sidebands regpion
    """
    # filter out events neither h_peak nor h_sidebands
    row_filter = (df.h_peak != 0) | (df.h_sidebands != 0)
    df = df[row_filter]
    
    # training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet1_qgl_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal', 'jet2_qgl_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']
    # training_features = [
    #     'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
    #     'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
    #     'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
    #     'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 
    #     'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
    #     'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
    # ]

    
    # df['mu1_pt_over_mass'] = df['mu1_pt']/df['dimuon_mass']
    # df['mu2_pt_over_mass'] = df['mu2_pt']/df['dimuon_mass']
    # df['njets'] = ak.fill_none(df['njets'], value=0)

    #df[df['njets_nominal']<2]['jj_dPhi_nominal'] = -1
    none_val = -99.0
    for field in df.fields:
        df[field] = ak.fill_none(df[field], value= none_val)
        inf_cond = (np.inf == df[field]) | (-np.inf == df[field]) 
        df[field] = ak.where(inf_cond, none_val, df[field])
        
    # print(f"df.h_peak: {df.h_peak}")
    print(f"sum df.h_peak: {ak.sum(df.h_peak)}")
    # overwrite dimuon mass for regions not in h_peak
    not_h_peak = (df.h_peak ==0)
    # df["dimuon_mass"] = ak.where(not_h_peak, 125.0,  df["dimuon_mass"])
    


    # idk why mmj variables are overwritten something to double chekc later
    df['mmj_min_dEta'] = df["mmj2_dEta"]
    df['mmj_min_dPhi'] = df["mmj2_dPhi"]

    # temporary definition of even bc I don't have it
    if "event" not in df.fields:
        df["event"] = np.arange(len(df.dimuon_pt))
    
    features = prepare_features(df,training_features, variation=variation, add_year=False)
    # features = training_features
    #model = f"{model}_{parameters['years'][0]}"
    # score_name = f"score_{model}_{variation}"
    score_name = "BDT_score"

    # df.loc[:, score_name] = 0
    score_total = np.zeros(len(df['dimuon_pt']))
    
    nfolds = 4
    
    for i in range(nfolds):
        # eval_folds are the list of test dataset chunks that each bdt is trained to evaluate
        eval_folds = [(i + f) % nfolds for f in [3]]
        # eval_filter = df.event.mod(nfolds).isin(eval_folds)
        eval_filter = (df.event % nfolds ) == (np.array(eval_folds) * ak.ones_like(df.event))
        scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
        scalers = np.load(scalers_path, allow_pickle=True)
        model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pkl"

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        # print(f"df_i: {len(df_i)}")
        # print(len
        if len(df_i) == 0:
            continue
        # df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
        print(f"scalers: {scalers.shape}")
        print(f"df_i: {df_i}")
        df_i_feat = df_i[features]
        # df_i_feat = np.transpose(np.array(ak.unzip(df_i_feat)))
        df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
        print(f"df_i_feat[:,0]: {df_i_feat[:,0]}")
        print(f'df_i.dimuon_cos_theta_cs: {df_i.dimuon_cos_theta_cs}')
        # print(f"type df_i_feat: {type(df_i_feat)}")
        # print(f"df_i_feat: {df_i_feat.shape}")
        df_i_feat = ak.Array(df_i_feat)
        df_i = (df_i_feat - scalers[0]) / scalers[1]
        if len(df_i) > 0:
            print(f"model: {model}")
            prediction = np.array(
                # bdt_model.predict_proba(df_i.values)[:, 1]
                bdt_model.predict_proba(df_i_feat)[:, 1]
            ).ravel()
            print(f"prediction: {prediction}")
            # df.loc[eval_filter, score_name] = prediction  # np.arctanh((prediction))
            # score_total = ak.where(eval_filter, prediction, score_total)
            score_total[eval_filter] = prediction

    df[score_name] = score_total
    return df