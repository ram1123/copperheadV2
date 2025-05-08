import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob
import pandas as pd
import glob

plt.style.use(hep.style.CMS)




def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False) & ak.fill_none((events.jet1_pt_nominal > 35), value=False) 
    # vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False)
    # vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    # region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    # region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    # region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))

    # region = (events.region == "h-peak") | (events.region == "h-sidebands")
    # region = events.region == "h-sidebands"
    VBF_filter = (
        vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    trues = ak.ones_like(dimuon_mass, dtype="bool")
    falses = ak.zeros_like(dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    return events[VBF_filter]
    # return events

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    # vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False)
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False) & ak.fill_none((events.jet1_pt_nominal > 35), value=False) 
    # vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    # vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def applyttH_hadronic_cut(events):
    btag_cut = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    ttH_hadronic_filter = (
        btag_cut
    )
    return events[ttH_hadronic_filter]

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events
    
if __name__ == '__main__':
    client =  Client(n_workers=15,  threads_per_worker=2, processes=True, memory_limit='8 GiB') 
    fields_2compute = [
        "wgt_nominal",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "mu1_pt",
        "mu2_pt",
        "mu1_eta",
        "mu2_eta",
        "mu1_phi",
        "mu2_phi",
        "dimuon_pt",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_mass",
        "jet1_phi_nominal",
        "jet1_pt_nominal",
        "jet2_pt_nominal",
        "jet2_phi_nominal",
        "jet1_eta_nominal",
        "jet2_eta_nominal",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        # "region",
        "event",
    ]
    dataset_dict = {
                    "data" : ["data_*"],
                    # "DY" : ["dy_M-100To200"],
                    # "TT" : ["ttjets*"],
                    # "ST" : ["*top"],
                    # "VV" : ["ww*","wz*", "zz"],
                    # "EWK" : ["ewk_lljj_mll50_mjj120"],
                    "ggH" : ["ggh_powhegPS"],
                    "VBF" : ["vbf_powheg_dipole"],
   }
    datasets = ["data", "DY", "TT", "ST", "VV", "EWK", "ggH", "VBF"]
    years = [
        # "2018",
        # "2017",
        # "2016",
        "2016postVFP",
        # "2016preVFP",
    ]
    region = "signal" # make this a constant bc we're adding too many loops
    categories = [
        "ggh", 
        # "vbf", 
        # "nocat"
    ]


    
    # label = "V2_Jan17_JecDefault_valerieZpt"
    # label="V2_Jan27_JecDefault_TrigMatchFixed_isoMu24Or27"
    # label="V2_Jan25_JecOn_valerieZpt"
    # label ="V2_Jan27_JecDefault_TrigMatchFixed_24Or27_EcalGapEleReject"
    # label ="V2_Jan27_JecDefault_TrigMatchFixed_24Or27_EcalGapEleReject_isGlobalOrTracker"
    # label="V2_Jan27_JecDefault_TrigMatchFixed_NoHlt"
    # label="V2_Jan27_JecDefault_TrigMatchFixed_NoHlt"
    # label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
    label="rereco_yun_Dec05_btagSystFixed_JesJerUncOn"
    column_list = ["label","region", "category", "dataset", "yield"]
    yield_df = pd.DataFrame(columns=column_list)
    
    
    
    for category in categories:
        for dataset, dataset_samples in dataset_dict.items():
            total_integral = 0
            for year in years:
                # load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
                load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/"
                # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/rereco_yun_Dec05_btagSystFixed_JesJerUncOn/stage1_output/{year}/"
                filelist = []
                for dataset_sample_name in dataset_samples:
                    # filelist = glob.glob(f"{load_path}/data_*")
                    sample_filelist = glob.glob(f"{load_path}/{dataset_sample_name}")
                    filelist += sample_filelist
                print(filelist)
                
                
                for file in filelist:
                    events = dak.from_parquet(f"{file}/*.parquet")
                    # events = dak.from_parquet(f"{file}/*/*.parquet")

                    events = ak.zip({field: events[field] for field in fields_2compute}).compute()
                    events = filterRegion(events, region=region)
                    if category.lower() == "ggh":
                        events = applyGGH_cutV1(events)
                    elif category.lower() == "vbf":
                        events = applyVBF_cutV1(events)
                    elif category.lower() == "nocat":
                        pass # keep stage1 output as is
                    elif category.lower() == "tth_hadronic":
                        events = applyttH_hadronic_cut(events)
                    else:
                        print("Error: not supported category!")
                        raise ValueError

                    
                    # sample_yield = ak.num(events.dimuon_mass, axis=0)
                    sample_yield = ak.sum(events.wgt_nominal, axis=0)
                    print(f"sample_yield for {file}: {sample_yield}")
                    total_integral += sample_yield
            print(f"total integral for {region} region : {total_integral}")
            new_row = {
                    "label": [label],
                    "region": [region],
                    "category": [category],
                    "dataset": [dataset], 
                    "yield": [total_integral]
                }
            new_row = pd.DataFrame(new_row)
            yield_df = pd.concat([yield_df, new_row], ignore_index=True)

    # save the yield df at the end
    yield_df.to_csv(f"yield_df_{label}.csv")
    print("Success!")
