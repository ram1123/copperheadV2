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
    # vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False) & ak.fill_none((events.jet1_pt_nominal > 35), value=False) 
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False)
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
    vbf_cut = ak.fill_none((events.jj_mass_nominal > 400), value=False) & ak.fill_none((events.jj_dEta_nominal > 2.5), value=False)
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
     
    # year = "2018"
    # year = "2016postVFP"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/rereco_yun_Dec05_btagSystFixed_JesJerUncOn/stage1_output/{year}/"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/ul_yun_Dec10/stage1_output/{year}/"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/ul_yun_Dec12_L1JecOff/stage1_output/{year}/"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/ul_yun_Dec12_JecOff/stage1_output/{year}/"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/ul_yun_Dec12_JecOff_JesJerUncOn/stage1_output/{year}/"
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/ul_yun_Dec15_JecOff_JesJerUncOn_2016LumiFix/stage1_output/{year}/"
    
    # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/V2_Dec20_RERECO_MuIdMuIsoRoccor/stage1_output/{year}/f1_0"
    
    # label = "V2_Jan17_JecDefault_valerieZpt"
    label = "test_test"
    total_integral = 0
    for year in ["2018", "2017", "2016postVFP", "2016preVFP"]:
    # for year in ["2018", "2017", "2016"]:
        load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0"
        # load_path =f"//depot/cms/users/yun79/hmm/copperheadV1clean/rereco_yun_Dec05_btagSystFixed_JesJerUncOn/stage1_output/{year}/"
        filelist = glob.glob(f"{load_path}/data_*")
        print(filelist)
        
        for region in ["signal"]:
            for file in filelist:
                # events_data = dak.from_parquet(f"{file}/*.parquet")
                events_data = dak.from_parquet(f"{file}/*/*.parquet")
                # print(events_data.fields)
                # events_data.fields
                events_data = ak.zip({field: events_data[field] for field in fields_2compute}).compute()
                
                # print(region)
                # raise ValueError
                events_data = filterRegion(events_data, region=region)
                events_data = applyGGH_cutV1(events_data)
                # events_data = applyVBF_cutV1(events_data)
                # events_data = applyttH_hadronic_cut(events_data)
                
                data_yield = ak.num(events_data.dimuon_mass, axis=0)
                # data_yield = ak.num(events_data.dimuon_mass, axis=0).compute()
                # ak.to_dataframe(events_data).to_csv("event_dataC_V1.csv")
                # df = pd.DataFrame({field: ak.fill_none(events_data[field], value=-999.9) for field in events_data.fields})
                # df.to_csv("event_dataC_V1.csv")
                print(f"data_yield for {file}: {data_yield}")
                total_integral += data_yield
    print(f"total integral for {region} region : {total_integral}")
