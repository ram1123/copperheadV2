import json
import os
import argparse

def for_all_years(value):
    out = {k: value for k in ["2016preVFP","2016postVFP", "2017", "2018"]}
    return out

def get_variations(sources):
    result = []
    for v in sources:
        result.append(v + "_up")
        result.append(v + "_down")
    return result


parameters = {}

parameters.update(
    {
        "muon_pt_cut": for_all_years(20.0),
        "muon_eta_cut": for_all_years(2.4),
        "muon_iso_cut": for_all_years(0.25),  # medium iso
        "muon_id": for_all_years("mediumId"),
        # "muon_flags": for_all_years(["isGlobal", "isTracker"]),
        "muon_flags": for_all_years([]),
        "muon_leading_pt": {"2016preVFP": 26.0,"2016postVFP": 26.0, "2017": 29.0, "2018": 26.0},
        "muon_trigmatch_iso": for_all_years(0.15),  # tight iso
        "muon_trigmatch_dr": for_all_years(0.1),
        "muon_trigmatch_id": for_all_years("tightId"),
        "electron_pt_cut": for_all_years(20.0),
        "electron_eta_cut": for_all_years(2.5),
        "electron_id": for_all_years("mvaFall17V2Iso_WP90"),
        "jet_pt_cut": for_all_years(25.0),
        "jet_eta_cut": for_all_years(4.7),
        "jet_id": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "tight", "2018": "tight"},
        # "jet_puid": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "loose", "2018": "loose"},
        "jet_puid": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "2017corrected", "2018": "loose"},
        "min_dr_mu_jet": for_all_years(0.4),
        "btag_loose_wp": {"2016preVFP": 0.2027,"2016postVFP": 0.1918 ,"2017": 0.1355, "2018": 0.1208},
        "btag_medium_wp": {"2016preVFP": 0.6001,"2016postVFP": 0.4847, "2017": 0.4506, "2018": 0.4168},
        "softjet_dr2": for_all_years(0.16),
    }
)

parameters["lumimask"] = {
    "2016preVFP": "data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2016postVFP": "data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2017": "data/lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
    "2018": "data/lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
}

parameters["hlt"] = {
    "2016preVFP": ["IsoMu24", "IsoTkMu24"],
    "2016postVFP": ["IsoMu24", "IsoTkMu24"],
    "2017": ["IsoMu27"],
    "2018": ["IsoMu24"],
}

parameters["roccor_file"] = {
    "2016preVFP": "data/roch_corr/RoccoR2016aUL.txt",
    "2016postVFP": "data/roch_corr/RoccoR2016bUL.txt",
    "2017": "data/roch_corr/RoccoR2017UL.txt",
    "2018": "data/roch_corr/RoccoR2018UL.txt",
}

parameters["nnlops_file"] = for_all_years("data/NNLOPS_reweight.root")

#parameters["btag_sf_csv"] = { #preUL
#    "2016preVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
#    "2016postVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
#    "2017": "data/btag/DeepCSV_94XSF_V5_B_F.csv",
#    "2018": "data/btag/DeepCSV_102XSF_V1.csv",
#}
parameters["btag_sf_json"] = {
    #"2016preVFP": "data/btag/DeepCSV_106XUL16preVFPSF_v1.csv",
    #"2016postVFP": "data/btag/DeepCSV_106XUL16postVFPSF_v2.csv",
    "2016preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz",
    "2016postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz",
    "2017": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz",
    "2018": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz",
}
parameters["btag_sf_csv"] = {

    "2016preVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
    "2016postVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
    "2017": "data/btag/DeepCSV_106XUL17SF.csv",
    "2018": "data/btag/DeepCSV_106XUL18SF.csv",
}

parameters["pu_file_data"] = {
    "2016preVFP": "data/pileup/puData2016_UL_withVar.root",
    "2016postVFP": "data/pileup/puData2016_UL_withVar.root",
    "2017": "data/pileup/puData2017_UL_withVar.root",
    "2018": "data/pileup/puData2018_UL_withVar.root",
}

parameters["pu_file_mc"] = {
    "2016preVFP": "data/pileup/pileup_profile_Summer16.root",
    "2016postVFP": "data/pileup/pileup_profile_Summer16.root",
    "2017": "data/pileup/mcPileup2017.root",
    "2018": "data/pileup/mcPileup2018.root",
}

parameters["muSFFileList"] = {
    "2016preVFP": [
        {
            "id": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        },
    ],
       "2016postVFP": [
        {
            "id": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        },
    ],
    "2017": [
        {
            "id": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root",
                "IsoMu27_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu27_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        }
    ],
    "2018": [
        {
            "id": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 8.95 / 59.74,
        },
        {
            "id": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 50.79 / 59.74,
        },
    ],
}

parameters["zpt_weights_file"] = for_all_years("data/zpt_weights.histo.json")
parameters["puid_sf_file"] = for_all_years("data/PUID_106XTraining_ULRun2_EffSFandUncties_v1.root")
parameters["res_calib_path"] = for_all_years("data/res_calib/")

parameters["sths_names"] = for_all_years(
    [
        "Yield",
        "PTH200",
        "Mjj60",
        "Mjj120",
        "Mjj350",
        "Mjj700",
        "Mjj1000",
        "Mjj1500",
        "PTH25",
        "JET01",
    ]
)

parameters["btag_systs"] = for_all_years(
    [
        "jes",
        "lf",
        "hfstats1",
        "hfstats2",
        "cferr1",
        "cferr2",
        "hf",
        "lfstats1",
        "lfstats2",
    ]
)

parameters.update(
    {
        "event_flags": for_all_years(
            [
                "BadPFMuonFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "globalSuperTightHalo2016Filter",
                "goodVertices",
                "BadChargedCandidateFilter",
            ]
        ),
        "do_l1prefiring_wgts": {"2016preVFP": True,"2016postVFP": True, "2017": True, "2018": False},
    }
)

parameters["n_pdf_variations"] = {"2016preVFP": 100, "2016postVFP": 100, "2017": 33, "2018": 33}

parameters["dnn_max"] = {"2016preVFP": 1.75, "2016postVFP": 1.75, "2017": 2.0, "2018": 2.35}


cross_sections = {
    "test": 6200.0,
    "dy_M-50": 6195.8, #UL, fromXSDB (averaged)
    "dy_M-50_nocut": 6450.0,
    "dy_M-100To200": 254.2,  #UL, fromXSDB
    "dy_0j": 4620.52,
    "dy_1j": 922.5,  #UL, fromXSDB
    "dy_2j": 293.6,  #UL, fromXSDB
    "dy_m105_160_mg": 47.17,
    "dy_m105_160_vbf_mg": {"2016": 1.77, "2017": 2.04, "2018": 2.03},
    "dy_m105_160_amc": 47.17,
    "dy_m105_160_vbf_amc": {"2016": 1.77, "2017": 2.04, "2018": 2.03},
    "ewk_lljj_mll105_160": {"2016": 0.0508896, "2017": 0.0508896, "2018": 0.0508896},
    "ewk_lljj_mll105_160_py": {"2016": 0.0508896, "2017": 0.0508896, "2018": 0.0508896},
    "ewk_lljj_mll105_160_ptj0": {"2016": 0.07486, "2017": 0.0789, "2018": 0.0789},
    "ewk_lljj_mll50_mjj120":1.719,
    "ewk_lljj_mll105_160_py_dipole": {"2016": 0.07486, "2017": 0.0789, "2018": 0.0789},
    "ewk_m50": 3.998,
    "st_top": 136.02,
    "st_t_top": 3.40,
    "st_t_antitop": 80.95,
    "st_tw_top": 32.51, #UL, fromXSDB
    "st_tw_antitop": 32.45, #UL, fromXSDB
    "ttjets_dl": 86.65,
    "ttjets_sl": 358.57,
    "ww_2l2nu": 11.09, #UL, fromXSDB
    "wz_3lnu": 5.22, #UL, fromXSDB (averaged)
    "wz_2l2q": 6.45, #UL, fromXSDB (averaged)
    "wz_1l1nu2q": 9.12, #UL, fromXSDB
    "zz": 12.17, #UL, fromXSDB
    "ttw": 0.2001,
    "ttz": 0.2529,
    "www": 0.2086,
    "wwz": 0.1651,
    "wzz": 0.05565,
    "zzz": 0.01398,
    "ggh_powheg": 0.01057,
    "ggh_powhegPS": 0.01057,
    "ggh_amcPS": 0.01057,
    "ggh_amcPS_TuneCP5down": 0.01057,
    "ggh_amcPS_TuneCP5up": 0.01057,
    "ggh_amc": 0.01057,
    "ggh_localTest": 0.01057,
    "vbf": 0.0008210722,
    "vbf_sync": 0.0008210722,
    "vbf_powheg": 0.0008210722,
    "vbf_powheg_herwig": 0.0008210722,
    "vbf_powheg_dipole": 0.0008210722,
    "vbf_powhegPS": 0.0008210722,
    "vbf_amc_herwig": 0.0008210722,
    "vbf_amcPS_TuneCP5down": 0.0008210722,
    "vbf_amcPS_TuneCP5up": 0.0008210722,
    "vbf_amcPS": 0.0008210722,
    "vbf_amc": 0.0008210722,
    "ggh_powhegPS_m120": 0.012652906,
    "ggh_amcPS_m120": 0.012652906,
    "vbf_powhegPS_m120": 0.0009534505,
    "vbf_amcPS_m120": 0.0009534505,
    "ggh_powhegPS_m130": 0.008504687,
    "ggh_amcPS_m130": 0.008504687,
    "vbf_powhegPS_m130": 0.0006826649,
    "vbf_amcPS_m130": 0.0006826649,
    "wmh": 0.000116,
    "wph": 0.000183,
    "zh": 0.000192,
    "tth": 0.000110,
}
parameters["cross_sections"] = cross_sections

integrated_lumis = {
    "2016preVFP" : 19500.0,
    "2016postVFP" : 16800.0,
    "2017" : 41530.0,
    "2018" : 59970.0,
}
parameters["integrated_lumis"] = integrated_lumis

jec_parameters = {}

jec_unc_to_consider = {
    "2016preVFP": [
        "Absolute",
        "Absolute2016",
        "BBEC1",
        "BBEC12016",
        "EC2",
        "EC22016",
        "HF",
        "HF2016",
        "RelativeBal",
        "RelativeSample2016",
        "FlavorQCD",
    ],
    "2016postVFP": [
        "Absolute",
        "Absolute2016",
        "BBEC1",
        "BBEC12016",
        "EC2",
        "EC22016",
        "HF",
        "HF2016",
        "RelativeBal",
        "RelativeSample2016",
        "FlavorQCD",
    ],
    "2017": [
        "Absolute",
        "Absolute2017",
        "BBEC1",
        "BBEC12017",
        "EC2",
        "EC22017",
        "HF",
        "HF2017",
        "RelativeBal",
        "RelativeSample2017",
        "FlavorQCD",
    ],
    "2018": [
        "Absolute",
        "Absolute2018",
        "BBEC1",
        "BBEC12018",
        "EC2",
        "EC22018",
        "HF",
        "HF2018",
        "RelativeBal",
        "RelativeSample2018",
        "FlavorQCD",
    ],
}

jec_parameters["jec_variations"] = {
    year: get_variations(jec_unc_to_consider[year]) for year in ["2016preVFP","2016postVFP", "2017", "2018"]
}

jec_parameters["runs"] = {
    "2016preVFP": ["B", "C", "D", "E", "F"],
    "2016postVFP": ["F", "G", "H"],
    "2017": ["B", "C", "D", "E", "F"],
    "2018": ["A", "B", "C", "D"],
}

jec_parameters["jec_levels_mc"] = for_all_years(
    ["L1FastJet", "L2Relative", "L3Absolute"]
)
jec_parameters["jec_levels_data"] = for_all_years(
    ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]
)

jec_parameters["jec_tags"] = {
    "2016preVFP": "Summer19UL16APV_V7_MC",
    "2016postVFP": "Summer19UL16_V7_MC",
    "2017": "Summer19UL17_V5_MC",
    "2018": "Summer19UL18_V5_MC",
}

jec_parameters["jer_tags"] = {
    "2016preVFP": "Summer20UL16APV_JRV3_MC",
    "2016postVFP": "Summer20UL16_JRV3_MC",
    "2017": "Summer19UL17_JRV2_MC",
    "2018": "Summer19UL18_JRV2_MC",
}

jec_parameters["jec_data_tags"] = {
    "2016preVFP": {
        "Summer19UL16APV_RunBCD_V7_DATA": ["B", "C", "D"],
        "Summer19UL16APV_RunEF_V7_DATA": ["E", "F"],
    },
    "2016postVFP": {
        "Summer19UL16_RunFGH_V7_DATA": ["F","G","H"],
    },
    "2017": {
        "Summer19UL17_RunB_V5_DATA": ["B"],
        "Summer19UL17_RunC_V5_DATA": ["C"],
        "Summer19UL17_RunD_V5_DATA": ["D"],
        "Summer19UL17_RunE_V5_DATA": ["E"],
        "Summer19UL17_RunF_V5_DATA": ["F"],
    },
    "2018": {
        "Summer19UL18_RunA_V5_DATA": ["A"],
        "Summer19UL18_RunB_V5_DATA": ["B"],
        "Summer19UL18_RunC_V5_DATA": ["C"],
        "Summer19UL18_RunD_V5_DATA": ["D"],
    },
}

jer_variations = ["jer1", "jer2", "jer3", "jer4", "jer5", "jer6"]
jec_parameters["jer_variations"] = {
    year: get_variations(jer_variations) for year in ["2016preVFP","2016postVFP", "2017", "2018"]
}

parameters["jec_parameters"] = jec_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="year",
    )
    args = parser.parse_args()
    config_to_save = {}
    for key, val in parameters.items():
        print(f"make parameters key: {key}")
        if "cross_sections" in key:
            config_to_save[key] = val
        elif "jec" in key: # if jec, then do it separately
            sub_jec_pars = {}
            for sub_key, sub_val in val.items():
                sub_jec_pars[sub_key] = sub_val[args.year]
            print(f"make parameters sub_jec_pars: {sub_jec_pars}")
            config_to_save[key] = sub_jec_pars
            # config_to_save[key] = val
        else:
            config_to_save[key] = val[args.year]
    config_to_save["do_roccor"] = True
    config_to_save["do_fsr"] = True
    config_to_save["do_geofit"] = True
    config_to_save["year"] = args.year
    config_to_save["do_jecunc"] = False
    config_to_save["do_jerunc"] = False
    print(f"make_parameters config_to_save: \n {config_to_save}")

   
    
    #save config as json
    directory = "./config"
    filename = directory+"/parameters.json"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w") as file:
        json.dump(config_to_save, file)

