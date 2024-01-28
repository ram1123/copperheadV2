import json
import os
import argparse

def for_all_years(value):
    out = {k: value for k in ["2016preVFP","2016postVFP", "2017", "2018"]}
    return out


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
        "jet_puid": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "loose", "2018": "loose"},
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
        config_to_save[key] = val[args.year]
    config_to_save["do_roccor"] = True
    config_to_save["do_fsr"] = True
    config_to_save["do_geofit"] = True
    print(f"make_parameters config_to_save: \n {config_to_save}")

    #save config as json
    directory = "./config"
    filename = directory+"/parameters.json"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w") as file:
        json.dump(config_to_save, file)

