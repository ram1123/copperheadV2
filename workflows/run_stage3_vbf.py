import argparse
import dask
from dask.distributed import Client

# from config.variables import variables_lookup
# from stage3.plotter import plotter
from stage3.make_templates import to_templates
from stage3.make_datacards import build_datacards
import time
__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
args = parser.parse_args()



year = args.years[0]
if "2016" in year:
    year = "2016"

# global parameters
parameters = {
    # < general settings >
    "years": args.years,
    "global_path": "/depot/cms/users/yun79/hmm/copperheadV1clean/",
    # "global_path": "/work/users/yun79/copperhead_outputs/copperheadV1clean",
    # "label": "DmitryMaster_JECoff_GeofitFixed_Oct29",
    # "label": "DmitryMaster_JECoff_GeofitFixed_Nov01",
    # "label": "rereco_yun_Nov04",
    "label": args.label,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "syst_variations": ["nominal"],
    # "syst_variations": ['nominal', 'Absolute', 'Absolute2018', 'BBEC1', 'BBEC12018', 'EC2', 'EC22018', 'HF', 'HF2018', 'RelativeBal', 'RelativeSample2018', 'FlavorQCD', 'jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6', ],
    # "syst_variations": ['nominal', 'Absolute', f'Absolute_{year}', 'BBEC1', f'BBEC1_{year}', 'EC2', f'EC2_{year}', 'HF', f'HF_{year}', 'RelativeBal', f'RelativeSample_{year}', 'FlavorQCD', 'jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6', ],
    # < plotting settings >
    "plot_vars": [],  # "dimuon_mass"],
    # "variables_lookup": variables_lookup,
    "dnn_models": {   
         "vbf": ["Dec31_stage2_test"],   
    },
    "bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": True,
    "templates_vars": [],  # "dimuon_mass"],
}


parameters["grouping"] = {
    # "data_A": "Data",
    # "data_B": "Data",
    # "data_C": "Data",
    # "data_D": "Data",
    # "data_E": "Data",
    # "data_F": "Data",
    # "data_G": "Data",
    # "data_H": "Data",
    "data": "Data",
    # # # # # "dy_m105_160_amc": "DY",
    # # # # # "dy_m105_160_vbf_amc": "DY",
    # "dy_m105_160_amc_01j": "DYJ01",
    # "dy_m105_160_vbf_amc_01j": "DYJ01", 
    # "dy_M-100To200_01j": "DYJ01",
    # "dy_m105_160_amc_2j": "DYJ2",
    # "dy_m105_160_vbf_amc_2j": "DYJ2",
    # "dy_M-100To200_2j": "DYJ2",
    # # # # # "ewk_lljj_mll105_160_py_dipole": "EWK",
    # "ewk_lljj_mll105_160_ptj0": "EWK",
    # "ttjets_dl": "TT+ST",
    # "ttjets_sl": "TT+ST",
    # "ttw": "TT+ST",
    # "ttz": "TT+ST",
    # "st_tw_top": "TT+ST",
    # "st_tw_antitop": "TT+ST",
    # "ww_2l2nu": "VV",
    # "wz_2l2q": "VV",
    # "wz_1l1nu2q": "VV", # bad for 2016
    # "wz_3lnu": "VV",
    # "zz": "VV",
    # # # # # # "www": "VVV",
    # # # # # # "wwz": "VVV",
    # # # # # # "wzz": "VVV",
    # # # # # # "zzz": "VVV",
    # # # # # #"ggh_amcPS": "ggH_hmm",
    "ggh_powhegPS": "ggH_hmm",
    "vbf_powheg_dipole": "qqH_hmm",
}




parameters["plot_groups"] = {
    # "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "stack": ["DY", "EWK", "TT+ST", "VV"],
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}


if __name__ == "__main__":
    start_time = time.time()
    from distributed import LocalCluster, Client
    cluster = LocalCluster(processes=True)
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    print("Local scale Client created")

    # add MVA scores to the list of variables to plot
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # skip plots for now
    # # make plots
    # yields = plotter(client, parameters)
    # print(yields)

    # save templates to ROOT files
    yield_df = to_templates(client, parameters)
    print(f'run stage3 yield_df: {yield_df}')
    # yield_df.to_csv("test.csv")
    # groups = [g for g in yield_df.group.unique() if g != "Data"]
    # print(f'parameters["templates_vars"]: {parameters["templates_vars"]}')
    
    # print(f"yield groups: {groups}")

    datacard_str = parameters["dnn_models"]["vbf"][0]
    print(f"datacard_str: {datacard_str}")
    # make datacards
    build_datacards(f"score_{datacard_str}", yield_df, parameters)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the elapsed time
    print(f"Execution time: {execution_time:.4f} seconds")