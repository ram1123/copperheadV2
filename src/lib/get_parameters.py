import glob
from omegaconf import OmegaConf
from rich import print

def getParametersForYr(parameter_path: str, year: str, analysis: str = "HMuMu") -> dict:
    """
    Load and merge YAML parameter files for the given year and analysis.

    Files are loaded from two subdirectories:
      - {parameter_path}/common/   — shared across all analyses
      - {parameter_path}/{analysis}/ — analysis-specific overrides

    Params:
    parameter_path -> root path, typically configs/parameters/
    year -> Run era year, e.g. "2017", "2022preEE"
    analysis -> analysis type: "HMuMu" (default) or "XZZ2l2nu"
    """
    common_files = sorted(glob.glob(f"{parameter_path}/common/*.yaml"))
    analysis_files = sorted(glob.glob(f"{parameter_path}/{analysis}/*.yaml"))
    filelist = common_files + analysis_files
    # print(f"getParametersForYr filelist: {filelist}")
    params = [OmegaConf.load(f) for f in filelist]
    merged_param = OmegaConf.merge(*params)
    yr_specific_params = {}
    for key, val in merged_param.items():
        # print(f"key: {key}, val: {val}")
        if "cross_sections" in key:
            yr_specific_params[key] = val
        elif "jec" in key: # if jec, then do it separately
            sub_jec_pars = {}
            for sub_key, sub_val in val.items():
                # print(f"sub_key: {sub_key}, sub_val: {sub_val}")
                sub_jec_pars[sub_key] = sub_val[year]
            yr_specific_params[key] = sub_jec_pars
        elif key == "switches":
            sub_switches = {}
            for sub_key, sub_val in val.items():
                # print(f"sub_key: {sub_key}, sub_val: {sub_val}")
                sub_switches[sub_key] = sub_val[year]
            yr_specific_params[key] = sub_switches
        else:
            # print(f"key: {key}, val: {val}")
            yr_specific_params[key] = val[year]
    yr_specific_params["do_roccor"] = True
    yr_specific_params["do_fsr"] = True
    yr_specific_params["do_geofit"] = True
    yr_specific_params["year"] = year
    yr_specific_params["do_jecunc"] = False
    yr_specific_params["do_jerunc"] = False

    # save year specific yaml for testing vs json version
    # directory = "./config"
    # filename = directory+"/parameters.yaml"
    # with open(filename, "w") as file:
    #     OmegaConf.save(config=yr_specific_params, f=file.name)
    return yr_specific_params
