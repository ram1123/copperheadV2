import glob
import os
from omegaconf import OmegaConf
from rich import print


def getParametersForYr(parameter_path: str, year: str, switches_path: str) -> dict:
    """
    This is a simple python function that takes in all the parameters defined by the local yaml files, merges them and returns a dictionary of omegaconf variables (which are basically dictionaries) for a given year
    If you would like to only accept certain yaml files, feel free to hard code the
    filelist varaibles to contain the yaml files you want

    Params:
    parameter_path -> path where parameter yaml files are saved in
        typically, the value is configs/parameters/
    year -> Run era year in question
    switches_path -> path to the switches yaml (required). Switches live in their own
        directory, configs/switches/, and are never merged from parameter_path: exactly
        ONE switches file is loaded per run. The default file is defined once, as the
        --switches-yaml default in cli/common_argparser.py.
    """
    filelist = glob.glob(parameter_path + "*.yaml")
    stray = sorted(f for f in filelist if os.path.basename(f).startswith("switches"))
    if stray:
        raise ValueError(
            f"switches files must live in configs/switches/, not in {parameter_path}: {stray} "
            "(everything in the parameters directory is merged, which would silently mix switch values)"
        )
    if not switches_path or not os.path.isfile(switches_path):
        raise FileNotFoundError(f"switches yaml not found: {switches_path!r}")
    filelist.append(str(switches_path))
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
