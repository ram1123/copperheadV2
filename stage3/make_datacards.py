import pandas as pd
# from python.io import mkdir
import os
import logging
from modules.utils import logger

rename_regions = {
    "h-peak": "SR",
    "h-sidebands": "SB",
    "z-peak": "Z",
}
# signal_groups = ["VBF", "ggH"]
signal_groups = ["qqH_hmm", "ggH_hmm"]
rate_syst_lookup = {
    "2016": {
        # "XsecAndNorm2016DYJ2": {"DY_2J": 1.1291},
        "XsecAndNorm2016DYJ2": {"DYJ2": 1.1291},
        "XsecAndNorm2016EWK": {"EWK": 1.06131},
        "XsecAndNormTT+ST": {"TT+ST": 1.182},
        "XsecAndNormVV": {"VV": 1.13203},
        "XsecAndNormggH": {"ggH": 1.38206},
    },
    "2017": {
        # "XsecAndNorm2017DYJ2": {"DY_2J": 1.13020},
        "XsecAndNorm2017DYJ2": {"DYJ2": 1.13020},
        "XsecAndNorm2017EWK": {"EWK": 1.05415},
        "XsecAndNormTT+ST": {"TT+ST": 1.18406},
        "XsecAndNormVV": {"VV": 1.05653},
        "XsecAndNormggH": {"ggH": 1.37126},
    },
    "2018": {
        # "XsecAndNorm2018DYJ2": {"DY_2J": 1.12320},
        "XsecAndNorm2018DYJ2": {"DYJ2": 1.12320},
        "XsecAndNorm2018EWK": {"EWK": 1.05779},
        "XsecAndNormTT+ST": {"TT+ST": 1.18582},
        "XsecAndNormVV": {"VV": 1.05615},
        "XsecAndNormggH": {"ggH": 1.38313},
    },
}
lumi_syst = {
    "2016": {
        # "uncor2016": 2.2,
        # "xyfac": 0.9,
        # "len": 0.0,
        # "bb": 0.4,
        # "beta": 0.5,
        # "calib": 0.0,
        # "ghost": 0.4,
        "lumi2016": 1.2,
    },
    "2017": {
        # "uncor2017": 2.0,
        # "xyfac": 0.8,
        # "len": 0.3,
        # "bb": 0.4,
        # "beta": 0.5,
        # "calib": 0.3,
        # "ghost": 0.1,
        "lumi2017": 2.3,
    },
    "2018": {
        # "uncor2018": 1.5,
        # "xyfac": 2.0,
        # "len": 0.2,
        # "bb": 0.0,
        # "beta": 0.0,
        # "calib": 0.2,
        # "ghost": 0.0,
        "lumi2018": 2.5,
    },
}

nuisance_titles = {
    "muID" : "muIDYEAR",
    "muIso" : "muIsoYEAR",
    "muTrig" : "muTrigYEAR",
    "pu" : "pu_wgtYEAR",
    "qgl" : "qgl_wgt",
}

def editNuisance_names(nuisance, nuisance_titles, year):
    nuisance_name = nuisance_titles[nuisance].replace("YEAR", str(year))
    return nuisance_name

def build_datacards(var_name, yield_df, parameters):
    if yield_df is None or yield_df.empty:
        logger.error("Yield DataFrame is empty. Cannot build datacards.")
        return

    channels = parameters["channels"]
    regions = parameters["regions"]
    years = parameters["years"]
    global_path = parameters["global_path"]
    # label = parameters["label"]

    outpath_postfix = "_" + parameters["outpath_postfix"] if parameters["outpath_postfix"] else ""

    datacard_path = f"{global_path}/stage3_datacards{outpath_postfix}/"
    templates_path = f"{global_path}/stage3_datacards{outpath_postfix}/stage3_templates{outpath_postfix}/{var_name}"
    templates_path_to_remove_from_datacard = f"{global_path}/stage3_datacards{outpath_postfix}/"
    # mkdir(datacard_path)
    datacard_path += "/" + var_name

    # mkdir(datacard_path)
    if not os.path.exists(datacard_path):
        os.makedirs(datacard_path)

    for year in years:
        if "2016" in year:
            year_savepath = year
            year = "2016"
        else:
            year_savepath = year
        for channel in channels:
            for region in regions:
                region_new = rename_regions[region]
                # bin_name = f"{channel}_{region_new}_{year}"
                bin_name = f"{channel}_{region_new}_{year_savepath}"
                datacard_name = (
                    # f"{datacard_path}/datacard_{channel}_{region_new}_{year}.txt"
                    f"{datacard_path}/datacard_{channel}_{region_new}_{year_savepath}.txt"
                )
                # templates_file = f"{templates_path}/{channel}_{region}_{year}.root"
                templates_file = f"{templates_path}/{channel}_{region}_{year_savepath}.root"
                datacard = open(datacard_name, "w")
                datacard.write("imax 1\n")
                datacard.write("jmax *\n")
                datacard.write("kmax *\n")
                datacard.write("---------------\n")
                datacard.write(
                    f"shapes * {bin_name} {templates_file.replace(templates_path_to_remove_from_datacard, '../')} $PROCESS $PROCESS_$SYSTEMATIC\n"
                )
                datacard.write("---------------\n")
                data_str = print_data(
                    yield_df, var_name, region, channel, year, bin_name
                )
                datacard.write(data_str)
                datacard.write("---------------\n")
                mc_str = print_mc(yield_df, var_name, region, channel, year, bin_name)
                datacard.write(mc_str)
                datacard.write("---------------\n")
                # shape_syst = print_shape_syst(yield_df, mc_df)
                # datacard.write("---------------\n")
                # datacard.write(systematics)
                # datacard.write(
                #     f"XSecAndNorm{year}DYJ01  rateParam {bin_name} DY_01J 1 [0.2,5]\n"
                # )
                datacard.write(
                    f"XSecAndNorm{year}DYJ01  rateParam {bin_name} DYJ01 1 [0.2,5]\n"
                )
                datacard.write(f"{bin_name} autoMCStats 0 1 1\n")
                datacard.write("---------------\n")
                # nuisnace edit start ----------------------------
                datacard.write(
                "nuisance edit rename"
                " (DYJ2|DYJ01|ggH_hmm|TT+ST|VV) * "
                "qgl_wgt  QGLweightPY \n"
                )
                datacard.write("nuisance edit rename EWK * qgl_wgt" " QGLweightHER \n")
                datacard.write(
                "nuisance edit rename qqH_hmm * qgl_wgt" " QGLweightPYDIPOLE \n"
                )
                datacard.write("---------------\n")
                # nuisnace edit end ----------------------------
                datacard.close()
                logger.info(f"Saved datacard to {datacard_name}")

                # debugging
                with open(datacard_name, 'r') as file:
                    logger.info("printing datacard")
                    content = file.read()  # Read the entire file content
                    logger.info(content)
    return


def print_data(yield_df, var_name, region, channel, year, bin_name):
    if 'group' not in yield_df.columns:
        logger.error("Yield DataFrame does not contain 'group' column.")
        return ""
    if "Data" not in yield_df.group.unique():
        logger.warning("No 'Data' group found in yield DataFrame.")
        return ""
    if "Data" in yield_df.group.unique():
        try:
            data_yield = yield_df.loc[
                (yield_df.var_name == var_name)
                & (yield_df.region == region)
                & (yield_df.channel == channel)
                & (yield_df.year == year)
                & (yield_df.variation == "nominal")
                & (yield_df.group == "Data"),
                "yield",
            ].values[0]
        except Exception:
            data_yield = 0
        data_str = "{:<20} {:>20}\n".format("bin", bin_name) + "{:<20} {:>20}\n".format(
            "observation", int(data_yield)
        )
        return data_str

    else:
        return ""


def print_mc(yield_df, var_name, region, channel, year, bin_name):
    # get yields for each process
    sig_counter = 0
    bkg_counter = 0
    mc_rows = []
    nuisances = {}
    all_nuisances = []
    nuisance_lines = {}

    groups = [g for g in yield_df.group.unique() if g != "Data"]
    for group in groups:
        if group in signal_groups:
            sig_counter -= 1
            igroup = sig_counter
        else:
            bkg_counter += 1
            igroup = bkg_counter

        try:
            mc_yield = yield_df.loc[
                (yield_df.var_name == var_name)
                & (yield_df.region == region)
                & (yield_df.channel == channel)
                & (yield_df.year == year)
                & (yield_df.variation == "nominal")
                & (yield_df.group == group),
                "yield",
            ].values[0]
        except Exception:
            mc_yield = 0

        mc_yield = round(mc_yield, 6)
        mc_row = {"group": group, "igroup": igroup, "yield": mc_yield}

        nuisances[group] = []
        variations = yield_df.loc[
            ((yield_df.group == group) & (yield_df.year == year)), "variation"
        ].unique()
        for v in variations:
            if v == "nominal":
                continue
            v_name = v.replace("Up", "").replace("Down", "")
            if v_name not in all_nuisances:
                all_nuisances.append(v_name)

                # adding nuisance name to match the official workspace -------
                # if v_name in nuisance_titles.keys():
                #     nuisance_name= editNuisance_names(v_name, nuisance_titles, year)
                # else:
                #     nuisance_name = v_name
                # adding nuisance name to match the official workspace -------
                nuisance_name = v_name
                # logger.info(f"nuisance_name: {nuisance_name}")
                nuisance_lines[v_name] = "{:<20} {:<9}".format(nuisance_name, "shape")
            if v_name not in nuisances[group]:
                nuisances[group].append(v_name)

        mc_rows.append(mc_row)

    mc_df = pd.DataFrame(mc_rows).sort_values(by="igroup")
    for group, gr_nuis in nuisances.items():
        for nuisance in gr_nuis:
            mc_df.loc[mc_df.group == group, nuisance] = "1.0"

    for rate_unc, apply_to in rate_syst_lookup[year].items():
        if rate_unc not in all_nuisances:
            all_nuisances.append(rate_unc)
            nuisance_lines[rate_unc] = "{:<20} {:<9}".format(rate_unc, "lnN")
        for group, value in apply_to.items():
            mc_df.loc[mc_df.group == group, rate_unc] = str(value)

    for lumi_unc, value in lumi_syst[year].items():
        if lumi_unc not in all_nuisances:
            all_nuisances.append(lumi_unc)
            nuisance_lines[lumi_unc] = "{:<20} {:<9}".format(lumi_unc, "lnN")
            mc_df.loc[:, lumi_unc] = str(1 + value / 100)

    mc_df = mc_df.fillna("-")

    # prepare datacard lines
    mc_str_1 = "{:<30}".format("bin")
    mc_str_2 = "{:<30}".format("process")
    mc_str_3 = "{:<30}".format("process")
    mc_str_4 = "{:<30}".format("rate")

    for group in groups:
        group_df = mc_df[mc_df.group == group]
        mc_str_1 += "{:<20}".format(bin_name)
        mc_str_2 += "{:<20}".format(group)
        mc_str_3 += "{:<20}".format(group_df["igroup"].values[0])
        mc_str_4 += "{:<20}".format(group_df["yield"].values[0])
        for nuisance in all_nuisances:
            nuisance_lines[nuisance] += "{:<20}".format(group_df[nuisance].values[0])

    process_lines = f"{mc_str_1}\n{mc_str_2}\n{mc_str_3}\n{mc_str_4}\n"

    mc_str = process_lines + "---------------\n"
    for nuisance in all_nuisances:
        mc_str += nuisance_lines[nuisance] + "\n"

    return mc_str
