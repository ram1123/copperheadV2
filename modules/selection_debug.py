import numpy as np
import awkward as ak

binning_based_on_significanceScan = np.array([
  0.000000,
  0.349433,
  0.662083,
  0.882777,
  1.066689,
  1.250601,
  1.388535,
  1.590838,
  1.793141,
  1.958661,
  2.069008,
  2.262116,
  2.482810,
  3.678237,
])

binning_based_on_significanceScanV2 = np.array(  # 17 bins /depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/best_binning_25bins_0p01.txt
    [  # one used for September 25, 2025 HiggsMuMu working group meeting.
        0.000000,
        0.179242,
        0.358485,
        0.537727,
        0.716970,
        0.896212,
        1.075455,
        1.254697,
        1.433940,
        1.613182,
        1.792425,
        1.971667,
        2.150910,
        2.330152,
        2.509395,
        2.688637,
        3.047122,
        4.301819,
    ]
)


# Binning for DNN scores
binning_HPScan_21bins = np.array([  #Latest training; 03 Sep 2025 (21 bins)
    0.0,
    0.382,
    0.579,
    0.733,
    0.863,
    0.979,
    1.087,
    1.191,
    1.291,
    1.389,
    1.487,
    1.584,
    1.683,
    1.783,
    1.884,
    1.989,
    2.098,
    2.214,
    2.338,
    2.478,
    2.65,
    3.188,
])

binning_HPScan_17bins = np.array(  # Latest training; 03 Sep 2025 (17 bins) having yields ~0.6 in each bin
    [
        0.0,
        0.435,
        0.655,
        0.826,
        0.972,
        1.105,
        1.233,
        1.355,
        1.476,
        1.596,
        1.719,
        1.842,
        1.97,
        2.104,
        2.249,
        2.409,
        2.606,
        3.188,
    ]
)

binning_HPScan_13bins = np.array([  #Latest training; 03 Sep 2025 (13 bins)
        0.0,
        0.511,
        0.765,
        0.962,
        1.136,
        1.298,
        1.457,
        1.614,
        1.775,
        1.94,
        2.115,
        2.309,
        2.539,
        3.188,
    ]
)

binning_August = np.array(  # _August DNN training
    [
        0.0,
        0.564,
        0.84,
        1.059,
        1.255,
        1.442,
        1.629,
        1.819,
        2.018,
        2.236,
        2.492,
        3.188,
    ]
)

binning_DNN_HIG19006 = np.array([
    0,
    0.07,
    0.432,
    0.71,
    0.926,
    1.114,
    1.28,
    1.428,
    1.564,
    1.686,
    1.798,
    1.9,
    2.0,
    2.8,
])

# binning = binning_HPScan_21bins
# binning = binning_HPScan_13bins
# binning = binning_HPScan_17bins
# binning = binning_based_on_significanceScan
binning = binning_based_on_significanceScanV2  # 17 bins; one used for September 25, 2025 HiggsMuMu working group meeting.

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115) & (dimuon_mass < 135)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115)) | ((dimuon_mass > 135) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events


def applyRegionCatCuts(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool,
):
    use_var = "nominal" if (isinstance(variation, str) and variation.startswith("wgt")) else variation

    # Helper to fetch the right column, falling back to _nominal or base if needed
    def varcol(base):
        """
        Fetch the appropriate column from the events object, handling variations.

        Attempts to retrieve the column named '{base}_{use_var}', falling back to '{base}_nominal' and then '{base}'.
        Raises a KeyError if none of these columns are present in events.fields.

        Parameters
        ----------
        base : str
            The base name of the column to retrieve.

        Returns
        -------
        awkward.Array
            The selected column from the events object.

        Raises
        ------
        KeyError
            If none of the candidate columns are found in events.fields.
        """
        # print(f"Fetching variable column for: {base}")
        # print(f"Using variation: {use_var}")
        for cand in (f"{base}_{use_var}", f"{base}_nominal", base):
            if cand in events.fields:
                return events[cand]
        raise KeyError(
            f"[selection] Missing required field for selection: tried {base}_{use_var}, {base}_nominal, {base}"
        )

    # do mass region cut
    mass = events.dimuon_mass
    z_peak = (mass > 70) & (mass < 110)
    h_sidebands = ((mass > 110) & (mass < 115)) | ((mass > 135) & (mass < 150))
    h_peak = (mass > 115) & (mass < 135)
    if region_name == "signal":
        region = h_sidebands | h_peak
    elif region_name == "h-peak":
        region = h_peak
    elif region_name == "h-sidebands":
        region = h_sidebands
    elif region_name == "z-peak":
        region = z_peak
    else:
        print("ERROR: Invalid region specified. Acceptable regions are: signal, h-peak, h-sidebands, z-peak")
        raise ValueError

    # --- category cuts: USE varcol(...) for JES/JER-affected columns ---
    nbt_loose = varcol("nBtagLoose")
    nbt_medium = varcol("nBtagMedium")
    jj_mass = varcol("jj_mass")
    jj_dEta = varcol("jj_dEta")
    jet1_pt = varcol("jet1_pt")
    njets = varcol("njets")  # if you cut on it anywhere

    # do category cut
    if category == "nocat":
        # print("nocat mode!")
        prod_cat_cut = ak.ones_like(region, dtype="bool")
        # prod_cat_cut = ak.fill_none(events[f"jj_mass_{variation}"] > 400, value=False)
        # prod_cat_cut = prod_cat_cut & ak.fill_none(events[f"jet1_pt_{variation}"] > 35, value=False)

    else:  # VBF or ggH
        prod_cat_cut = ak.ones_like(region, dtype="bool")
        # NOTE: fatjet and MET veto for VH: nfatJets_drmuon == 0 and MET_pt < 150 GeV
        fatjet_veto = ak.fill_none((events.nfatJets_drmuon == 0), value=False)
        met_veto = ak.fill_none((events.MET_pt < 150), value=False)
        # prod_cat_cut = prod_cat_cut & fatjet_veto
        # prod_cat_cut = prod_cat_cut & met_veto
        # prod_cat_cut = prod_cat_cut & fatjet_veto & met_veto

        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none((njets >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter

        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (jj_mass > 400) & (jj_dEta > 2.5) & (jet1_pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut = prod_cat_cut & vbf_cut
            prod_cat_cut = (
                prod_cat_cut & (~btag_cut)
            )  # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & ~vbf_cut
            prod_cat_cut = (
                prod_cat_cut & (~btag_cut)
            )  # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            print("Error: invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'.")
            raise ValueError("Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'.")

    if do_vbf_filter_study:
        if "dy_" in process:
            vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
            is_vbf_filter = ("dy_VBF_filter" in process) or (
                process == "dy_m105_160_vbf_amc"
            )
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")

                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                # print(f"cutting off inclusive dy: {process}")
                prod_cat_cut = prod_cat_cut & ~vbf_filter
        else:
            # print(f"no extra processing for {process}")
            pass

    category_selection = prod_cat_cut & region
    # filter events fro selected category

    # print(f"len(events) {process} b4 selection: {len(events)}")
    events = events[category_selection]
    return events
