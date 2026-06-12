import numpy as np
import awkward as ak
import pandas as pd


def filterRegion(events, region="h-peak"):
    if isinstance(events, pd.DataFrame):
        fields = events.columns
    else: # awkward zip
        fields = events.fields  
    if "dimuon_mass" not in fields:
        raise ValueError("dimuon_mass not found in events fields for region selection.")
    dimuon_mass = events["dimuon_mass"]
    z_peak = (dimuon_mass >= 70.0) & (dimuon_mass < 110.0)
    h_peak = (dimuon_mass >= 115.0) & (dimuon_mass < 135.0)
    h_sidebands = ((dimuon_mass >= 110.0) & (dimuon_mass < 115.0)) | (
        (dimuon_mass >= 135.0) & (dimuon_mass < 150.0)
    )
    if region == "z-peak":
        mask = z_peak
    elif region == "h-peak":
        mask = h_peak
    elif region == "h-sidebands":
        mask = h_sidebands
    elif region == "signal":
        mask = h_sidebands | h_peak
    elif region == "full":
        mask = z_peak | h_sidebands | h_peak
    else:
        raise ValueError(
            f"Invalid region selection: {region}. Valid options are: 'z-peak', 'h-peak', 'h-sidebands', 'signal', 'full'."
        )

    return mask, events[mask]


def applyRegionCatCuts(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    do_VH_veto: bool = False,
    jj_eta_region: str = "all",
    njets_selection: str = "inclusive",  # available options ["inclusive", "0", "1", "2"],
):
    use_var = (
        "nominal"
        if (isinstance(variation, str) and variation.startswith("wgt"))
        else variation
    )

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
    region, _ = filterRegion(events, region=region_name)

    # --- category cuts: USE varcol(...) for JES/JER-affected columns ---
    nbt_loose = varcol("nBtagLoose")
    nbt_medium = varcol("nBtagMedium")
    jj_mass = varcol("jj_mass")
    jj_dEta = varcol("jj_dEta")
    jet1_pt = varcol("jet1_pt")
    njets = varcol("njets")

    prod_cat_cut = ak.ones_like(region, dtype="bool")

    # do category cut
    if category == "nocat":
        prod_cat_cut = prod_cat_cut  # no additional cut
    else:  # VBF or ggH
        if do_VH_veto:
            print("Applying VH veto!")
            # NOTE: fatjet and MET veto for VH: nfatJets_drmuon == 0 and MET_pt < 150 GeV
            fatjet_veto = ak.fill_none((events.nfatJets_drmuon == 0), value=False)
            met_veto = ak.fill_none((events.MET_pt < 150), value=False)

            # INFO: Apply both fatjet and MET vetoes together
            prod_cat_cut = prod_cat_cut & fatjet_veto & met_veto

        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none(
            (njets >= 2), value=False
        )
        btag_cut = btagLoose_filter | btagMedium_filter

        vbf_cut = (jj_mass > 400) & (jj_dEta > 2.5) & (jet1_pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)

        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut = prod_cat_cut & vbf_cut
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & (~vbf_cut)
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        elif category == "bJetVeto":
            # print("ggH mode!")
            prod_cat_cut = prod_cat_cut & (
                ~btag_cut
            )  # btag cut is for VH and ttH categories
        else:
            raise ValueError(
                "Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )

    if do_vbf_filter_study:
        if process.startswith("dy"):
            vbf_filter = ak.fill_none((events.gjj_mass > 350), value=False)
            is_vbf_filter = ("dy_VBF_filter" in process)
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")

                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                prod_cat_cut = prod_cat_cut & (~vbf_filter)
            print(f"[INFO] applying VBF filter cut for {process}!")
    # ---------------------------------------------------------
    #  Select events based on number of jets
    # ---------------------------------------------------------
    if njets_selection != "inclusive":
        if njets_selection == "0":
            njets_mask = (njets == 0)
        elif njets_selection == "1":
            njets_mask = (njets == 1)
        elif njets_selection == "2":
            njets_mask = (njets >= 2)
        else:
            raise ValueError(
                f"Invalid njets_selection='{njets_selection}'. Valid options: 'inclusive', '0', '1', '2'."
            )
        prod_cat_cut = prod_cat_cut & ak.fill_none(njets_mask, value=False)

    # ---------------------------------------------------------
    #  jet-eta region selection (pair topology)
    # ---------------------------------------------------------
    if jj_eta_region and jj_eta_region != "all":

        # 1) prefer precomputed mask if present
        if jj_eta_region in events.fields:
            jj_eta_mask = ak.fill_none(events[jj_eta_region], value=False)

        else:
            # 2) compute from jet1_eta/jet2_eta (variation-safe)
            jet1_eta = varcol("jet1_eta")
            jet2_eta = varcol("jet2_eta")

            a1 = abs(jet1_eta)
            a2 = abs(jet2_eta)

            # basic regions
            j1_c = a1 < 2.5
            j2_c = a2 < 2.5

            j1_f25 = a1 > 2.5
            j2_f25 = a2 > 2.5

            j1_he = (a1 > 2.5) & (a1 < 3.0)
            j2_he = (a2 > 2.5) & (a2 < 3.0)

            j1_f30 = a1 > 3.0
            j2_f30 = a2 > 3.0

            masks = {
                "jj_both_central": j1_c & j2_c,
                "jj_non_central": ~ (j1_c & j2_c),
                "jj_one_fwd25_one_central": (j1_f25 & j2_c) | (j2_f25 & j1_c),
                "jj_one_he_one_central": (j1_he & j2_c) | (j2_he & j1_c),
                "jj_one_fwd30_one_central": (j1_f30 & j2_c) | (j2_f30 & j1_c),
                "jj_both_fwd25": j1_f25 & j2_f25,
                "jj_both_he": j1_he & j2_he,
                "jj_both_fwd30": j1_f30 & j2_f30,
                "jj_one_he_one_fwd30": (j1_he & j2_f30) | (j2_he & j1_f30),
            }

            if jj_eta_region not in masks:
                raise ValueError(
                    f"Invalid jj_eta_region='{jj_eta_region}'. "
                    f"Valid: all, {', '.join(masks.keys())}"
                )

            jj_eta_mask = ak.fill_none(masks[jj_eta_region], value=False)

        prod_cat_cut = prod_cat_cut & jj_eta_mask

    category_selection = prod_cat_cut & region
    events = events[category_selection]
    return events


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



# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026
#     0.000000,
#   0.518166,
#   1.036333,
#   1.554499,
#   2.072665,
#   2.590832,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May19_2026
#     0.000000,
#   0.574031,
#   1.051195,
#   1.528359,
#   2.005523,
#   (7.254329+0.1),
# ])

# binning = np.array([ # Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026
#   (-7.254329-0.1),
#   -6.287085,
#   -1.450866,
#   -0.483622,
#   0.483622,
#   1.450866,
#   2.418110,
#   5.319841,
#   (7.254329+0.1),
# ])


# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_June01_2026
#   0.000000,
#   0.483622,
#   0.967244,
#   1.450866,
#   1.934488,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_June01_2026 NOTE: found it surprising that the results were same, but we checked the stage1 output and DNN model paths and they were correct
#   0.000000,
#   0.483622,
#   0.967244,
#   1.450866,
#   1.934488,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc june 04 2026
#   0.000000,
#   0.164871,
#   0.329742,
#   0.494613,
#   0.659484,
#   0.824356,
#   0.989227,
#   1.154098,
#   1.318969,
#   1.483840,
#   1.648711,
#   1.813582,
#   1.978453,
#   (7.254329+0.1),
# ])




# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc june 04 2026
#   0.000000,
#   0.148048,
#   0.296095,
#   0.444143,
#   0.592190,
#   0.740238,
#   0.888285,
#   1.036333,
#   1.184380,
#   1.332428,
#   1.480475,
#   1.628523,
#   1.776570,
#   1.924618,
#   2.072665,
#   2.220713,
#   (7.254329+0.1),
# ])



# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins june 07 2026
#   0.000000,
#   0.127269,
#   0.254538,
#   0.381807,
#   0.509076,
#   0.636345,
#   0.763614,
#   0.890882,
#   1.018151,
#   1.145420,
#   1.272689,
#   1.399958,
#   1.527227,
#   1.654496,
#   1.781765,
#   1.909034,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins june 07 2026
#   0.000000,
#   0.127269,
#   0.190903,
#   0.254538,
#   0.318172,
#   0.381807,
#   0.509076,
#   0.572710,
#   0.699979,
#   0.827248,
#   0.954517,
#   1.018151,
#   1.145420,
#   1.336324,
#   1.463593,
#   1.527227,
#   1.654496,
#   1.781765,
#   1.909034,
#   1.972668,
#   2.099937,
#   (7.254329+0.1),
# ])


# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll june 08 2026. it has 24 bins
#   0.000000,
#   0.102174,
#   0.153260,
#   0.204347,
#   0.306521,
#   0.357608,
#   0.459781,
#   0.510868,
#   0.613042,
#   0.715215,
#   0.817389,
#   0.919563,
#   1.072823,
#   1.174997,
#   1.277171,
#   1.430431,
#   1.532605,
#   1.685865,
#   1.788039,
#   1.941299,
#   1.992386,
#   2.043473,
#   2.094560,
#   2.196733,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins june 08 2026. it has 17 bins
#   0.000000,
#   0.129542,
#   0.259083,
#   0.388625,
#   0.518166,
#   0.647708,
#   0.777249,
#   0.906791,
#   1.036333,
#   1.165874,
#   1.295416,
#   1.424957,
#   1.554499,
#   1.684041,
#   1.813582,
#   1.943124,
#   2.202207,
#   (7.254329+0.1),
# ])



# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_7Bins june 08 2026. it has 7 bins
#   0.000000,
#   0.315406,
#   0.630811,
#   0.946217,
#   1.261622,
#   1.577028,
#   1.892434,
#   (7.254329+0.1),
# ])



# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins june 08 2026. it has 11 bins
#   0.000000,
#   0.190903,
#   0.381807,
#   0.572710,
#   0.763614,
#   0.954517,
#   1.145420,
#   1.336324,
#   1.527227,
#   1.718130,
#   1.909034,
#   (7.254329+0.1),
# ])


# binning = np.array([ #  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_Max40bins june 08 2026. it has 11 bins
#   0.000000,
#   0.196063,
#   0.392126,
#   0.588189,
#   0.784252,
#   0.980315,
#   1.176378,
#   1.372441,
#   1.568503,
#   1.764566,
#   1.960629,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc June 11 2026. max n bins 57
#   0.000000,
#   0.154347,
#   0.308695,
#   0.463042,
#   0.617390,
#   0.771737,
#   0.926085,
#   1.080432,
#   1.234779,
#   1.389127,
#   1.543474,
#   1.697822,
#   1.852169,
#   2.006516,
#   (7.254329+0.1),
# ])

# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc June 11 2026. max n bins 57
#   0.000000,
#   0.148048,
#   0.296095,
#   0.444143,
#   0.592190,
#   0.740238,
#   0.888285,
#   1.036333,
#   1.184380,
#   1.332428,
#   1.480475,
#   1.628523,
#   1.776570,
#   1.924618,
#   2.072665,
#   2.220713,
#   (7.254329+0.1),
# ])


# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc June 11 2026. max n bins 100
#   0.000000,
#   0.086361,
#   0.172722,
#   0.259083,
#   0.345444,
#   0.431805,
#   0.604527,
#   0.777249,
#   0.949972,
#   1.122694,
#   1.209055,
#   1.381777,
#   1.554499,
#   1.640860,
#   1.727221,
#   1.813582,
#   1.986304,
#   2.072665,
#   2.159026,
#   2.245387,
#   (7.254329+0.1),
# ])


# binning = np.array([ #  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc June 11 2026. max n bins 70
#   0.000000,
#   0.118923,
#   0.237847,
#   0.356770,
#   0.475694,
#   0.594617,
#   0.713541,
#   0.832464,
#   0.951387,
#   1.070311,
#   1.189234,
#   1.308158,
#   1.427081,
#   1.546004,
#   1.664928,
#   1.783851,
#   1.902775,
#   2.021698,
#   2.259545,
#   (7.254329+0.1),
# ])

binning = np.array([ #  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc June 11 2026. max n bins 70
  0.000000,
  0.103633,
  0.207267,
  0.310900,
  0.414533,
  0.518166,
  0.621800,
  0.725433,
  0.829066,
  0.932699,
  1.036333,
  1.139966,
  1.243599,
  1.347232,
  1.450866,
  1.554499,
  1.658132,
  1.865399,
  1.969032,
  2.176299,
  (7.254329+0.1),
])
