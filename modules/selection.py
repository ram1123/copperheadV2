import numpy as np
import awkward as ak

# Binning for DNN scores
binning = np.array(
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

binning_vbf_v0 = np.array(
    [
        0.0,
        0.511,
        0.764,
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

binning_old = np.array([
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
    # 2.1,
    # 2.2,
    # 2.3,
    2.4,
    # 2.5,
    # 2.6,
    # 2.7,
    2.8,
])

binning_v1 = np.array([
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

binning_DNNTrainedWith2018Only = np.array([
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
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
])


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
        print("ERROR: acceptable region!")
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
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none((njets >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # in the future none values will be replaced with False
        vbf_cut = (jj_mass > 400) & (jj_dEta > 2.5) & (jet1_pt > 35)
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        if category == "vbf":
            # print("vbf mode!")
            prod_cat_cut = vbf_cut
            prod_cat_cut = (
                prod_cat_cut & (~btag_cut)
            )  # btag cut is for VH and ttH categories
        elif category == "ggh":
            # print("ggH mode!")
            prod_cat_cut = ~vbf_cut
            prod_cat_cut = (
                prod_cat_cut & (~btag_cut)
            )  # btag cut is for VH and ttH categories
        else:
            print("Error: invalid category option!")
            raise ValueError

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
