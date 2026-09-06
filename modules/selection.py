import numpy as np
import awkward as ak
import pandas as pd
from modules.classify_year import is_run3

# jj_eta_region names using both jet1 AND jet2 (only meaningful for njets>=2 --
# jet2 is null for njets<2, so these silently evaluate False there, see
# applyRegionCatCuts). Canonical source of truth for these names -- mirrored
# (not imported, Snakemake reads it as plain CLI strings) in
# workflow/Snakefile's PAIR_JJ_ETA_REGIONS.
PAIR_JJ_ETA_REGIONS = [
    "jj_both_central",
    "jj_non_central",
    "jj_one_fwd25_one_central",
    "jj_one_he_one_central",
    "jj_one_fwd30_one_central",
    "jj_both_fwd25",
    "jj_both_he",
    "jj_both_fwd30",
    "jj_one_he_one_fwd30",
]

# jj_eta_region names using jet1 alone, with njets==1 baked directly into the
# mask (see applyRegionCatCuts) -- deliberately self-gating rather than
# relying on the caller to also pass njets_selection="1", so combining one of
# these with any other njets_selection can't silently return nonsense: it
# either matches the njets==1 subset (njets_selection in ("inclusive", "1"))
# or is provably empty (njets_selection in ("0", "2")). Canonical source of
# truth, mirrored in workflow/Snakefile's SINGLE_JET_ETA_REGIONS.
SINGLE_JET_ETA_REGIONS = [
    "single_central",
    "single_fwd25",
    "single_he",
    "single_fwd30",
]


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
    year: str | None = None,
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
        process_lower = process.lower()
        if process_lower.startswith("dy"):
            gjj_threshold = 300 if (year is not None and is_run3(year)) else 350
            vbf_filter = ak.fill_none((events.gjj_mass > gjj_threshold), value=False)
            is_vbf_filter = "dy_vbf_filter" in process_lower
            if is_vbf_filter:
                # print(f"applying VBF filter cut on: {process}")

                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                prod_cat_cut = prod_cat_cut & (~vbf_filter)

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
    #  jet-eta region selection (pair topology for njets>=2, single-jet
    #  topology for njets==1; a 0-jet event has nothing to region-split)
    # ---------------------------------------------------------
    # A 0-jet selection can never satisfy any region mask (there are no jets
    # to check |eta| on) -- reject rather than silently return an empty,
    # confusing-looking selection.
    if njets_selection == "0" and jj_eta_region and jj_eta_region != "all":
        raise ValueError(
            f"jj_eta_region='{jj_eta_region}' is incompatible with njets_selection='0' "
            "-- a 0-jet selection has no jets to region-split; use jj_eta_region='all'."
        )
    # A 1-jet selection only has a single real jet (jet2 is null), so a
    # PAIR_JJ_ETA_REGIONS mask (needs jet1 AND jet2) can only ever be empty
    # there -- reject rather than silently return zero events.
    if njets_selection == "1" and jj_eta_region in PAIR_JJ_ETA_REGIONS:
        raise ValueError(
            f"jj_eta_region='{jj_eta_region}' is a two-jet condition, incompatible with "
            f"njets_selection='1' -- use one of {SINGLE_JET_ETA_REGIONS} instead."
        )
    # Symmetric case: a SINGLE_JET_ETA_REGIONS mask bakes njets==1 into its
    # own definition (see below), so it can only ever be empty against a
    # njets>=2 selection -- reject for the same "don't silently return zero
    # events" reason.
    if njets_selection == "2" and jj_eta_region in SINGLE_JET_ETA_REGIONS:
        raise ValueError(
            f"jj_eta_region='{jj_eta_region}' is a single-jet (njets==1) condition, "
            f"incompatible with njets_selection='2' -- use one of {PAIR_JJ_ETA_REGIONS} "
            "(or 'all') instead."
        )

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

            # basic regions -- boundaries deliberately half-open (<=/> ) so
            # every |eta| value lands in exactly one of central/fwd25, and
            # exactly one of he/fwd30 (an event with |eta| exactly 2.5 or 3.0
            # -- rare but real, confirmed on data: 12/23692 single-jet
            # ttjets_dl 2026 events sit exactly at 2.5 -- used to fall into
            # neither bucket of either pair with the old strict <>/<> split).
            j1_c = a1 <= 2.5
            j2_c = a2 <= 2.5

            j1_f25 = a1 > 2.5
            j2_f25 = a2 > 2.5

            j1_he = (a1 > 2.5) & (a1 <= 3.0)
            j2_he = (a2 > 2.5) & (a2 <= 3.0)

            j1_hf = a1 > 3.0
            j2_hf = a2 > 3.0

            # njets==1 => jet1 is the one real jet (leading-pT slot, always
            # filled first when any jet exists) and jet2 is null -- baked
            # directly into the single-jet masks below so they're correct
            # regardless of what njets_selection the caller passed (see
            # SINGLE_JET_ETA_REGIONS docstring at the top of this module).
            is_single_jet = (njets == 1)

            masks = {
                "jj_both_central": j1_c & j2_c,
                "jj_non_central": ~ (j1_c & j2_c),
                "jj_one_fwd25_one_central": (j1_f25 & j2_c) | (j2_f25 & j1_c),
                "jj_one_he_one_central": (j1_he & j2_c) | (j2_he & j1_c),
                "jj_one_fwd30_one_central": (j1_hf & j2_c) | (j2_hf & j1_c),
                "jj_both_fwd25": j1_f25 & j2_f25,
                "jj_both_he": j1_he & j2_he,
                "jj_both_fwd30": j1_hf & j2_hf,
                "jj_one_he_one_fwd30": (j1_he & j2_hf) | (j2_he & j1_hf),
                "single_central": is_single_jet & j1_c,
                "single_fwd25": is_single_jet & j1_f25,
                "single_he": is_single_jet & j1_he,
                "single_fwd30": is_single_jet & j1_hf,
            }
            assert set(masks.keys()) == set(PAIR_JJ_ETA_REGIONS) | set(SINGLE_JET_ETA_REGIONS), (
                "masks dict drifted from the module-level PAIR_JJ_ETA_REGIONS/"
                "SINGLE_JET_ETA_REGIONS name lists -- keep them in sync."
            )

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


def apply_jet_horn_ptcut(
    events,
    he_pt_cut: float | None = None,
    hf_pt_cut: float | None = None,
    variation: str = "nominal",
    max_jet_slots: int = 4,
):
    """
    Post-hoc HE/HF forward-jet pT mitigation cut, for validating stage-1 output
    that was produced with a looser jet pT threshold than the official JME
    mitigation (see the jme-horn-region-official-recommendation notes) --
    without needing to rerun stage-1.

    Regions (matching `jetHorn_region` in src/copperhead_processor.py's
    jet_loop, and the HE/HF split from the JME "Mitigation techniques" slide):
      HE: 2.5 < |eta| <= 3.0
      HF: |eta| > 3.0
    A jet in one of these regions with pt below the corresponding threshold is
    treated as if it had not passed the jet selection at all; jets with
    |eta| <= 2.5 are never affected. Passing `None` for a threshold disables
    the cut for that region (e.g. he_pt_cut=50, hf_pt_cut=None applies the cut
    to HE only).

    Scope / what this does NOT do: stage-1 only ever saves the leading 2 (or 4,
    if `save_four_jets_kinematics` was on) jets as flat jet1..jet4 columns, not
    the full per-event jet collection. So a jet beyond the saved slots that
    would have been promoted into jet1..jet4 after this cut can't be recovered
    -- `njets_{variation}` is decremented by however many of the *saved* slots
    got cut, which undercounts the true effect for events with more real jets
    than saved slots. Only jet{i}_pt/eta/phi/mass and njets are remapped;
    everything derived from the original jet1/jet2 pairing (jj_mass, jj_dEta,
    zeppenfeld, mmj_*_dEta/dPhi/dR, rpt, pt_centrality, puId, rapidity, ...) is
    left untouched and will still reflect the pre-cut jet1/jet2 identities --
    treat pairwise/topology variables as stale/approximate when this is used.

    Parameters
    ----------
    events : awkward.Array or dask_awkward.Array
        Loaded stage-1/compacted ntuple.
    he_pt_cut, hf_pt_cut : float or None
        pT threshold (GeV) for the HE / HF region; None disables that region's cut.
    variation : str
        Column suffix to operate on (only "nominal" is meaningful for most
        validation-plotting use cases, since jet1_pt_nominal etc. is what's loaded).
    max_jet_slots : int
        Highest jet slot to consider (1-4); slots whose pt/eta columns aren't
        present in `events.fields` are skipped.

    Returns
    -------
    Same type as `events`, with jet{i}_pt/eta/phi/mass_{variation} and
    njets_{variation} replaced by the post-cut, re-compacted values.
    """
    if he_pt_cut is None and hf_pt_cut is None:
        return events  # no-op

    slots = [
        i for i in range(1, max_jet_slots + 1)
        if f"jet{i}_pt_{variation}" in events.fields and f"jet{i}_eta_{variation}" in events.fields
    ]
    if not slots:
        raise KeyError(
            f"apply_jet_horn_ptcut: no jet{{i}}_pt_{variation}/jet{{i}}_eta_{variation} "
            "columns found in events.fields."
        )

    # attributes to remap in lockstep with pt/eta wherever they were saved for
    # every considered slot (see jet_loop's unconditional jet1/jet2 + gated
    # jet3/jet4 kinematics block in src/copperhead_processor.py)
    extra_attrs = [
        a for a in ("phi", "mass")
        if all(f"jet{i}_{a}_{variation}" in events.fields for i in slots)
    ]

    # -999.0 sentinel for "no jet in this slot" -- robust to either an awkward
    # None (padded slot) or a bare NaN surviving the parquet round-trip, since
    # both compare False against `> -900` below.
    def col(name):
        return ak.fill_none(events[name], -999.0)

    pt_per_slot = {i: col(f"jet{i}_pt_{variation}") for i in slots}
    eta_per_slot = {i: col(f"jet{i}_eta_{variation}") for i in slots}
    extra_per_slot = {
        a: {i: col(f"jet{i}_{a}_{variation}") for i in slots} for a in extra_attrs
    }

    has_jet = {i: pt_per_slot[i] > -900.0 for i in slots}
    fails_cut = {}
    for i in slots:
        abs_eta = abs(eta_per_slot[i])
        in_he = (abs_eta > 2.5) & (abs_eta <= 3.0)
        in_hf = abs_eta > 3.0
        fail_he = (in_he & (pt_per_slot[i] < he_pt_cut)) if he_pt_cut is not None else (in_he & False)
        fail_hf = (in_hf & (pt_per_slot[i] < hf_pt_cut)) if hf_pt_cut is not None else (in_hf & False)
        fails_cut[i] = has_jet[i] & (fail_he | fail_hf)
    passes = {i: has_jet[i] & ~fails_cut[i] for i in slots}

    # stack slots into a jagged (nevents, n_slots) array, drop the failing
    # entries per event (this compacts the survivors, preserving pT-descending
    # order since we're only ever removing entries from an already-sorted
    # list), then pad back out to a fixed number of columns.
    n_slots = len(slots)

    def stack(per_slot_dict):
        # ak.concatenate(axis=1) here produces a *regular* (fixed-size) typed
        # array; boolean-masking a regular array with a same-shaped regular
        # mask uses numpy's flat semantics (global flatten) instead of the
        # per-event jagged compaction we need, so force it to `var` type.
        regular = ak.concatenate([per_slot_dict[i][:, None] for i in slots], axis=1)
        return ak.from_regular(regular, axis=1)

    pass_stack = stack(passes)
    fail_stack = stack(fails_cut)

    n_dropped = ak.sum(fail_stack, axis=1)
    njets_col = f"njets_{variation}"
    if njets_col in events.fields:
        events[njets_col] = events[njets_col] - n_dropped

    for attr, per_slot in {"pt": pt_per_slot, "eta": eta_per_slot, **extra_per_slot}.items():
        stacked = stack(per_slot)
        compacted = ak.pad_none(stacked[pass_stack], target=n_slots, axis=1)
        for new_idx, orig_i in enumerate(slots):
            events[f"jet{orig_i}_{attr}_{variation}"] = ak.fill_none(compacted[:, new_idx], -999.0)

    return events


def applyRegionCatCutsByScore(
    events,
    category: str,
    region_name: str,
    process: str,
    variation: str,
    do_vbf_filter_study: bool = False,
    year: str | None = None,
    do_VH_veto: bool = False,
    jj_eta_region: str = "all",
    njets_selection: str = "inclusive",
):
    """
    Apply the same region-level selection as `applyRegionCatCuts`, but assign
    ggH/VBF categories using transformer scores instead of the cut-based VBF
    definition.

    Strategy:
    - if transf_vbf_score > transf_ggh_score, tag as VBF
    - otherwise tag as ggH
    """
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

    required_fields = {"transf_vbf_score", "transf_ggh_score"}
    missing_fields = sorted(required_fields - set(events.fields))
    if missing_fields:
        raise KeyError(
            "Missing transformer score field(s) required for score-based "
            f"categorization: {missing_fields}"
        )

    vbf_score = ak.fill_none(events["transf_vbf_score"], float("-inf"))
    ggh_score = ak.fill_none(events["transf_ggh_score"], float("-inf"))
    # is_vbf = ak.fill_none(vbf_score > ggh_score, value=False)
    is_vbf = ak.fill_none((vbf_score/(vbf_score + ggh_score)) > 0.92522, value=False)
    # is_vbf = ak.fill_none(vbf_score > 0.925, value=False)

    if category == "nocat":
        prod_cat_cut = prod_cat_cut  # no additional cut
    else:
        # NOTE: btag cut for VH and ttH categories
        btagLoose_filter = ak.fill_none((nbt_loose >= 2), value=False)
        btagMedium_filter = ak.fill_none((nbt_medium >= 1), value=False) & ak.fill_none(
            (njets >= 2), value=False
        )
        btag_cut = btagLoose_filter | btagMedium_filter

        if category == "vbf":
            prod_cat_cut = prod_cat_cut & is_vbf
            prod_cat_cut = prod_cat_cut & (~btag_cut)         
        elif category == "ggh":
            prod_cat_cut = prod_cat_cut & (~is_vbf)
            prod_cat_cut = prod_cat_cut & (~btag_cut)        
        else:
            raise ValueError(
                "Invalid category option! Valid options are: 'vbf', 'ggh', 'nocat'."
            )

    if do_vbf_filter_study:
        process_lower = process.lower()
        if process_lower.startswith("dy"):
            gjj_threshold = 300 if (year is not None and is_run3(year)) else 350
            vbf_filter = ak.fill_none((events.gjj_mass > gjj_threshold), value=False)
            is_vbf_filter = "dy_vbf_filter" in process_lower
            if is_vbf_filter:
                prod_cat_cut = prod_cat_cut & vbf_filter
            else:
                prod_cat_cut = prod_cat_cut & (~vbf_filter)

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