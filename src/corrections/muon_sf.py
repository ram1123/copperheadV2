import numpy as np
import awkward as ak
import correctionlib

from modules.correctionlib_file_cache import get_corrset, get_corr_input_names
from modules.utils import logger


def _is_missing_category_error(err):
    return isinstance(err, IndexError) and "Index not available in Category" in str(err)


def _evaluate_nom_up_down(corr, eta, pt):
    """
    Evaluate nominal/systup/systdown for a correctionlib SF, tolerating a
    Category axis that's missing "systup" or "systdown" for some (eta, pt) bin
    -- seen in an early-calibration muon trigger SF payload where one narrow
    bin lacked "systdown". correctionlib fails the whole vectorized evaluate()
    call if any single event in the chunk falls in such a bin (there's no
    per-event partial result), so on failure this falls back to a symmetric
    mirror of the other variation around nominal for the whole chunk rather
    than only the offending events.
    """
    nom = corr.evaluate(eta, pt, "nominal")

    try:
        up = corr.evaluate(eta, pt, "systup")
    except Exception as err:
        if not _is_missing_category_error(err):
            raise
        logger.warning(f"correctionlib systup evaluation failed, will mirror systdown: {err}")
        up = None

    try:
        down = corr.evaluate(eta, pt, "systdown")
    except Exception as err:
        if not _is_missing_category_error(err):
            raise
        logger.warning(f"correctionlib systdown evaluation failed, will mirror systup: {err}")
        down = None

    if up is None and down is not None:
        up = nom - (down - nom)
    if down is None and up is not None:
        down = nom - (up - nom)
    if up is None and down is None:
        up = nom
        down = nom

    return nom, up, down


def add_muon_sfs_correctionlib(mu1, mu2, config):
    """
    Add muon SFs using correctionlib (supports both Run 2 and Run 3, depending on configuration).

    Convention (as in your working patch):
      - ID/Iso event SF = SF(mu1) * SF(mu2)
      - Trigger event SF = SF(leading muon only)
        (safe fallback when only SF map is available; avoids pt-threshold issues on subleading muon)

    Parameters
    ----------
    mu1, mu2 : awkward records
        Muon objects with fields `.eta` and `.pt` (can be ak.Array or dak.Array).
    mu_sf_lookup_info : dict
        Must contain:
          - "json_file": path to muon_Z.json(.gz)
          - "id": correction name
          - "iso": correction name
          - "trig": correction name
    """
    mu_sf_lookup_info = config["muSFFileList"]
    
    mu_sf_lookup_file = mu_sf_lookup_info.get("json_file")
    id_name = mu_sf_lookup_info.get("id")
    iso_name = mu_sf_lookup_info.get("iso")
    trig_name = mu_sf_lookup_info.get("trig")

    logger.debug("Muon SF lookup info from config: %s", mu_sf_lookup_info)
    logger.debug(
        f"Muon ID SF correction name: {id_name}\n"
        f"Muon Iso SF correction name: {iso_name}\n"
        f"Muon Trigger SF correction name: {trig_name}"
    )
    logger.debug(f"mu_sf_lookup_file: {mu_sf_lookup_file}")

    corrset = get_corrset(mu_sf_lookup_file)
    muID_corr = corrset[id_name]
    muIso_corr = corrset[iso_name]
    muTrig_corr = corrset[trig_name]
    logger.debug(f"get_corr_input_names muID_corr: {get_corr_input_names(muID_corr)}")
    logger.debug(f"get_corr_input_names muIso_corr: {get_corr_input_names(muIso_corr)}")
    logger.debug(f"get_corr_input_names muTrig_corr: {get_corr_input_names(muTrig_corr)}")

    # -----------------------------
    # ID (event = mu1*mu2)
    # -----------------------------
    mu1_id_nom, mu1_id_up, mu1_id_down = _evaluate_nom_up_down(muID_corr, mu1.eta_raw, mu1.pt_raw)
    mu2_id_nom, mu2_id_up, mu2_id_down = _evaluate_nom_up_down(muID_corr, mu2.eta_raw, mu2.pt_raw)

    muID = {
        "nom": mu1_id_nom * mu2_id_nom,
        "up": mu1_id_up * mu2_id_up,
        "down": mu1_id_down * mu2_id_down,
    }

    # -----------------------------
    # ISO (event = mu1*mu2)
    # -----------------------------
    mu1_iso_nom, mu1_iso_up, mu1_iso_down = _evaluate_nom_up_down(muIso_corr, mu1.eta_raw, mu1.pt_raw)
    mu2_iso_nom, mu2_iso_up, mu2_iso_down = _evaluate_nom_up_down(muIso_corr, mu2.eta_raw, mu2.pt_raw)

    muIso = {
        "nom": mu1_iso_nom * mu2_iso_nom,
        "up": mu1_iso_up * mu2_iso_up,
        "down": mu1_iso_down * mu2_iso_down,
    }

    # -----------------------------
    # TRIGGER (leading muon only)
    # Logic: If muon is within bounds, compute SF; otherwise SF=1.0
    # -----------------------------
    # Define trigger SF acceptance window

    trig_eta_upper_limit = config["muon_eta_cut"]
    trig_pt_lower_limit = config["muon_leading_pt"]
    logger.debug(f"trig_eta_upper_limit: {trig_eta_upper_limit}")
    logger.debug(f"trig_pt_lower_limit: {trig_pt_lower_limit}")
    in_bounds = (mu1.pt_raw > trig_pt_lower_limit) & (abs(mu1.eta_raw) < trig_eta_upper_limit)

    # Clip inputs to valid range to prevent correctionlib from throwing errors
    # (we only use the computed SF where in_bounds=True; out-of-bounds get SF=1.0)
    pt_safe = ak.where(mu1.pt_raw > trig_pt_lower_limit, mu1.pt_raw, (trig_pt_lower_limit + .01))
    eta_safe = ak.where(abs(mu1.eta_raw) < trig_eta_upper_limit, mu1.eta_raw, 0.0)

    # Evaluate trigger SFs using safe inputs
    mu_trig_nom_eval, mu_trig_up_eval, mu_trig_down_eval = _evaluate_nom_up_down(
        muTrig_corr, eta_safe, pt_safe
    )

    # Apply logic: use computed SF if in_bounds, otherwise use 1.0
    mu_trig_nom = ak.where(in_bounds, mu_trig_nom_eval, 1.0)
    mu_trig_up = ak.where(in_bounds, mu_trig_up_eval, 1.0)
    mu_trig_down = ak.where(in_bounds, mu_trig_down_eval, 1.0)

    # Log fraction of out-of-bounds events
    n_total = ak.count(in_bounds)
    n_out_of_bounds = ak.sum(~in_bounds)
    if n_out_of_bounds > 0:
        logger.debug(
            f"Trigger SF: {n_out_of_bounds}/{n_total} events have leading muon out of bounds "
            f"(pt<=trig_pt_lower_limit or |eta|>=trig_eta_upper_limit), assigning SF=1."
        )

    muTrig = {"nom": mu_trig_nom, "up": mu_trig_up, "down": mu_trig_down}

    return muID, muIso, muTrig


def _has_category(corr, key, eta=0.5, pt=40.0):
    """True if the correction's scale_factors category has `key` (probed eagerly on one point)."""
    try:
        corr.evaluate(eta, pt, key)
        return True
    except Exception:
        return False


def muon_sfs_stat_syst_correctionlib(mu1, mu2, config):
    """
    Run3 muon ID/Iso/trigger SFs with the MUO stat and syst components as separate nuisances
    (stat: uncorrelated across years, syst: correlated). The payload's "stat"/"syst" entries are
    absolute uncertainties; each variation shifts both muons coherently.

    Trigger (single-muon HLT, either muon can fire it):
        SF = [1 - (1 - SF1*eMC1)(1 - SF2*eMC2)] / [1 - (1 - eMC1)(1 - eMC2)]
    with eMC from "nominal_MCeff"; a muon below muon_leading_pt or outside the acceptance gets
    eff 0. Payloads without MC efficiencies (2025) fall back to the leading-muon SF.

    Returns {"muID"|"muIso"|"muTrig": {"nom": arr, "stat": (up, down), "syst": (up, down)}},
    with up/down as absolute event SFs.
    """
    info = config["muSFFileList"]
    corrset = get_corrset(info["json_file"])
    corrs = {"muID": corrset[info["id"]], "muIso": corrset[info["iso"]], "muTrig": corrset[info["trig"]]}
    mus = (mu1, mu2)

    def per_muon(corr, eta, pt):
        nom = corr.evaluate(eta, pt, "nominal")
        return nom, corr.evaluate(eta, pt, "stat"), corr.evaluate(eta, pt, "syst")

    out = {}
    for name in ("muID", "muIso"):
        (n1, st1, sy1), (n2, st2, sy2) = (per_muon(corrs[name], m.eta_raw, m.pt_raw) for m in mus)
        out[name] = {
            "nom": n1 * n2,
            "stat": ((n1 + st1) * (n2 + st2), (n1 - st1) * (n2 - st2)),
            "syst": ((n1 + sy1) * (n2 + sy2), (n1 - sy1) * (n2 - sy2)),
        }

    # Trigger: clip inputs into the SF validity window; out-of-window muons cannot fire the HLT
    trig = corrs["muTrig"]
    eta_max, pt_min = config["muon_eta_cut"], config["muon_leading_pt"]
    has_mc_eff = _has_category(trig, "nominal_MCeff")
    in_win, eff, sf, st, sy = [], [], [], [], []
    for m in mus:
        in_win.append((m.pt_raw > pt_min) & (abs(m.eta_raw) < eta_max))
        pt_safe = ak.where(m.pt_raw > pt_min, m.pt_raw, pt_min + 0.01)
        eta_safe = ak.where(abs(m.eta_raw) < eta_max, m.eta_raw, 0.0)
        n, s_st, s_sy = per_muon(trig, eta_safe, pt_safe)
        sf.append(n)
        st.append(s_st)
        sy.append(s_sy)
        if has_mc_eff:
            eff.append(ak.where(in_win[-1], trig.evaluate(eta_safe, pt_safe, "nominal_MCeff"), 0.0))

    if has_mc_eff:
        p_mc = 1.0 - (1.0 - eff[0]) * (1.0 - eff[1])
        def event_sf(s1, s2):
            p_data = 1.0 - (1.0 - np.minimum(s1 * eff[0], 1.0)) * (1.0 - np.minimum(s2 * eff[1], 1.0))
            return ak.where(p_mc > 0, p_data / ak.where(p_mc > 0, p_mc, 1.0), 1.0)
    else:
        logger.warning(f"{info['trig']} has no MC efficiencies in {info['json_file']}; "
                       "using the leading-muon trigger SF")
        def event_sf(s1, s2):
            return ak.where(in_win[0], s1, 1.0)

    out["muTrig"] = {
        "nom": event_sf(sf[0], sf[1]),
        "stat": (event_sf(sf[0] + st[0], sf[1] + st[1]), event_sf(sf[0] - st[0], sf[1] - st[1])),
        "syst": (event_sf(sf[0] + sy[0], sf[1] + sy[1]), event_sf(sf[0] - sy[0], sf[1] - sy[1])),
    }
    return out
