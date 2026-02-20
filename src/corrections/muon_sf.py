import numpy as np
import awkward as ak
import dask_awkward as dak
import correctionlib
import dask

from src.corrections.correctionlib_file_cache import get_corrset
from modules.utils import logger, get_corr_input_names


def add_muon_sfs_correctionlib(mu1, mu2, mu_sf_lookup_info, year: str):
    """
    Add Run-3 muon SFs using correctionlib.

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
    mu_sf_lookup_file = mu_sf_lookup_info.get("json_file")
    id_name = mu_sf_lookup_info.get("id")
    iso_name = mu_sf_lookup_info.get("iso")
    trig_name = mu_sf_lookup_info.get("trig")

    logger.info("Muon SF lookup info from config: %s", mu_sf_lookup_info)
    logger.info(
        f"Muon ID SF correction name: {id_name}\n"
        f"Muon Iso SF correction name: {iso_name}\n"
        f"Muon Trigger SF correction name: {trig_name}"
    )
    logger.info(f"mu_sf_lookup_file: {mu_sf_lookup_file}")

    corrset = get_corrset(mu_sf_lookup_file)
    muID_corr = corrset[id_name]
    muIso_corr = corrset[iso_name]
    muTrig_corr = corrset[trig_name]
    logger.info(f"get_corr_input_names muID_corr: {get_corr_input_names(muID_corr)}")
    logger.info(f"get_corr_input_names muIso_corr: {get_corr_input_names(muIso_corr)}")
    logger.info(f"get_corr_input_names muTrig_corr: {get_corr_input_names(muTrig_corr)}")

    # -----------------------------
    # ID (event = mu1*mu2)
    # -----------------------------
    mu1_id_nom = muID_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "nominal")
    mu1_id_up = muID_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "systup")
    mu1_id_down = muID_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "systdown")

    mu2_id_nom = muID_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "nominal")
    mu2_id_up = muID_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "systup")
    mu2_id_down = muID_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "systdown")

    muID = {
        "nom": mu1_id_nom * mu2_id_nom,
        "up": mu1_id_up * mu2_id_up,
        "down": mu1_id_down * mu2_id_down,
    }

    # -----------------------------
    # ISO (event = mu1*mu2)
    # -----------------------------
    mu1_iso_nom = muIso_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "nominal")
    mu1_iso_up = muIso_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "systup")
    mu1_iso_down = muIso_corr.evaluate(mu1.eta_raw, mu1.pt_raw, "systdown")

    mu2_iso_nom = muIso_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "nominal")
    mu2_iso_up = muIso_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "systup")
    mu2_iso_down = muIso_corr.evaluate(mu2.eta_raw, mu2.pt_raw, "systdown")

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
    in_bounds = (mu1.pt_raw > 26.0) & (abs(mu1.eta_raw) < 2.4)

    # Clip inputs to valid range to prevent correctionlib from throwing errors
    # (we only use the computed SF where in_bounds=True; out-of-bounds get SF=1.0)
    if year == "2017":
        min_trig_pt = 29.0
    else:
        min_trig_pt = 26.0
    pt_safe = ak.where(mu1.pt_raw > min_trig_pt, mu1.pt_raw, (min_trig_pt + .01))
    eta_safe = ak.where(abs(mu1.eta_raw) < 2.4, mu1.eta_raw, 0.0)

    # Evaluate trigger SFs using safe inputs
    mu_trig_nom_eval = muTrig_corr.evaluate(eta_safe, pt_safe, "nominal")
    mu_trig_up_eval = muTrig_corr.evaluate(eta_safe, pt_safe, "systup")
    mu_trig_down_eval = muTrig_corr.evaluate(eta_safe, pt_safe, "systdown")

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
            f"(pt<=26 or |eta|>=2.4), assigning SF=1."
        )

    muTrig = {"nom": mu_trig_nom, "up": mu_trig_up, "down": mu_trig_down}

    return muID, muIso, muTrig
