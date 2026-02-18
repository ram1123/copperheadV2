import numpy as np
import awkward as ak
import dask_awkward as dak
import correctionlib

from src.corrections.correctionlib_file_cache import get_corrset
from modules.utils import logger


def _is_typetracer(x):
    try:
        return ak.backend(x) == "typetracer"
    except Exception:
        return False


def add_muon_sfs_run3_correctionlib(mu1, mu2, mu_sf_lookup_info):
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

    logger.debug("Muon SF lookup info from config: %s", mu_sf_lookup_info)
    logger.info(
        f"Muon ID SF correction name: {id_name}\n"
        f"Muon Iso SF correction name: {iso_name}\n"
        f"Muon Trigger SF correction name: {trig_name}"
    )
    logger.info(f"mu_sf_lookup_file: {mu_sf_lookup_file}")

    corrset = correctionlib.CorrectionSet.from_file(mu_sf_lookup_file)
    muID_corr = corrset[id_name]
    muIso_corr = corrset[iso_name]
    muTrig_corr = corrset[trig_name]

    # -------------------------------------------------
    # dask-safe correctionlib evaluate (partition-wise)
    # -------------------------------------------------
    def _eval_corr_partition(json_file, corr_name, eta, pt, syst):
        if _is_typetracer(eta) or _is_typetracer(pt):
            # return a float32 typetracer array with same form as pt (or eta)
            return ak.zeros_like(pt, dtype=np.float32)

        corrset = get_corrset(json_file)
        corr = corrset[corr_name]

        names = [inp.name for inp in corr.inputs]
        argmap = {
            "eta": eta,
            "abseta": abs(eta),
            "pt": pt,
            "scale_factors": syst,
        }
        args = [argmap[n] for n in names]
        out = corr.evaluate(*args)
        return out.astype(np.float32, copy=False)


    def eval_corr(json_file, corr_name, eta, pt, syst):
        if isinstance(eta, dak.Array) or isinstance(pt, dak.Array):
            meta = ak.to_backend(
                ak.Array(np.empty((0,), dtype=np.float32)),
                "typetracer",
            )
            return dak.map_partitions(
                lambda e, p: _eval_corr_partition(json_file, corr_name, e, p, syst),
                eta,
                pt,
                meta=meta,
            )
        return _eval_corr_partition(json_file, corr_name, eta, pt, syst)

    # -----------------------------
    # choose leading muon for trig
    # -----------------------------
    is_mu1_lead = mu1.pt >= mu2.pt
    lead_eta = ak.where(is_mu1_lead, mu1.eta, mu2.eta)
    lead_pt = ak.where(is_mu1_lead, mu1.pt, mu2.pt)

    # (optional but robust) protect exact-edge issues if pt is extremely close to threshold
    # If your selection guarantees lead_pt > 26, this does nothing in practice.
    eps = 1e-3
    lead_pt = ak.where(lead_pt < 26.0, 26.0 + eps, lead_pt)

    # -----------------------------
    # ID (event = mu1*mu2)
    # -----------------------------
    mu1_id_nom = eval_corr(mu_sf_lookup_file, id_name, mu1.eta, mu1.pt, "nominal")
    mu1_id_up = eval_corr(mu_sf_lookup_file, id_name, mu1.eta, mu1.pt, "systup")
    mu1_id_down = eval_corr(mu_sf_lookup_file, id_name, mu1.eta, mu1.pt, "systdown")

    mu2_id_nom = eval_corr(mu_sf_lookup_file, id_name, mu2.eta, mu2.pt, "nominal")
    mu2_id_up = eval_corr(mu_sf_lookup_file, id_name, mu2.eta, mu2.pt, "systup")
    mu2_id_down = eval_corr(mu_sf_lookup_file, id_name, mu2.eta, mu2.pt, "systdown")

    muID = {
        "nom": mu1_id_nom * mu2_id_nom,
        "up": mu1_id_up * mu2_id_up,
        "down": mu1_id_down * mu2_id_down,
    }

    # -----------------------------
    # ISO (event = mu1*mu2)
    # -----------------------------
    mu1_iso_nom = eval_corr(mu_sf_lookup_file, iso_name, mu1.eta, mu1.pt, "nominal")
    mu1_iso_up = eval_corr(mu_sf_lookup_file, iso_name, mu1.eta, mu1.pt, "systup")
    mu1_iso_down = eval_corr(mu_sf_lookup_file, iso_name, mu1.eta, mu1.pt, "systdown")

    mu2_iso_nom = eval_corr(mu_sf_lookup_file, iso_name, mu2.eta, mu2.pt, "nominal")
    mu2_iso_up = eval_corr(mu_sf_lookup_file, iso_name, mu2.eta, mu2.pt, "systup")
    mu2_iso_down = eval_corr(mu_sf_lookup_file, iso_name, mu2.eta, mu2.pt, "systdown")

    muIso = {
        "nom": mu1_iso_nom * mu2_iso_nom,
        "up": mu1_iso_up * mu2_iso_up,
        "down": mu1_iso_down * mu2_iso_down,
    }

    # -----------------------------
    # TRIGGER (leading muon only)
    # -----------------------------
    mu_trig_nom = eval_corr(mu_sf_lookup_file, trig_name, lead_eta, lead_pt, "nominal")
    mu_trig_up = eval_corr(mu_sf_lookup_file, trig_name, lead_eta, lead_pt, "systup")
    mu_trig_down = eval_corr(mu_sf_lookup_file, trig_name, lead_eta, lead_pt, "systdown")

    muTrig = {"nom": mu_trig_nom, "up": mu_trig_up, "down": mu_trig_down}

    return muID, muIso, muTrig
