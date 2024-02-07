import numpy as np
import uproot

import coffea
from coffea.lookup_tools import dense_lookup


def pu_lookups(parameters, mode="nom", auto=[]):
    lookups = {}
    branch = {"nom": "pileup", "up": "pileup_plus", "down": "pileup_minus"}
    for mode in ["nom", "up", "down"]:
        pu_hist_data = uproot.open(parameters["pu_file_data"])[branch[mode]].values()

        nbins = len(pu_hist_data)
        edges = [[i for i in range(nbins)]]

        if len(auto) == 0:
            pu_hist_mc = uproot.open(parameters["pu_file_mc"])["pu_mc"].values()
        else:
            pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]
        

        lookup = dense_lookup.dense_lookup(pu_reweight(pu_hist_data, pu_hist_mc), edges)
        lookups[mode] = lookup
    return lookups


def pu_reweight(pu_hist_data, pu_hist_mc):
    #print(pu_hist_mc)
    pu_arr_mc_ = np.zeros(len(pu_hist_mc))
    for ibin, value in enumerate(pu_hist_mc):
        pu_arr_mc_[ibin] = max(value, 0)

    pu_arr_data = np.zeros(len(pu_hist_data))
    for ibin, value in enumerate(pu_hist_data):
        pu_arr_data[ibin] = max(value, 0)

    pu_arr_mc_ref = pu_arr_mc_
    pu_arr_mc = pu_arr_mc_ / pu_arr_mc_.sum()
    pu_arr_data = pu_arr_data / pu_arr_data.sum()
    #print(pu_arr_mc)
    weights = np.ones(len(pu_hist_mc))
    weights[pu_arr_mc != 0] = pu_arr_data[pu_arr_mc != 0] / pu_arr_mc[pu_arr_mc != 0]
    maxw = min(weights.max(), 5.0)
    cropped = []
    while maxw > 3:
        cropped = []
        for i in range(len(weights)):
            cropped.append(min(maxw, weights[i]))
        shift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        if abs(shift) > 0.0025:
            break
        maxw *= 0.95

    maxw /= 0.95
    if len(cropped) > 0:
        for i in range(len(weights)):
            cropped[i] = min(maxw, weights[i])
        normshift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        for i in range(len(weights)):
            weights[i] = cropped[i] * (1 - normshift)
    return weights


def checkIntegral(wgt1, wgt2, ref):
    myint = 0
    refint = 0
    for i in range(len(wgt1)):
        myint += wgt1[i] * ref[i]
        refint += wgt2[i] * ref[i]
    return (myint - refint) / refint


def pu_evaluator(parameters, numevents, ntrueint, auto_pu):
    if auto_pu:
        lookups = pu_lookups(parameters, auto=ntrueint)
        #print("Hello")
    pu_weights = {}
    for var, lookup in lookups.items():
        pu_weights[var] = np.ones(numevents)
        pu_weights[var] = lookup(ntrueint)
        pu_weights[var] = np.array(pu_weights[var])
        pu_weights[var][ntrueint > 100] = 1
        pu_weights[var][ntrueint < 1] = 1
    return pu_weights



# NNLOPS SF-------------------------------------------------------------------------

class NNLOPS_Evaluator(object):
    def __init__(self, input_path):
        with uproot.open(input_path) as f:
            self.ratio_0jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_0jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_0jet"],
            }
            self.ratio_1jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_1jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_1jet"],
            }
            self.ratio_2jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_2jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_2jet"],
            }
            self.ratio_3jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_3jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_3jet"],
            }

    def evaluate(self, hig_pt, njets, mode):
        result = np.ones(len(hig_pt), dtype=float)
        hig_pt = np.array(hig_pt)
        njets = np.array(njets)
        result[njets == 0] = np.interp(
            np.minimum(hig_pt[njets == 0], 125.0),
            self.ratio_0jet[mode].member("fX"),
            self.ratio_0jet[mode].member("fY"),
        )
        result[njets == 1] = np.interp(
            np.minimum(hig_pt[njets == 1], 625.0),
            self.ratio_1jet[mode].member("fX"),
            self.ratio_1jet[mode].member("fY"),
        )
        result[njets == 2] = np.interp(
            np.minimum(hig_pt[njets == 2], 800.0),
            self.ratio_2jet[mode].member("fX"),
            self.ratio_2jet[mode].member("fY"),
        )
        result[njets > 2] = np.interp(
            np.minimum(hig_pt[njets > 2], 925.0),
            self.ratio_3jet[mode].member("fX"),
            self.ratio_3jet[mode].member("fY"),
        )
        return result


def nnlops_weights(df, numevents, parameters, dataset):
    nnlops = NNLOPS_Evaluator(parameters["nnlops_file"])
    nnlopsw = np.ones(numevents, dtype=float)
    if "amc" in dataset:
        nnlopsw = nnlops.evaluate(df.HTXS.Higgs_pt, df.HTXS.njets30, "mcatnlo")
    elif "powheg" in dataset:
        nnlopsw = nnlops.evaluate(df.HTXS.Higgs_pt, df.HTXS.njets30, "powheg")
    return nnlopsw


# Mu SF-------------------------------------------------------------------------

def musf_lookup(parameters):
    mu_id_vals = 0
    mu_id_err = 0
    mu_iso_vals = 0
    mu_iso_err = 0
    mu_trig_vals_data = 0
    mu_trig_err_data = 0
    mu_trig_vals_mc = 0
    mu_trig_err_mc = 0

    for scaleFactors in parameters["muSFFileList"]:
        id_file = uproot.open(scaleFactors["id"][0])
        iso_file = uproot.open(scaleFactors["iso"][0])
        print(f'lepton sf scaleFactors["trig"][0: {scaleFactors["trig"][0]}')
        trig_file = uproot.open(scaleFactors["trig"][0])
        mu_id_vals += id_file[scaleFactors["id"][1]].values() * scaleFactors["scale"]
        mu_id_err += (
            id_file[scaleFactors["id"][1]].variances() ** 0.5 * scaleFactors["scale"]
        )
        mu_id_edges = [
            id_file[scaleFactors["id"][1]].axis(0).edges(),
            id_file[scaleFactors["id"][1]].axis(1).edges(),
        ]
        mu_iso_vals += iso_file[scaleFactors["iso"][1]].values() * scaleFactors["scale"]
        mu_iso_err += (
            iso_file[scaleFactors["iso"][1]].variances() ** 0.5 * scaleFactors["scale"]
        )
        mu_iso_edges = [
            iso_file[scaleFactors["iso"][1]].axis(0).edges(),
            iso_file[scaleFactors["iso"][1]].axis(1).edges(),
        ]
        mu_trig_vals_data += (
            trig_file[scaleFactors["trig"][1]].values() * scaleFactors["scale"]
        )
        mu_trig_vals_mc += (
            trig_file[scaleFactors["trig"][2]].values() * scaleFactors["scale"]
        )
        mu_trig_err_data += (
            trig_file[scaleFactors["trig"][1]].variances() ** 0.5
            * scaleFactors["scale"]
        )
        mu_trig_err_mc += (
            trig_file[scaleFactors["trig"][2]].variances() ** 0.5
            * scaleFactors["scale"]
        )
        mu_trig_edges = [
            trig_file[scaleFactors["trig"][1]].axis(0).edges(),
            trig_file[scaleFactors["trig"][1]].axis(1).edges(),
        ]

    mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
    mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
    mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
    mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)

    mu_trig_eff_data = dense_lookup.dense_lookup(mu_trig_vals_data, mu_trig_edges)
    print(f'lepton sf mu_trig_vals_mc: {mu_trig_vals_mc}')
    print(f'lepton sf mu_trig_edges: {mu_trig_edges}')
    mu_trig_eff_mc = dense_lookup.dense_lookup(mu_trig_vals_mc, mu_trig_edges)
    mu_trig_err_data = dense_lookup.dense_lookup(mu_trig_err_data, mu_trig_edges)
    mu_trig_err_mc = dense_lookup.dense_lookup(mu_trig_err_mc, mu_trig_edges)

    return {
        "mu_id_sf": mu_id_sf,
        "mu_id_err": mu_id_err,
        "mu_iso_sf": mu_iso_sf,
        "mu_iso_err": mu_iso_err,
        "mu_trig_eff_data": mu_trig_eff_data,
        "mu_trig_eff_mc": mu_trig_eff_mc,
        "mu_trig_err_data": mu_trig_err_data,
        "mu_trig_err_mc": mu_trig_err_mc,
    }



def musf_evaluator(lookups, year, numevents, muons):
    sf = pd.DataFrame(
        index=mu1.index,
        columns=[
            "muID_nom",
            "muID_up",
            "muID_down",
            "muIso_nom",
            "muIso_up",
            "muIso_down",
            "muTrig_nom",
            "muTrig_up",
            "muTrig_down",
        ],
    )
    sf = sf.fillna(1.0)

    for how in ["nom", "up", "down"]:
        sf[f"trig_num_{how}"] = 1.0
        sf[f"trig_denom_{how}"] = 1.0

    pt = muons.pt_raw
    eta = muons.eta_raw
    abs_eta = abs(muons.eta_raw)

    if "2016" in year:
        muID_ = lookups["mu_id_sf"](eta, pt)
        muIso_ = lookups["mu_iso_sf"](eta, pt)
        muIDerr = lookups["mu_id_err"](eta, pt)
        muIsoerr = lookups["mu_iso_err"](eta, pt)
    else:
        muID_ = lookups["mu_id_sf"](pt, abs_eta)
        muIso_ = lookups["mu_iso_sf"](pt, abs_eta)
        muIDerr = lookups["mu_id_err"](pt, abs_eta)
        muIsoerr = lookups["mu_iso_err"](pt, abs_eta)

    muTrig_data = lookups["mu_trig_eff_data"](abs_eta, pt)
    muTrig_mc = lookups["mu_trig_eff_mc"](abs_eta, pt)
    muTrigerr_data = lookups["mu_trig_err_data"](abs_eta, pt)
    muTrigerr_mc = lookups["mu_trig_err_mc"](abs_eta, pt)

    sf["trig_num_nom"] = ak.prod(1.0 - muTrig_data, axis=1)
    # sf["trig_num_up"] *= 1.0 - ak.to_numpy(muTrig_data - muTrigerr_data)
    # sf["trig_num_down"] *= 1.0 - ak.to_numpy(muTrig_data + muTrigerr_data)
    # sf["trig_denom_nom"] *= 1.0 - ak.to_numpy(muTrig_mc)
    # sf["trig_denom_up"] *= 1.0 - ak.to_numpy(muTrig_mc - muTrigerr_mc)
    # sf["trig_denom_down"] *= 1.0 - ak.to_numpy(muTrig_mc + muTrigerr_mc)

    sf["muID_nom"] *=  ak.prod(muID_, axis=1)
    # sf["muID_up"] *= ak.to_numpy(muID_ + muIDerr)
    # sf["muID_down"] *= ak.to_numpy(muID_ - muIDerr)
    # sf["muIso_nom"] *= ak.to_numpy(muIso_)
    # sf["muIso_up"] *= ak.to_numpy(muIso_ + muIsoerr)
    # sf["muIso_down"] *= ak.to_numpy(muIso_ - muIsoerr)
    print(f'copperheadV2 lepton sf  sf["trig_num_nom"]: \n {sf["trig_num_nom"]}')
    print(f'copperheadV2 lepton sf  sf["muID_nom"]: \n {sf["muID_nom"]}')
    for how in ["nom", "up", "down"]:
        sf[f"trig_num_{how}"] = 1 - sf[f"trig_num_{how}"]
        sf[f"trig_denom_{how}"] = 1 - sf[f"trig_denom_{how}"]
        cut = sf[f"trig_denom_{how}"] != 0
        sf.loc[cut, f"muTrig_{how}"] = (
            sf.loc[cut, f"trig_num_{how}"] / sf.loc[cut, f"trig_denom_{how}"]
        )
    muID = {"nom": sf["muID_nom"], "up": sf["muID_up"], "down": sf["muID_down"]}
    muIso = {"nom": sf["muIso_nom"], "up": sf["muIso_up"], "down": sf["muIso_down"]}
    muTrig = {"nom": sf["muTrig_nom"], "up": sf["muTrig_up"], "down": sf["muTrig_down"]}

    return muID, muIso, muTrig