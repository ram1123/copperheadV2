import numpy as np
import awkward as ak
from coffea.lookup_tools import txt_converters, rochester_lookup

def apply_roccor(events, roccor_file_path: str, is_mc:bool):
    rochester_data = txt_converters.convert_rochester_file(
        roccor_file_path, loaduncs=True
    )
    roccor_lookup = rochester_lookup.rochester_lookup(rochester_data)
    if is_mc:
        hasgen = ak.is_none(events.Muon.matched_gen.pt, axis=1) == False
        print(f"rochester apply roccor hasgen: {hasgen}")
        np.random.seed(0) # random seed for testing and developing
        print(f"rochester apply roccor ak.count(events.Muon.pt, axis=None): {ak.count(events.Muon.pt, axis=None)}")
        print(f"rochester apply roccor events.Muon.pt: {events.Muon.pt}")
        mc_rand = np.random.rand(ak.count(events.Muon.pt, axis=None))
        print(f"rochester apply roccor mc_rand shape b4 unflatten: {mc_rand.shape}")
        print(f"rochester apply roccor mc_rand b4 unflatten: {mc_rand}")
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        print(f"rochester apply roccor mc_rand after unflatten: {mc_rand}")

    #     corrections = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))
    #     errors = np.array(ak.flatten(ak.ones_like(df.Muon.pt)))
    #     mc_kspread = rochester.kSpreadMC(
    #         df.Muon.charge[hasgen],
    #         df.Muon.pt[hasgen],
    #         df.Muon.eta[hasgen],
    #         df.Muon.phi[hasgen],
    #         df.Muon.matched_gen.pt[hasgen],
    #     )

    #     mc_ksmear = rochester.kSmearMC(
    #         df.Muon.charge[~hasgen],
    #         df.Muon.pt[~hasgen],
    #         df.Muon.eta[~hasgen],
    #         df.Muon.phi[~hasgen],
    #         df.Muon.nTrackerLayers[~hasgen],
    #         mc_rand[~hasgen],
    #     )

    #     errspread = rochester.kSpreadMCerror(
    #         df.Muon.charge[hasgen],
    #         df.Muon.pt[hasgen],
    #         df.Muon.eta[hasgen],
    #         df.Muon.phi[hasgen],
    #         df.Muon.matched_gen.pt[hasgen],
    #     )
    #     errsmear = rochester.kSmearMCerror(
    #         df.Muon.charge[~hasgen],
    #         df.Muon.pt[~hasgen],
    #         df.Muon.eta[~hasgen],
    #         df.Muon.phi[~hasgen],
    #         df.Muon.nTrackerLayers[~hasgen],
    #         mc_rand[~hasgen],
    #     )
    #     hasgen_flat = np.array(ak.flatten(hasgen))
    #     corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
    #     corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
    #     errors[hasgen_flat] = np.array(ak.flatten(errspread))
    #     errors[~hasgen_flat] = np.array(ak.flatten(errsmear))

    #     corrections = ak.unflatten(corrections, ak.num(df.Muon.pt, axis=1))
    #     errors = ak.unflatten(errors, ak.num(df.Muon.pt, axis=1))

    # else:
    #     corrections = rochester.kScaleDT(
    #         df.Muon.charge, df.Muon.pt, df.Muon.eta, df.Muon.phi
    #     )
    #     errors = rochester.kScaleDTerror(
    #         df.Muon.charge, df.Muon.pt, df.Muon.eta, df.Muon.phi
    #     )

    # df["Muon", "pt_roch"] = df.Muon.pt * corrections
    # df["Muon", "pt_roch_up"] = df.Muon.pt_roch + df.Muon.pt * errors
    # df["Muon", "pt_roch_down"] = df.Muon.pt_roch - df.Muon.pt * errors
