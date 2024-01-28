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
        print(f"ak.sum(~hasgen) : {ak.sum(~hasgen)}")
        print(f"rochester apply roccor hasgen: {hasgen}")
        np.random.seed(0) # random seed for testing and developing
        # print(f"rochester apply roccor ak.count(events.Muon.pt, axis=None): {ak.count(events.Muon.pt, axis=None)}")
        print(f"rochester apply roccor events.Muon.pt: {events.Muon.pt}")
        mc_rand_size = ak.count(events.Muon.pt[~hasgen], axis=None)
        print(f"mc_rand_size : {mc_rand_size}")
        mc_rand = np.random.rand(mc_rand_size) # ak doesn't have rand function
        print(f"rochester apply roccor mc_rand shape b4 unflatten: {mc_rand.shape}")
        print(f"rochester apply roccor mc_rand b4 unflatten: {mc_rand}")
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt[~hasgen], axis=1))
        print(f"rochester apply roccor mc_rand after unflatten: {mc_rand}")
        
        # corrections = ak.ones_like(events.Muon.pt)
        # errors = ak.ones_like(events.Muon.pt)
        mu_count = ak.count(events.Muon.pt, axis=None)
        corrections = np.ones(mu_count)
        errors = np.ones(mu_count)

        """
        Observed slight discrepencies (order of 0.0001) between results from coffea0.7 and coffea2024
        Thought this could be due to less precise data type, so I increased the datatype to float64, 
        but with same results. I am now suspecting different with akward array verions. Coffeaa0.7
        uses awkward=1.10.3, and coffea2014 uses awkward=2.5.2. I tried to either upgrade coffea0.7 or
        downgrade coffea2024 to match the awkward array versions to test my theory, but awkward version
        dependencies between the two coffeas are non-intersecting. So best to just tough it out

        This slight diescrepency is an issue when values are low to orders of 0.001. This applies to
        errspread and errsmear. But since the way we apply these errors is to add them on top of the 
        nominal value, this diescrepency could be negligible.
        """
        # events["Muon","pt"] = ak.values_astype(events.Muon.pt, "float64")
        # events["Muon","eta"] = ak.values_astype(events.Muon.eta, "float64")
        # events["Muon","phi"] = ak.values_astype(events.Muon.phi, "float64")
        print(f"rochester apply roccor events.Muon.pt.type: {events.Muon.pt.type.content}")
        print(f"rochester apply roccor df.Muon.pt.type: {mc_rand.type.content}")
        
        mc_kspread = roccor_lookup.kSpreadMC(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        # print(f"rochester apply roccor mc_kspread: {mc_kspread}")
        print(f"rochester apply roccor mc_kspread: {ak.to_numpy(ak.flatten(mc_kspread))}")
        mc_ksmear = roccor_lookup.kSmearMC(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand,
        )
        # print(f"rochester apply roccor mc_ksmear: {mc_ksmear}")
        print(f"rochester apply roccor mc_ksmear: {ak.to_numpy(ak.flatten(mc_ksmear))}")
        errspread = roccor_lookup.kSpreadMCerror(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        # print(f"rochester apply roccor errspread: {errspread}")
        print(f"rochester apply roccor errspread: {ak.to_numpy(ak.flatten(errspread))}")
        errsmear = roccor_lookup.kSmearMCerror(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand,
        )
        # print(f"rochester apply roccor errsmear: {errsmear}")
        print(f"rochester apply roccor errsmear: {ak.to_numpy(ak.flatten(errsmear))}")
        # corrections[hasgen] = mc_kspread
        # corrections[~hasgen] = mc_ksmear
        # errors[hasgen] = errspread
        # errors[~hasgen] = errsmear
        
        hasgen_flat = np.array(ak.flatten(hasgen))
        """
        in-place assignments are not supported by awkward afaik, so this long winded
        way to defining np arrays and later turning into awkward array is used
        """
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))

        # corrections and errors back to awkard 
        mu_num = ak.num(events.Muon.pt, axis=1)
        corrections = ak.unflatten(corrections, mu_num)
        errors = ak.unflatten(errors, mu_num)


    else: # if data
        corrections = rochester.kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
        errors = rochester.kScaleDTerror(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )

    events["Muon", "pt_roch"] = events.Muon.pt * corrections
    events["Muon", "pt_roch_up"] = events.Muon.pt_roch + events.Muon.pt * errors
    events["Muon", "pt_roch_down"] = events.Muon.pt_roch - events.Muon.pt * errors
    print(f'rochester apply roccor events["Muon", "pt_roch"]: {ak.to_numpy(ak.flatten(events["Muon", "pt_roch"]))}')
    print(f'rochester apply roccor events["Muon", "pt_roch_up"]: {ak.to_numpy(ak.flatten(events["Muon", "pt_roch_up"]))}')
    print(f'rochester apply roccor events["Muon", "pt_roch_down"]: {ak.to_numpy(ak.flatten(events["Muon", "pt_roch_down"]))}')
