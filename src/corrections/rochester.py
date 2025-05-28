import numpy as np
import awkward as ak
from coffea.lookup_tools import txt_converters, rochester_lookup
import pandas as pd
import correctionlib.schemav2 as cs
from src.corrections.MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var
import correctionlib

def apply_roccor(events, roccor_file_path: str, is_mc:bool, test_mode=False):
    rochester_data = txt_converters.convert_rochester_file(
        roccor_file_path, loaduncs=True, 
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    if is_mc:
        hasgen = ak.is_none(events.Muon.matched_gen.pt, axis=1) == False


        # testing start -----------------------------------------------------------
        resrng = cs.Correction(
            name="resrng",
            description="Deterministic smearing value generator",
            version=1,
            inputs=[
                cs.Variable(name="pt", type="real", description="Unsmeared jet pt"),
                cs.Variable(name="eta", type="real", description="Jet pseudorapdity"),
                cs.Variable(name="phi", type="real", description="Jet phi"),
                cs.Variable(name="charge", type="real", description="Muon charge"),
                # cs.Variable(name="event", type="real", description="Event number"),
            ],
            output=cs.Variable(name="rng", type="real"),
            data=cs.HashPRNG(
                nodetype="hashprng",
                inputs=["pt", "eta", "phi", "charge"],
                distribution="stdflat",
            )
        )
        mc_rand = resrng.to_evaluator().evaluate(
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.charge,
            # events.event,
        )
        # print(f"mc_rand: {ak.to_numpy(ak.flatten(mc_rand.compute()))}")
        # mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        
        # testing end --------------------------------------------------------------


        mc_kspread = rochester.kSpreadMC(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.matched_gen.pt,
        )

        mc_ksmear = rochester.kSmearMC(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.nTrackerLayers,
            mc_rand,
        )
        errspread = rochester.kSpreadMCerror(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.matched_gen.pt,
        )
        errsmear = rochester.kSmearMCerror(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.nTrackerLayers,
            mc_rand,
        )
        # hasgen_flat = np.array(ak.flatten(hasgen))
        corrections = ak.ones_like(events.Muon.pt)
        corrections = ak.where(hasgen, mc_kspread, corrections)
        corrections = ak.where((~hasgen), mc_ksmear, corrections)
        # corrections[hasgen] = mc_kspread
        # corrections[~hasgen] = mc_ksmear
        
        errors = ak.ones_like(events.Muon.pt)
        errors = ak.where(hasgen, errspread, errors)
        errors = ak.where((~hasgen), errsmear, errors)
        # errors[hasgen] = np.array(ak.flatten(errspread))
        # errors[~hasgen] = np.array(ak.flatten(errsmear))

        # corrections = ak.unflatten(corrections, ak.num(events.Muon.pt, axis=1))
        # errors = ak.unflatten(errors, ak.num(events.Muon.pt, axis=1))

    else:
        corrections = rochester.kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
        errors = rochester.kScaleDTerror(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
    # print(f"corrections: {ak.to_numpy(ak.flatten(corrections.compute()))}")
    events["Muon", "pt_roch"] = events.Muon.pt * corrections
    # uncommenting these lines below add really significant more compute time. not reccommended unless necessary
    # events["Muon", "pt_roch_up"] = events.Muon.pt_roch + events.Muon.pt * errors
    # events["Muon", "pt_roch_down"] = events.Muon.pt_roch - events.Muon.pt * errors


def apply_roccorRun3(events, roccor_file_path: str, is_mc:bool, test_mode=False):
    cset = correctionlib.CorrectionSet.from_file(roccor_file_path)
    if is_mc: # MC: both scale correction to gen Z peak AND resolution correction to Z width in data
        
        events["Muon", "ptscalecorr"] = pt_scale(
            0, # 1 for data, 0 for mc 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge, 
            cset, 
            nested=True
        )
        
        events["Muon", "ptcorr"] = pt_resol( # TODO: find out why pt_scale isn't used for ptcorr for MC, bc for data pt_scale is used
            events.Muon.ptscalecorr, 
            events.Muon.eta, 
            events.Muon.nTrackerLayers, 
            cset, 
            events, # for more muon variables
            nested=True
        )
        
        # uncertainties
        events["Muon", "ptscalecorr_up"] = pt_scale_var(
            events.Muon.ptcorr, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge,
            "up",
            cset, 
            nested=True
        )
        events["Muon", "ptscalecorr_dn"] = pt_scale_var(
            events.Muon.ptcorr, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge,
            "dn",
            cset, 
            nested=True
        )
        
        events["Muon", "ptcorr_resolup"] = pt_resol_var(
            events.Muon.ptscalecorr, 
            events.Muon.ptcorr, 
            events.Muon.eta, 
            "up",
            cset, 
            nested=True
        )
        events["Muon", "ptcorr_resoldn"] = pt_resol_var(
            events.Muon.ptscalecorr, 
            events.Muon.ptcorr, 
            events.Muon.eta, 
            "dn",
            cset, 
            nested=True
        )
    else: # data
        events["Muon", "ptcorr"] = pt_scale(
            1, # 1 for data, 0 for mc 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge, 
            cset, 
            nested=True # for awkward arrays. Set False for 1d arrays
        )

    # rename the rochester corrected pt to one we use
    events["Muon", "pt_roch"] = events["Muon", "ptcorr"]



    
        