"""
All classes must has "Custom" in their names to ensure only those are read by the category
wrapper
"""

import awkward as ak


class CustomInvBtagCut:
    name = "InvBtagCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        # btag_cut = ak.fill_none(events.nBtagLoose_nominal >= 2, value=False) | ak.fill_none(events.nBtagMedium_nominal >= 1, value=False)
        btagLoose_filter = ak.fill_none((events.nBtagLoose_nominal >= 2), value=False)
        btagMedium_filter = ak.fill_none((events.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((events.njets_nominal >= 2), value=False)
        btag_cut = btagLoose_filter | btagMedium_filter
        return ~btag_cut

class CustomVbfCut: # specified in line 827 of AN-19-124
    name = "VbfCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        return vbf_cut

class CustomJet1PtCut: # specified in line 827 of AN-19-124
    name = "Jet1PtCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        jet1_ptCut = ak.fill_none(events.jet1_pt_nominal > 35, value=False) 
        return jet1_ptCut

class CustomInvVbfCut: # specified in line 830 of AN-19-124
    name = "InvVbfCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
        vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        jet1_ptCut = ak.fill_none(events.jet1_pt_nominal > 35, value=False) 
        vbf_cut = vbf_cut & jet1_ptCut
        return ~vbf_cut

class CustomInvJet1PtCut: # specified in line 827 of AN-19-124
    name = "InvJet1PtCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        jet1_ptCut = ak.fill_none(events.jet1_pt_nominal > 35, value=False) 
        return ~jet1_ptCut