import unittest

import awkward as ak

from src.corrections.jet import btag_jet_selection


class BtagJetSelectionTest(unittest.TestCase):
    """Regression tests for the dedicated b-tag jet preselection."""

    def test_run3_btag_selection_uses_dedicated_tight_jet_id(self):
        """Run-3 b-tag selection should accept tight jets even without lep-veto."""
        jets = ak.Array(
            [
                [
                    {"pt": 30.0, "eta": 0.2, "jetId": 2},
                    {"pt": 35.0, "eta": 0.3, "jetId": 6},
                    {"pt": 18.0, "eta": 0.4, "jetId": 2},
                ]
            ]
        )
        clean = ak.Array([[True, True, True]])
        pass_jet_puid = ak.Array([[False, False, False]])
        config = {
            "jet_id": "tightPassLepVeto",
            "btag_jet_id": "tight",
            "btag_jet_pt_cut": 20.0,
            "btag_jet_eta_cut": 2.5,
            "jec_parameters": {"jet_algorithm": "AK4PFPuppi"},
        }

        mask = btag_jet_selection(jets, clean, pass_jet_puid, config, year="2022preEE")

        self.assertEqual(ak.to_list(mask), [[True, True, False]])

    def test_run2_btag_selection_keeps_chs_puid_requirement(self):
        """Run-2 CHS b-tag selection should still enforce the loose PUID mask."""
        jets = ak.Array(
            [
                [
                    {"pt": 30.0, "eta": 2.3, "jetId": 2},
                    {"pt": 30.0, "eta": 2.45, "jetId": 2},
                ]
            ]
        )
        clean = ak.Array([[True, True]])
        pass_jet_puid = ak.Array([[False, True]])
        config = {
            "jet_id": "tight",
            "btag_jet_id": "tight",
            "btag_jet_pt_cut": 20.0,
            "btag_jet_eta_cut": 2.4,
            "jec_parameters": {"jet_algorithm": "AK4PFchs"},
        }

        mask = btag_jet_selection(jets, clean, pass_jet_puid, config, year="2016preVFP")

        self.assertEqual(ak.to_list(mask), [[False, False]])


if __name__ == "__main__":
    unittest.main()
