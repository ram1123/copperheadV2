import unittest

import awkward as ak
import dask_awkward as dak
from unittest.mock import patch

from src.corrections.evaluator import btag_weights_jsonKeepDim
from src.corrections.jet import btag_jet_selection


class BtagJetSelectionTest(unittest.TestCase):
    """Regression tests for the dedicated b-tag jet preselection."""

    def test_run3_btag_selection_uses_dedicated_tight_jet_id(self):
        """Run-3 b-tag selection should accept tight jets even without lep-veto."""
        jets = ak.Array(
            [
                [
                    {
                        "pt": 30.0,
                        "eta": 0.2,
                        "jetId": 2,
                        "neHEF": 0.1,
                        "neEmEF": 0.1,
                        "muEF": 0.0,
                        "chEmEF": 0.0,
                    },
                    {
                        "pt": 35.0,
                        "eta": 0.3,
                        "jetId": 6,
                        "neHEF": 0.1,
                        "neEmEF": 0.1,
                        "muEF": 0.0,
                        "chEmEF": 0.0,
                    },
                    {
                        "pt": 18.0,
                        "eta": 0.4,
                        "jetId": 2,
                        "neHEF": 0.1,
                        "neEmEF": 0.1,
                        "muEF": 0.0,
                        "chEmEF": 0.0,
                    },
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

    def test_btag_systematic_weights_follow_nominal_normalization(self):
        """B-tag systematic up/down weights should use the same normalization as nominal."""

        class DummyWeights:
            def __init__(self, values):
                self._values = dak.from_awkward(ak.Array(values), npartitions=1)

            def weight(self):
                return self._values

        class DummyBTagJson:
            def evaluate(self, variation, hadron_flavour, eta, pt, score):
                if variation == "central":
                    return ak.ones_like(score) * 2.0
                if variation.startswith("up_"):
                    return ak.ones_like(score) * 3.0
                if variation.startswith("down_"):
                    return ak.ones_like(score) * 0.5
                raise AssertionError(f"Unexpected variation: {variation}")

        jets = ak.Array(
            [
                [
                    {
                        "pt": 50.0,
                        "eta": 0.2,
                        "hadronFlavour": 5,
                        "btagDeepB": 0.8,
                    }
                ],
                [
                    {
                        "pt": 60.0,
                        "eta": 0.3,
                        "hadronFlavour": 0,
                        "btagDeepB": 0.4,
                    }
                ],
            ]
        )

        with patch(
            "src.corrections.evaluator.dak.map_partitions",
            side_effect=lambda func, arr, keepdims=True: ak.Array(
                [func(arr.compute(), axis=None, keepdims=keepdims)]
            ),
        ):
            btag_wgt, btag_syst = btag_weights_jsonKeepDim(
                processor=None,
                systs=["jes"],
                jets=jets,
                btag_eta_val=2.5,
                weights=DummyWeights([2.0, 1.0]),
                bjet_sel_mask=None,
                btag_json=DummyBTagJson(),
            )

        self.assertEqual(ak.to_list(ak.flatten(btag_wgt, axis=None)), [1.0, 1.0])
        self.assertEqual(ak.to_list(ak.flatten(btag_syst["jes"]["up"], axis=None)), [1.5, 1.5])
        self.assertEqual(ak.to_list(ak.flatten(btag_syst["jes"]["down"], axis=None)), [0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
