#!/usr/bin/env python
import argparse
import awkward as ak
import dask_awkward as dak
from distributed import Client
import ROOT
import os

from plotter.validation_plotter_unified import applyRegionCatCuts

ROOT.gROOT.SetBatch(True)

# python plotter/get2Dplots.py --input /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/compacted/vbf_powheg_dipole/0/ --outputTag VBFSignal
# python plotter/get2Dplots.py --input /depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_08June/stage1_output/2018/compacted/dy_VBF_filter/0/ --outputTag DYVBFFilter

def main(args):
    # 1) start Dask
    client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB') if not args.use_gateway else _make_gateway_client()
    print(f"Dask client: {client}")

    # 2) read the parquet dataset (may be a glob)
    print(f"Reading input from: {args.input}")
    variables = ["mu1_pt", "mu1_ptErr", "mu1_eta", "mu2_pt", "mu2_ptErr", "mu2_eta"]
    events = dak.from_parquet(os.path.join(args.input, "*.parquet"))

    # 2. Apply the region and category cuts
    events = applyRegionCatCuts(
        events,
        category="vbf",
        region_name="signal",
        njets="inclusive",
        process="vbf_powheg_dipole",
        do_vbf_filter_study=False
    )

    # read only the variables we need
    events = events[variables]

    # 3) compute the variables
    mu1_pt = events["mu1_pt"].compute().to_numpy()
    mu1_ptErr = events["mu1_ptErr"].compute().to_numpy()
    mu1_eta = events["mu1_eta"].compute().to_numpy()
    mu2_pt = events["mu2_pt"].compute().to_numpy()
    mu2_ptErr = events["mu2_ptErr"].compute().to_numpy()
    mu2_eta = events["mu2_eta"].compute().to_numpy()

    # 4) make the 2D histogram using ROOT
    hist1_dpTvspT = ROOT.TH2F("hist1_dpTvspT", "#Delta pT vs pT (Leading Muon); pT (GeV); #Delta pT (GeV)",
                     100, 0, 200, 100, 0, 50)
    hist1_dpTvsEta = ROOT.TH2F("hist1_dpTvsEta", "#Delta pT vs Eta (Leading Muon); Eta; #Delta pT (GeV)",
                     100, -2.5, 2.5, 100, 0, 50)
    hist2_dpTvspT = ROOT.TH2F("hist2_dpTvspT", "#Delta pT vs pT (Subleading Muon); pT (GeV); #Delta pT (GeV)",
                     100, 0, 200, 100, 0, 50)
    hist2_dpTvsEta = ROOT.TH2F("hist2_dpTvsEta", "#Delta pT vs Eta (Subleading Muon); Eta; #Delta pT (GeV)",
                     100, -2.5, 2.5, 100, 0, 50)
    for pt, ptErr, eta in zip(mu1_pt, mu1_ptErr, mu1_eta):
        if pt > 0 and ptErr > 0:
            delta_pt = ptErr
            hist1_dpTvspT.Fill(pt, delta_pt)
            hist1_dpTvsEta.Fill(eta, delta_pt)

    for pt, ptErr, eta in zip(mu2_pt, mu2_ptErr, mu2_eta):
        if pt > 0 and ptErr > 0:
            delta_pt = ptErr
            hist2_dpTvspT.Fill(pt, delta_pt)
            hist2_dpTvsEta.Fill(eta, delta_pt)

    # 5) save the histograms to a PDF file
    output_file = f"deltaPT_vs_pTEta_{args.outputTag}"
    with ROOT.TFile(f"{output_file}.root", "RECREATE") as f:
        hist1_dpTvspT.Write()
        hist1_dpTvsEta.Write()
        hist2_dpTvspT.Write()
        hist2_dpTvsEta.Write()

    # 6) Optionally, you can also plot the histograms using ROOT
    c1 = ROOT.TCanvas("c1", "Delta pT vs pT",
                        800, 600)
    hist1_dpTvspT.Draw("COLZ")
    c1.SaveAs(f"{args.outputTag}_deltaPT_vs_pT_leading.pdf")
    c2 = ROOT.TCanvas("c2", "Delta pT vs Eta",
                        800, 600)
    hist1_dpTvsEta.Draw("COLZ")
    c2.SaveAs(f"{args.outputTag}_deltaPT_vs_eta_leading.pdf")

    c3 = ROOT.TCanvas("c3", "Delta pT vs pT (Subleading Muon)",
                        800, 600)
    hist2_dpTvspT.Draw("COLZ")
    c3.SaveAs(f"{args.outputTag}_deltaPT_vs_pT_subleading.pdf")
    c4 = ROOT.TCanvas("c4", "Delta pT vs Eta (Subleading Muon)",
                        800, 600)
    hist2_dpTvsEta.Draw("COLZ")
    c4.SaveAs(f"{args.outputTag}_deltaPT_vs_eta_subleading.pdf")



def _make_gateway_client():
    from dask_gateway import Gateway
    gw = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786"
    )
    cluster = gw.list_clusters()[0]
    return gw.connect(cluster.name).get_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make 2D <ΔpT> profile of muons"
    )
    parser.add_argument(
        "--input", required=True,
        help="Parquet input path (can include wildcards)"
    )
    parser.add_argument(
        "--outputTag", default="Test",
        help="Name of the output PDF"
    )
    parser.add_argument(
        "--use_gateway", action="store_true",
        help="Use Dask Gateway instead of a local Client"
    )
    args = parser.parse_args()
    main(args)
