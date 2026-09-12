#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TLorentzVector.h>
#include <TPad.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>

// NNPDF31_nnlo_as_0118_mc_hessian_pdfas only: LHEPdfWeight[0] is the central
// member, [1..100] the 100 Hessian eigenvector members (101-102 are alphaS,
// excluded here). PDF-only SHAPE uncertainty on p_T(mumu) -- each variation
// is rescaled to the nominal yield, so normalization/rate is removed by
// construction; use pdf_unc_yield.C for the total-yield number.
void pdf_unc(const char* inputName, const char* outputName) {
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    std::unique_ptr<TFile> fin(TFile::Open(inputName));
    TTreeReader reader(fin->Get<TTree>("Events"));
    TTreeReaderValue<Float_t> genWeight(reader, "genWeight");
    TTreeReaderArray<Float_t> pdf(reader, "LHEPdfWeight");
    TTreeReaderArray<Float_t> pt(reader, "Muon_pt"), eta(reader, "Muon_eta");
    TTreeReaderArray<Float_t> phi(reader, "Muon_phi"), mass(reader, "Muon_mass");
    TTreeReaderArray<Int_t> charge(reader, "Muon_charge");

    const double edges[] = {0, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500};
    const int nbins = sizeof(edges) / sizeof(edges[0]) - 1;
    auto hist = [&](const std::string& n) {
        auto h = std::make_unique<TH1D>(n.c_str(),
                    ";p_{T}(#mu#mu) [GeV];Weighted events", nbins, edges);
        h->SetDirectory(nullptr);
        h->Sumw2();
        return h;
    };
    auto nominal = hist("process");
    std::array<std::unique_ptr<TH1D>, 100> up, down;
    for (unsigned k = 0; k < up.size(); ++k) up[k] = hist("up" + std::to_string(k));

    while (reader.Next()) {
        int a = -1, b = -1;
        for (unsigned j = 0; j < pt.GetSize(); ++j) {
            if (std::abs(eta[j]) >= 2.4) continue;
            if (a < 0 || pt[j] > pt[a]) { b = a; a = j; }
            else if (b < 0 || pt[j] > pt[b]) b = j;
        }
        if (b < 0 || pt[a] <= 26 || pt[b] <= 20 || charge[a] * charge[b] >= 0) continue;
        TLorentzVector mu1, mu2;
        mu1.SetPtEtaPhiM(pt[a], eta[a], phi[a], mass[a]);
        mu2.SetPtEtaPhiM(pt[b], eta[b], phi[b], mass[b]);
        double x = std::min((mu1 + mu2).Pt(), edges[nbins] - 1e-6);
        double w = *genWeight;
        nominal->Fill(x, w * pdf[0]);
        for (unsigned k = 0; k < up.size(); ++k) up[k]->Fill(x, w * pdf[k + 1]);
    }

    auto band = hist("pdf_band");
    for (unsigned k = 0; k < up.size(); ++k) {
        up[k]->Scale(nominal->Integral() / up[k]->Integral());  // shape only
        down[k] = hist("down" + std::to_string(k));
        for (int bin = 1; bin <= nbins; ++bin) {
            double c = nominal->GetBinContent(bin), d = up[k]->GetBinContent(bin) - c;
            down[k]->SetBinContent(bin, c - d);
            band->SetBinContent(bin, c);
            band->SetBinError(bin, std::hypot(band->GetBinError(bin), d));
        }
    }

    // Top pad: nominal + band (styled in place; band's content/errors, and
    // what gets Write()'n below, are unaffected by draw-style setters).
    // Bottom pad: relative band around 1 (band is a few % of content,
    // invisible against the full spectrum on the top pad's linear scale).
    band->SetFillColorAlpha(kOrange + 1, 0.5);
    band->SetLineColor(kOrange + 1);
    band->SetMarkerSize(0);
    band->SetMinimum(0);
    band->SetMaximum(1.4 * band->GetMaximum());
    nominal->SetLineColor(kBlack);
    nominal->SetLineWidth(2);
    nominal->SetMarkerStyle(20);

    std::unique_ptr<TH1D> ratio(static_cast<TH1D*>(band->Clone()));
    ratio->SetDirectory(nullptr);
    ratio->SetTitle(";p_{T}(#mu#mu) [GeV];Varied / nominal");
    double maxRel = 0;
    for (int bin = 1; bin <= nbins; ++bin) {
        double c = nominal->GetBinContent(bin);
        double rel = c > 0 ? band->GetBinError(bin) / c : 0;
        ratio->SetBinContent(bin, 1);
        ratio->SetBinError(bin, rel);
        maxRel = std::max(maxRel, rel);
    }
    ratio->SetMinimum(1 - 1.3 * maxRel);
    ratio->SetMaximum(1 + 1.3 * maxRel);
    ratio->GetYaxis()->SetNdivisions(505);

    TCanvas canvas("pdf_band_canvas", "", 800, 800);
    TPad p1("p1", "", 0, 0.30, 1, 1), p2("p2", "", 0, 0, 1, 0.30);
    p1.SetBottomMargin(0.02);
    p2.SetTopMargin(0.03);
    p2.SetBottomMargin(0.32);
    p1.Draw();
    p2.Draw();

    p1.cd();
    band->Draw("E2");
    nominal->Draw("HIST SAME");
    nominal->Draw("E0 X0 SAME");
    TLegend legend(0.55, 0.75, 0.88, 0.88);
    legend.SetBorderSize(0);
    legend.AddEntry(nominal.get(), "Nominal", "lep");
    legend.AddEntry(band.get(), Form("PDF unc. (max %.1f%%)", 100 * maxRel), "f");
    legend.Draw();

    p2.cd();
    ratio->Draw("E2");
    TLine line(edges[0], 1, edges[nbins], 1);
    line.SetLineStyle(2);
    line.Draw();

    std::string base = outputName;
    base.erase(base.rfind(".root"));
    canvas.SaveAs((base + ".pdf").c_str());
    canvas.SaveAs((base + ".png").c_str());

    TFile fout(outputName, "RECREATE");
    nominal->Write();
    band->Write();
    for (unsigned k = 0; k < up.size(); ++k) { up[k]->Write(); down[k]->Write(); }
}
