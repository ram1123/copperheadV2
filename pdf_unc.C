#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TLorentzVector.h>
#include <TPad.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// For NNPDF31_nnlo_as_0118_mc_hessian_pdfas ONLY.
// Verify the printed branch title against the indexing option before use.
// hasCentral=true:  IDs 325300-325402 (103 weights).
// hasCentral=false: IDs 325301-325402 (102 weights); nominal must use member 0.
// Assumes Muon_pt/eta/phi/mass are Float_t and Muon_charge is Int_t.
void pdf_unc(const char* inputName = "sample.root",
                 const char* outputName = "pdf_mumu_pt.root",
                 bool shapeOnly = false,
                 bool hasCentral = true)
{
    if (std::string(inputName) == outputName)
        throw std::runtime_error("Input and output must be different files");
    // TFile::Open handles both local paths and remote URLs (root://, https://...).
    std::unique_ptr<TFile> inputPtr(TFile::Open(inputName, "READ"));
    if (!inputPtr || inputPtr->IsZombie())
        throw std::runtime_error("Cannot open input");
    TFile& input = *inputPtr;
    auto* tree = input.Get<TTree>("Events");
    if (!tree) throw std::runtime_error("Missing Events tree");
    for (const char* name : {"genWeight", "LHEPdfWeight", "Muon_pt",
                             "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"})
        if (!tree->GetBranch(name))
            throw std::runtime_error(std::string("Missing branch: ") + name);

    std::cout << "PDF metadata: "
              << tree->GetBranch("LHEPdfWeight")->GetTitle() << '\n'
              << "Expected IDs: " << (hasCentral ? "325300" : "325301")
              << "-325402. Check this mapping!\n";

    // Variable-width bins in GeV. Under/overflow are folded into edge bins.
    const double edges[] = {0, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500};
    const int nbins = sizeof(edges) / sizeof(edges[0]) - 1;
    auto makeHist = [&](const std::string& name) {
        auto h = std::make_unique<TH1D>(name.c_str(),
                    ";p_{T}(#mu#mu) [GeV];Weighted events", nbins, edges);
        h->SetDirectory(nullptr);
        h->Sumw2();
        return h;
    };
    auto nominal = makeHist("process");
    std::array<std::unique_ptr<TH1D>, 100> up, down;
    for (unsigned k = 0; k < up.size(); ++k)
        up[k] = makeHist("process_pdf" + std::to_string(k + 1) + "Up");

    TTreeReader reader(tree);
    TTreeReaderValue<Float_t> genWeight(reader, "genWeight");
    TTreeReaderArray<Float_t> pdf(reader, "LHEPdfWeight");
    TTreeReaderArray<Float_t> pt(reader, "Muon_pt"), eta(reader, "Muon_eta");
    TTreeReaderArray<Float_t> phi(reader, "Muon_phi"), mass(reader, "Muon_mass");
    TTreeReaderArray<Int_t> charge(reader, "Muon_charge");
    Long64_t selected = 0;
    while (reader.Next()) {
        if (pdf.GetSize() != (hasCentral ? 103u : 102u))
            throw std::runtime_error("Unexpected PDF count; inspect sample metadata");
        const auto n = pt.GetSize();
        if (eta.GetSize() != n || phi.GetSize() != n || mass.GetSize() != n ||
            charge.GetSize() != n)
            throw std::runtime_error("Inconsistent muon arrays");

        // Illustrative selection, NOT a complete CMS analysis selection:
        // two leading muons within |eta| < 2.4, opposite sign, pT > 26/20 GeV.
        int a = -1, b = -1;
        for (unsigned j = 0; j < n; ++j) {
            if (std::abs(eta[j]) >= 2.4) continue;
            if (a < 0 || pt[j] > pt[a]) { b = a; a = j; }
            else if (b < 0 || pt[j] > pt[b]) b = j;
        }
        if (b < 0 || pt[a] <= 26 || pt[b] <= 20 || charge[a] * charge[b] >= 0)
            continue;
        TLorentzVector mu1, mu2;
        mu1.SetPtEtaPhiM(pt[a], eta[a], phi[a], mass[a]);
        mu2.SetPtEtaPhiM(pt[b], eta[b], phi[b], mass[b]);
        double x = (mu1 + mu2).Pt();
        if (!std::isfinite(x)) throw std::runtime_error("Nonfinite dimuon pT");
        if (x >= edges[nbins]) x = std::nextafter(edges[nbins], edges[0]);
        if (x < edges[0]) x = edges[0];

        // Multiply by luminosity/xsec normalization, pileup and muon SFs here
        // as required by your analysis. Keep signed generator weights.
        const double w = *genWeight;
        if (!std::isfinite(w)) throw std::runtime_error("Nonfinite event weight");
        for (unsigned j = 0; j < pdf.GetSize(); ++j)
            if (!std::isfinite(pdf[j])) throw std::runtime_error("Nonfinite PDF weight");
        nominal->Fill(x, w * (hasCentral ? pdf[0] : 1.0));
        for (unsigned k = 0; k < up.size(); ++k)
            up[k]->Fill(x, w * pdf[k + (hasCentral ? 1 : 0)]);
        ++selected;
    }
    if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd)
        throw std::runtime_error("Tree reading failed; check branch types");
    if (!selected) throw std::runtime_error("No selected events");

    auto band = makeHist("pdf_band");
    unsigned negativeBins = 0;
    for (unsigned k = 0; k < up.size(); ++k) {
        if (shapeOnly) {
            if (nominal->Integral() <= 0 || up[k]->Integral() <= 0)
                throw std::runtime_error("Shape normalization requires positive integrals");
            up[k]->Scale(nominal->Integral() / up[k]->Integral());
        }
        down[k] = makeHist("process_pdf" + std::to_string(k + 1) + "Down");
        for (int bin = 1; bin <= nbins; ++bin) {
            const double central = nominal->GetBinContent(bin);
            const double delta = up[k]->GetBinContent(bin) - central;
            down[k]->SetBinContent(bin, central - delta);
            // Display convention only: mirrored templates share the same MC events.
            // These errors do not model the statistical covariance of the reflection.
            down[k]->SetBinError(bin, nominal->GetBinError(bin));
            if (up[k]->GetBinContent(bin) < 0 || down[k]->GetBinContent(bin) < 0)
                ++negativeBins;
        }
    }
    for (int bin = 1; bin <= nbins; ++bin) {
        double variance = 0;
        for (const auto& h : up) {
            const double delta = h->GetBinContent(bin) - nominal->GetBinContent(bin);
            variance += delta * delta;
        }
        band->SetBinContent(bin, nominal->GetBinContent(bin));
        band->SetBinError(bin, std::sqrt(variance)); // PDF-only band, not MC statistics.
    }

    // Draw the nominal spectrum with the PDF-only uncertainty band and save it
    // as PDF and PNG next to the ROOT file (outputName with .root -> .pdf/.png).
    // Clones are styled for drawing so the histograms written below stay clean.
    // The PDF uncertainty here is only a few percent, so it is invisible against
    // the full spectrum: a lower pad shows the relative band around 1.
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    std::unique_ptr<TH1D> bandDraw(static_cast<TH1D*>(band->Clone("pdf_band_draw")));
    bandDraw->SetDirectory(nullptr);
    bandDraw->SetTitle("PDF uncertainty (NNPDF31 Hessian)"
                       ";;Weighted events");
    bandDraw->SetFillColorAlpha(kOrange + 1, 0.5);
    bandDraw->SetFillStyle(1001);
    bandDraw->SetLineColor(kOrange + 1);
    bandDraw->SetMarkerSize(0);
    bandDraw->SetMinimum(0.0);
    bandDraw->SetMaximum(1.4 * bandDraw->GetMaximum());
    bandDraw->GetYaxis()->SetTitleOffset(1.5);

    nominal->SetLineColor(kBlack);
    nominal->SetLineWidth(2);
    nominal->SetMarkerStyle(20);
    nominal->SetMarkerSize(0.8);

    // Relative band: content 1, error = sigma_pdf / nominal, per bin.
    std::unique_ptr<TH1D> ratio(static_cast<TH1D*>(band->Clone("pdf_band_ratio")));
    ratio->SetDirectory(nullptr);
    double maxRel = 0.0;
    for (int bin = 1; bin <= nbins; ++bin) {
        const double c = nominal->GetBinContent(bin);
        const double rel = c > 0 ? band->GetBinError(bin) / c : 0.0;
        ratio->SetBinContent(bin, 1.0);
        ratio->SetBinError(bin, rel);
        maxRel = std::max(maxRel, rel);
    }
    const double pad = maxRel > 0 ? 1.3 * maxRel : 0.05;
    ratio->SetTitle(";p_{T}(#mu#mu) [GeV];Varied / nominal");
    ratio->SetFillColorAlpha(kOrange + 1, 0.5);
    ratio->SetFillStyle(1001);
    ratio->SetLineColor(kOrange + 1);
    ratio->SetMarkerSize(0);
    ratio->SetMinimum(1.0 - pad);
    ratio->SetMaximum(1.0 + pad);
    ratio->GetYaxis()->SetNdivisions(505);
    ratio->GetYaxis()->SetTitleOffset(1.5);
    ratio->GetYaxis()->SetLabelSize(0.08);
    ratio->GetYaxis()->SetTitleSize(0.09);
    ratio->GetXaxis()->SetLabelSize(0.08);
    ratio->GetXaxis()->SetTitleSize(0.09);

    TCanvas canvas("pdf_band_canvas", "PDF uncertainty", 800, 800);
    TPad pad1("pad1", "", 0.0, 0.30, 1.0, 1.0);
    TPad pad2("pad2", "", 0.0, 0.00, 1.0, 0.30);
    pad1.SetBottomMargin(0.02);
    pad1.SetLeftMargin(0.13);
    pad2.SetTopMargin(0.03);
    pad2.SetBottomMargin(0.32);
    pad2.SetLeftMargin(0.13);
    pad1.Draw();
    pad2.Draw();

    pad1.cd();
    bandDraw->Draw("E2");
    nominal->Draw("HIST SAME");
    nominal->Draw("E0 X0 SAME");
    TLegend legend(0.55, 0.75, 0.88, 0.88);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);
    legend.AddEntry(nominal.get(), "Nominal", "lep");
    legend.AddEntry(bandDraw.get(), Form("PDF uncertainty (max %.1f%%)", 100 * maxRel),
                    "f");
    legend.Draw();
    pad1.RedrawAxis();

    pad2.cd();
    ratio->Draw("E2");
    TLine one(edges[0], 1.0, edges[nbins], 1.0);
    one.SetLineStyle(2);
    one.Draw();
    pad2.RedrawAxis();

    std::string plotBase = outputName;
    const std::string dotRoot = ".root";
    if (plotBase.size() > dotRoot.size() &&
        plotBase.compare(plotBase.size() - dotRoot.size(), dotRoot.size(),
                         dotRoot) == 0)
        plotBase.erase(plotBase.size() - dotRoot.size());
    canvas.SaveAs((plotBase + ".pdf").c_str());
    canvas.SaveAs((plotBase + ".png").c_str());
    std::cout << "Plot: " << plotBase << ".pdf / " << plotBase << ".png\n";

    // CREATE refuses to overwrite an existing file.
    TFile output(outputName, "CREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create output; file may exist");
    nominal->Write();
    band->Write();
    for (unsigned k = 0; k < up.size(); ++k) { up[k]->Write(); down[k]->Write(); }
    output.Close();
    std::cout << "Selected events: " << selected
              << "\nCentral yield: " << nominal->Integral()
              << "\nTreatment: " << (shapeOnly ? "shape only" : "shape and normalization")
              << "\nOutput: " << outputName << '\n';
    if (negativeBins)
        std::cerr << "WARNING: " << negativeBins
                  << " eigenvector/bin pairs contain negative contents. Review binning"
                     " and MC statistics before fitting; contents were not clipped.\n";
}

