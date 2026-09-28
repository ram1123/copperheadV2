#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

// NNPDF31_nnlo_as_0118_mc_hessian_pdfas only: 100 Hessian eigenvector members
// (+ central if hasCentral). Total PDF uncertainty on the selected yield
// (rate only, no p_T(mumu) shape): symmetric Hessian envelope, Eq. 6.5.
void pdf_unc_yield(const char* inputName, bool hasCentral) {
    std::unique_ptr<TFile> fin(TFile::Open(inputName));
    TTreeReader reader(fin->Get<TTree>("Events"));
    TTreeReaderValue<Float_t> genWeight(reader, "genWeight");
    TTreeReaderArray<Float_t> pdf(reader, "LHEPdfWeight");
    TTreeReaderArray<Float_t> pt(reader, "Muon_pt"), eta(reader, "Muon_eta");
    TTreeReaderArray<Int_t> charge(reader, "Muon_charge");

    double nominal = 0;
    double up[100] = {0};
    while (reader.Next()) {
        int a = -1, b = -1;
        for (unsigned j = 0; j < pt.GetSize(); ++j) {
            if (std::abs(eta[j]) >= 2.4) continue;
            if (a < 0 || pt[j] > pt[a]) { b = a; a = j; }
            else if (b < 0 || pt[j] > pt[b]) b = j;
        }
        if (b < 0 || pt[a] <= 26 || pt[b] <= 20 || charge[a] * charge[b] >= 0) continue;

        double w = *genWeight;
        nominal += w * (hasCentral ? pdf[0] : 1.0);
        for (unsigned k = 0; k < 100; ++k) up[k] += w * pdf[k + hasCentral];
    }

    double variance = 0;
    for (unsigned k = 0; k < 100; ++k) variance += (up[k] - nominal) * (up[k] - nominal);
    double band = std::sqrt(variance);

    printf("Nominal yield: %.6g\n", nominal);
    printf("Up yield:      %.6g (+%.2f%%)\n", nominal + band, 100 * band / nominal);
    printf("Down yield:    %.6g (-%.2f%%)\n", std::max(0.0, nominal - band), 100 * band / nominal);
}
