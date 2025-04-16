//
// Requered by the script: `main_script_dask_to_run_cppProgram.py`
// Run this script with the command:
// root -l -b -q 'muon_pt_reader.C("root://eos.cms.rcac.purdue.edu:1094//store/user/rasharma/customNanoAOD_Others/UL2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/2A37DC39-1CD3-0446-8F14-92355316FFAB_NanoAOD.root")'
//
// MOTIVATION:
// - We noted that when we print any info for each events (any branch), then for
//   the events for which there is LZMA compression issue, it prints the error message
//   so, my another python script that runs this script looks into the log and mark
//   the files as corrupted if there is "lzma" in the print out from this script.

void muon_pt_reader(const char *filename = "input.root")
{
    TFile *f = TFile::Open(filename);
    if (!f || f->IsZombie())
    {
        std::cerr << "ERRORERROR: Cannot open file: " << filename << std::endl;
        return;
    }

    TTree *tree = (TTree *)f->Get("Events");
    if (!tree)
    {
        std::cerr << "ERRORERROR: TTree 'Events' not found in " << filename << std::endl;
        return;
    }

    const int MAXMU = 3;
    Float_t Muon_pt[MAXMU];
    Int_t nMuon = 0;

    tree->SetBranchAddress("nMuon", &nMuon);
    tree->SetBranchAddress("Muon_pt", Muon_pt);

    Long64_t nentries = tree->GetEntries();
    if (nentries <= 0)
    {
        std::cerr << "ERRORERROR: No entries in the TTree." << std::endl;
        return;
    }
    for (Long64_t i = 0; i < nentries; ++i)
    {
        tree->GetEntry(i);
        if (nMuon < 2) continue;

        std::vector<float> pts;
        for (int j = 0; j < nMuon && j < MAXMU; ++j)
            pts.push_back(Muon_pt[j]);

        std::sort(pts.begin(), pts.end(), std::greater<float>());
        if (i%50000 == 0)
        std::cout << "Event " << i
                  << ": Leading pT = " << pts[0]
                  << ", Subleading pT = " << pts[1] << " GeV" << std::endl;
    }

    f->Close();
}
