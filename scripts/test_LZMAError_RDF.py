import ROOT
import os
import sys
import glob

list_of_files = glob.iglob(
    "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/"
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root"
)

log_file = "TTTo2L2Nu_TuneCP5_13TeV.txt"
for file in list_of_files:
    inFile = file.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu:1094/")
    try:
        df = ROOT.RDataFrame("Events", inFile)

        df = df.Filter('nMuon == 2')\
            .Filter('Muon_charge[0] != Muon_charge[1]')\
            .Define('Dimuon_mass', 'InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)')\
            .Filter('Dimuon_mass > 70')

        print(df.Mean('Dimuon_mass').GetValue())

    except Exception as e:
        print(f"Error processing file: {inFile}")
        with open(error_log_file, "a") as log_file:
            log_file.write(f"{inFile}\n")
    # print("\n\n")

# print the number of files processed
print(f"Number of files processed:finished.")
