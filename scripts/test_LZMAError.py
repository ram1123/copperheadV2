# use uproot to open root file and hepconvert.root_to_parquet to convert to parquet

import uproot
import hepconvert
import tempfile

# inFile = "root://eos.cms.rcac.purdue.edu:1094//store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/A69F9BE8-588A-4141-9FEF-CE6F4ABDA839_NanoAOD.root"

# file = uproot.open(inFile)
# tree = file["Events"]
# print(tree.keys())

# hepconvert.root_to_parquet(in_file=inFile, out_file="test.parquet", tree="Events")


inFile = "root://eos.cms.rcac.purdue.edu:1094//store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/A69F9BE8-588A-4141-9FEF-CE6F4ABDA839_NanoAOD.root"
# inFile = "root://eos.cms.rcac.purdue.edu:1094//store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9/91FEA034-FF6B-9D4A-8991-0426884F3E27_NanoAOD.root"
inFile = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9/91FEA034-FF6B-9D4A-8991-0426884F3E27_NanoAOD.root"
outFile = "test.parquet"

# hepconvert.root_to_parquet(
#     in_file=inFile,
#     out_file=outFile,
#     tree="Events",
#     keep_branches=["dimuon_pt"],
#     force=True,
# )
with tempfile.NamedTemporaryFile(suffix=".parquet", dir="/tmp/") as tmp:
    print(tmp.name)
    hepconvert.root_to_parquet(
        in_file=str(inFile),
        out_file=tmp.name,
        tree="Events",
        keep_branches=["dimuon_pt"],
        force=True,
    )
