"""
Requered by the script: `main_script_dask.py`
"""
import ROOT
import sys

file = sys.argv[1]

root_file = ROOT.TFile.Open(file, "READ")
if not root_file or root_file.IsZombie() or not root_file.IsOpen():
    raise RuntimeError(f"Unable to open file: {file}")

df = ROOT.RDataFrame("Events", root_file)

filtered_df = df.Filter('nMuon == 2')\
                .Filter('Muon_charge[0] != Muon_charge[1]')\
                .Filter('Muon_isGlobal[0] && Muon_isGlobal[1]')\
                .Filter('Muon_isTracker[0] && Muon_isTracker[1]')\
                .Filter('Muon_pt[0] > 20 && Muon_pt[1] > 20')\
                .Filter('Muon_pfRelIso04_all[0] < 0.15 && Muon_pfRelIso04_all[1] < 0.15')\
                .Define('Dimuon_mass', 'InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)')\
                .Filter('Dimuon_mass > 70')\
                .Define('Electron_veto', f"""
                    Sum((Electron_pt > 10) &&
                        (abs(Electron_eta) < 2.5))
                        """)\
                .Filter('Electron_veto')\

mean_mass = filtered_df.Mean('Dimuon_mass').GetValue()
root_file.Close()

print(f"Successfully processed: {file}, Mean Dimuon Mass: {mean_mass:.2f}")
