import ROOT as rt
import numpy as np
import ROOT
ROOT.gStyle.SetOptStat(0)


# def toNumpy_n_preprcess(muon_etas):
#     muon_etas = np.array(muon_etas)
#     eta_filter = (
#         (muon_etas != -999.0)
#     )
#     muon_etas = muon_etas[eta_filter]
#     return muon_etas

def sort_n_preprcess(lead_muon_etas, sub_lead_muon_etas, lead_muon_pts, sub_lead_muon_pts):
    """
    sort the eta values in the axis=1 such that lead_muon_etas actually contains the eta values from the leading pT muon
    """
    # convert to numpy arrays
    lead_muon_etas = np.array(lead_muon_etas)
    sub_lead_muon_etas = np.array(sub_lead_muon_etas)
    lead_muon_pts = np.array(lead_muon_pts)
    sub_lead_muon_pts = np.array(sub_lead_muon_pts)

    # sort the pt
    pt_filter = lead_muon_pts > sub_lead_muon_pts
    lead_muon_etas = np.where(pt_filter, lead_muon_etas, sub_lead_muon_etas)
    sub_lead_muon_etas = np.where((~pt_filter), lead_muon_etas, sub_lead_muon_etas)

    # check
    lead_muon_pts = np.where(pt_filter, lead_muon_pts, sub_lead_muon_pts)
    sub_lead_muon_pts = np.where((~pt_filter), lead_muon_pts, sub_lead_muon_pts)
    check_val =np.any(lead_muon_pts <sub_lead_muon_pts)
    print(f"sanity check pt sort: {check_val}")
    # raise ValueError
    
    # remove nan values from etas
    eta_filter = (
        (lead_muon_etas != -999.0)
    )
    lead_muon_etas = lead_muon_etas[eta_filter]
    eta_filter = (
        (sub_lead_muon_etas != -999.0)
    )
    sub_lead_muon_etas = sub_lead_muon_etas[eta_filter]
    return lead_muon_etas, sub_lead_muon_etas


def GetGenMuonEtas(tree_files):
    nEntries = 0
    lead_muon_etas = []
    sub_lead_muon_etas = []
    lead_muon_pts = []
    sub_lead_muon_pts = []
    for file in tree_files:
        tree = file.Get("otree")
        nEntries += tree.GetEntries()
        for entry in tree:
            
            lead_muon_eta = entry.gen_leading_photon_Eta
            sub_lead_muon_eta = entry.gen_Subleading_photon_Eta
        
            lead_muon_etas.append(lead_muon_eta)
            sub_lead_muon_etas.append(sub_lead_muon_eta)

            lead_muon_pt = entry.gen_leading_photon_Pt
            sub_lead_muon_pt = entry.gen_Subleading_photon_Pt
            lead_muon_pts.append(lead_muon_pt)
            sub_lead_muon_pts.append(sub_lead_muon_pt)

    print(f"nEntries: {nEntries}")
    print(f"lead_muon_etas: {len(lead_muon_etas)}")
    print(f"sub_lead_muon_etas: {len(sub_lead_muon_etas)}")
    
    # lead_muon_etas = toNumpy_n_preprcess(lead_muon_etas)
    # sub_lead_muon_etas = toNumpy_n_preprcess(sub_lead_muon_etas)
    lead_muon_etas, sub_lead_muon_etas = sort_n_preprcess(lead_muon_etas, sub_lead_muon_etas, lead_muon_pt, sub_lead_muon_pt)
    muon_etas = np.concat([lead_muon_etas,sub_lead_muon_etas], axis=0)

    print(f"lead_muon_etas: {lead_muon_etas.shape}")
    print(f"sub_lead_muon_etas: {sub_lead_muon_etas.shape}")

    return muon_etas, lead_muon_etas, sub_lead_muon_etas

def GetGenMuonEtasInsideMuonRange(tree_files):
    nEntries = 0
    lead_muon_etas = []
    sub_lead_muon_etas = []
    lead_muon_pts = []
    sub_lead_muon_pts = []
    for file in tree_files:
        tree = file.Get("otree")
        nEntries += tree.GetEntries()
        for entry in tree:
            
            lead_muon_eta = entry.gen_leading_photon_Eta
            sub_lead_muon_eta = entry.gen_Subleading_photon_Eta
        
            if (abs(lead_muon_eta) < 2.4) and (abs(sub_lead_muon_eta) < 2.4):
                lead_muon_etas.append(lead_muon_eta)
                sub_lead_muon_etas.append(sub_lead_muon_eta)
                lead_muon_pt = entry.gen_leading_photon_Pt
                sub_lead_muon_pt = entry.gen_Subleading_photon_Pt
                lead_muon_pts.append(lead_muon_pt)
                sub_lead_muon_pts.append(sub_lead_muon_pt)

    # lead_muon_etas = toNumpy_n_preprcess(lead_muon_etas)
    # sub_lead_muon_etas = toNumpy_n_preprcess(sub_lead_muon_etas)
    lead_muon_etas, sub_lead_muon_etas = sort_n_preprcess(lead_muon_etas, sub_lead_muon_etas, lead_muon_pt, sub_lead_muon_pt)
    muon_etas = np.concat([lead_muon_etas,sub_lead_muon_etas], axis=0)

    return muon_etas, lead_muon_etas, sub_lead_muon_etas


def plotEtas(etas, nbins_l, save_fname, xlow, xhigh):
    values = etas
    weights=np.ones_like(values)
    for nbins in nbins_l:
        hist = ROOT.TH1F("hist", f"2018;nbins: {nbins};Entries", nbins, xlow, xhigh)
        hist.FillN(len(values), values, weights)
        canvas = ROOT.TCanvas("canvas", "Canvas for Plotting", 800, 600)
                
        # Draw the histogram
        hist.Draw('E')
        
        # Create a legend
        legend = ROOT.TLegend(0.35, 0.85, 0.65, 0.93)  # (x1,y1,x2,y2) in NDC coordinates
        
        # Add entries
        legend.AddEntry(hist, f"Entries: {hist.GetEntries():.2e}", "l")  # "l" means line
        legend.Draw()
        
        # Save the canvas as a PNG
        save_full_path = f"{save_fname}_nbins{nbins}.pdf"
        canvas.SetTicks(2, 2)
        canvas.Update()
        canvas.SaveAs(save_full_path)


tree_files = [
    # rt.TFile("/depot/cms/users/yun79/test/CMSSW_13_0_14/src/GEN-SIM-analyzer/GenAnalyzer/flat_minAOD.root"),
    # rt.TFile("/depot/cms/users/yun79/test/CMSSW_13_0_14/src/GEN-SIM-analyzer/GenAnalyzer/flat_minAOD_4mil.root"),
    # rt.TFile("/depot/cms/users/yun79/test/CMSSW_13_0_14/src/GEN-SIM-analyzer/GenAnalyzer/flat_minAOD_status1_800k.root"),
    rt.TFile("/depot/cms/users/yun79/test/CMSSW_13_0_14/src/GEN-SIM-analyzer/GenAnalyzer/flat_minAOD_status1_4mil.root"),
    rt.TFile("/depot/cms/users/yun79/test/CMSSW_13_0_14/src/GEN-SIM-analyzer/GenAnalyzer/flat_minAOD_status1_4mil_2.root"),
]

nbins_l = [60, 64, 128, 256]
xlow =-2
xhigh =2
# event_length= "800k"
# event_length= "4mil"
event_length= "8mil"



muon_etas, lead_muon_etas, sub_lead_muon_etas = GetGenMuonEtas(tree_files)
etas = muon_etas
save_fname = f"gen_muon_eta_ROOT_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)

etas = lead_muon_etas
save_fname = f"gen_mu1_eta_ROOT_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)

etas = sub_lead_muon_etas
save_fname = f"gen_mu2_eta_ROOT_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)




#inside muon range
muon_etas, lead_muon_etas, sub_lead_muon_etas = GetGenMuonEtasInsideMuonRange(tree_files)

etas = muon_etas
save_fname = f"gen_muon_eta_ROOT_InsideMuRange_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)

etas = lead_muon_etas
save_fname = f"gen_mu1_eta_ROOT_InsideMuRange_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)

etas = sub_lead_muon_etas
save_fname = f"gen_mu2_eta_ROOT_InsideMuRange_{event_length}"
plotEtas(etas, nbins_l, save_fname, xlow, xhigh)


