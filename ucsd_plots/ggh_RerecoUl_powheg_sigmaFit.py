import dask_awkward as dak
import awkward as ak
from distributed import LocalCluster, Client, progress
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import glob
import pandas as pd
import itertools
import glob
import ROOT as rt
import ROOT

plt.style.use(hep.style.CMS)



"""
Compare ggH MC histograms to see if there's a diff in sigma
"""


def applyVBF_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    print(f"sum btag_cut: {np.sum(btag_cut)}")
    # vbf_cut = ak.fill_none(events.vbf_cut, value=False
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    # region = events.h_peak 
    # region = events.h_sidebands | events.h_peak
    # region = events.h_sidebands 
    dimuon_mass = events.dimuon_mass

    VBF_filter = (
        vbf_cut & 
        # region &
        ~btag_cut # btag cut is for VH and ttH categories
    )
    trues = ak.ones_like(dimuon_mass, dtype="bool")
    falses = ak.zeros_like(dimuon_mass, dtype="bool")
    events["vbf_filter"] = ak.where(VBF_filter, trues,falses)
    print(f"sum (events.jj_mass_nominal > 400): {np.sum((events.jj_mass_nominal > 400))}")
    print(f"sum (events.jj_dEta_nominal > 2.5): {np.sum((events.jj_dEta_nominal > 2.5))}")
    print(f"sum  (events.jet1_pt_nominal > 35) : {np.sum( (events.jet1_pt_nominal > 35) )}")
    print(f"sum vbf_cut: {np.sum(vbf_cut)}")
    print(f"sum VBF_filter: {np.sum(VBF_filter)}")
    print(f"events.jj_mass_nominal: {events.jj_mass_nominal}")
    return events[VBF_filter]
    # return events

def applyGGH_cutV1(events):
    btag_cut =ak.fill_none((events.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((events.nBtagMedium_nominal >= 1), value=False)
    vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35) 
    vbf_cut = ak.fill_none(vbf_cut, value=False)
    dimuon_mass = events.dimuon_mass
    # region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    # region = events.region == "h-peak"
    # region = events.region == "h-sidebands"
    ggH_filter = (
        ~vbf_cut & 
        ~btag_cut # btag cut is for VH and ttH categories
    )
    return events[ggH_filter]

def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    events = events[region]
    return events

def generateRooHist(x, events, name=""):
    dimuon_mass = ak.to_numpy(events.dimuon_mass)
    wgts = ak.to_numpy(events.wgt_nominal)
    nbins = x.getBins()
    TH = rt.TH1D("TH", "TH", nbins, x.getMin(), x.getMax())
    TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # fill the histograms with mass and weights 
    roohist = rt.RooDataHist(name, name, rt.RooArgSet(x), TH)
    return roohist

def generateRooHist(x, dimuon_mass, wgts, name=""):
    print("generateRooHist version 2")
    dimuon_mass = np.asarray(ak.to_numpy(dimuon_mass)).astype(np.float64) # explicit float64 format is required
    wgts = np.asarray(ak.to_numpy(wgts)).astype(np.float64) # explicit float64 format is required
    nbins = x.getBins()
    TH = rt.TH1D("TH", "TH", nbins, x.getMin(), x.getMax())
    TH.FillN(len(dimuon_mass), dimuon_mass, wgts) # fill the histograms with mass and weights 
    roohist = rt.RooDataHist(name, name, rt.RooArgSet(x), TH)
    return roohist

def normalizeRooHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Scale(1/THist.Integral())
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized



def categorize_eta(eta):
    """Classify eta values into Barrel, Overlap, or Endcap regions."""
    abs_eta = abs(eta)
    if abs_eta < 0.9:
        return "B" # for Barrel
    elif 0.9 <= abs_eta < 1.8:
        return "O" # for Overlap
    else:
        return "E" # for Endcap

def classify_events(df):
    """Categorize events based on mu1_eta and mu2_eta."""
    df['mu1_region'] = df['mu1_eta'].apply(categorize_eta)
    df['mu2_region'] = df['mu2_eta'].apply(categorize_eta)
    df['EtaCat'] = df['mu1_region'] + df['mu2_region']
    return df


def addEtaCategories(events):
    """
    take mu1_eta and mu2_eta and add a field called "EtaCat"
    """
    # first convert necessary info to pd df bc that's what chatgpt is using
    df = pd.DataFrame({
        "mu1_eta" : ak.to_numpy(events.mu1_eta),
        "mu2_eta" : ak.to_numpy(events.mu2_eta),
    })
    df = classify_events(df)
    events["EtaCat"] = df.EtaCat
    return events


def generate_eta_categories():
    """Generate all possible combinations of mu1_eta and mu2_eta categories."""
    eta_categories = ["B", "O", "E"]
    combinations = list(itertools.product(eta_categories, repeat=2))
    # category_list = [f"{mu1}_{mu2}" for mu1, mu2 in combinations]
    category_list = [f"{mu1}{mu2}" for mu1, mu2 in combinations]
    return category_list

def filterEtaCat(events, eta_cat):
    cat_filter = events.EtaCat == eta_cat
    dimuon_mass = events.dimuon_mass[cat_filter]
    wgt = events.wgt_nominal[cat_filter]
    return dimuon_mass, wgt

def fitPlot_ggh(dimuon_mass, wgt, label, save_filename):
    """
    generate histogram from dimuon mass and wgt, fit DCB
    aftwards, plot the histogram and return the fit params
    as fit DCB sigma and chi2_dof
    """
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 115, 135) # signal region
    # nbins = 100
    nbins = 80
    mass.setBins(nbins)
    hist = generateRooHist(mass, dimuon_mass, wgt, name=f"{label} hist")
    print(f"fitPlot_ggh hist: {hist}")

    # --------------------------------------------------
    # Fitting
    # --------------------------------------------------
    
    MH = rt.RooRealVar("MH" , "MH", 125, 110, 150)
    # MH.setConstant(True)
    sigma = rt.RooRealVar("sigma" , "sigma", 1.8228, .1, 4.0)
    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 1.12842, 0.01, 65)
    n1 = rt.RooRealVar("n1" , "n1", 4.019960, 0.01, 100)
    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 1.3132, 0.01, 65)
    n2 = rt.RooRealVar("n2" , "n2", 9.97411, 0.01, 100)
    name = f"DCB fit"
    model = rt.RooDoubleCBFast(name,name,mass, MH, sigma, alpha1, n1, alpha2, n2)

    device = "cpu"
    _ = model.fitTo(hist,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result = model.fitTo(hist,  EvalBackend=device, Save=True, SumW2Error=True)
    fit_result.Print()

    # --------------------------------------------------
    # plotting
    # --------------------------------------------------
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    
    frame = mass.frame()
    frame.SetTitle(f"{label}")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    hist.plotOn(frame, Name=hist.GetName())
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "MC", "L")
    model.plotOn(frame, Name=model.GetName(), LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "DCB fit", "L")

    # add chi2 dof
    ndf = model.getParameters(ROOT.RooArgSet(mass)).getSize()
    print(f"ndf: {ndf}")
    chi2_ndf = frame.chiSquare(model.GetName(), hist.GetName(), ndf)
    chi2_text = " chi2/ndf = {:.3f}".format(chi2_ndf)
    legend.AddEntry("", chi2_text, "")

    # add sigma
    sigma_val = sigma.getVal()
    sigma_text = " fit sigma = {:.3f}".format(sigma_val)
    legend.AddEntry("", sigma_text, "")
    
    frame.Draw()
    legend.Draw()        
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(save_filename)
    
    return sigma_val, chi2_ndf
    
# def plotMuonEtas(amc_events, powheg_events):
#     """
#     plot side by side muon eta variables to see if there's a discrepancy
#     between two different mc modeling samples
#     """
#     # Apply a CMS or ATLAS style
#     hep.style.use("CMS")  # You can also use "ATLAS"
    
#     # Example muon eta values for two events (Replace these with actual data)
#     muon_eta_event1 = np.random.uniform(-2.4, 2.4, size=10)  
#     muon_eta_event2 = np.random.uniform(-2.4, 2.4, size=10)  
    
#     # Define histogram parameters
#     bins = 30
#     range_eta = (-2.4, 2.4)

#     # normalize wgts
#     wgts_amc = amc_events.wgt_nominal/np.sum(amc_events.wgt_nominal)
#     wgts_powheg = powheg_events.wgt_nominal/np.sum(powheg_events.wgt_nominal)
    
#     # Compute histograms
#     hist_amc, bin_edges = np.histogram(amc_events.mu1_eta, bins=bins, range=range_eta, weights=wgts_amc)
#     hist_powheg, _ = np.histogram(powheg_events.mu1_eta, bins=bins, range=range_eta, weights=wgts_powheg)
    
#     # Compute bin centers
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Create figure
#     plt.figure(figsize=(8, 6))
    
#     # Plot histograms as step plots
#     plt.step(bin_centers, hist_amc, where="mid", linewidth=2, label="AMC")
#     plt.step(bin_centers, hist_powheg, where="mid", linewidth=2, label="Powheg")
    
#     # Labels, legend, and title
#     plt.xlabel(r"Muon $\eta$")
#     plt.ylabel("Entries")
#     plt.title("Muon $\eta$ Distribution for Two Events")
#     plt.legend()
#     plt.savefig("test.pdf")

def plotMuonEtas(amc_events, powheg_events):
    """
    plot side by side muon eta variables to see if there's a discrepancy
    between two different mc modeling samples
    """
    eta_name = "mu_eta"
    eta = rt.RooRealVar(eta_name, eta_name, 0, -2.5, 2.5) # signal region
    # nbins = 100
    nbins = 30
    eta.setBins(nbins)
    # enforce float values
    # mu_eta = ak.values_astype(amc_events.mu1_eta, np.float64)
    # wgts = ak.values_astype(amc_events.wgt_nominal, np.float64)
    mu_eta = amc_events.mu1_eta
    wgts = amc_events.wgt_nominal
    print(f"is any mu_eta : {ak.any(ak.is_none(mu_eta))}")
    print(f"is any wgts : {ak.any(ak.is_none(wgts))}")
    hist_amc = generateRooHist(eta, mu_eta, wgts, name=f"amc mu1 eta")
    hist_amc = normalizeRooHist(eta, hist_amc)
    
    hist_powheg = generateRooHist(eta, powheg_events.mu1_eta, powheg_events.wgt_nominal, name=f"powheg mu1 eta")
    hist_powheg = normalizeRooHist(eta, hist_powheg)

    # ------------------------------------
    # Plotting
    # ------------------------------------
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    
    frame = eta.frame()
    frame.SetTitle(f"{label}")
    frame.SetXTitle(f"Dimuon Mass (GeV)")
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    
    hist_amc.plotOn(frame, Name=hist_amc.GetName(), LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "AMC", "L")
    hist_powheg.plotOn(frame, Name=hist_powheg.GetName(), LineColor=rt.kBlue)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1), "Powheg", "L")

    frame.Draw()
    legend.Draw()        
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("test.pdf")


V1_fields_2compute = [
    "wgt_nominal",
    "dimuon_mass",
    "mu1_eta",
    "mu2_eta"
]



if __name__ == "__main__":
    client =  Client(n_workers=15,  threads_per_worker=2, processes=True, memory_limit='8 GiB') 

   
    
    # rerecoPowheg_years = ["2017", "2018"]
    years = ["2017", "2018"]

    rerecoPowheg_events = []
    for year in years:
        rerecoPowheg_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//rereco_yun_Dec05_btagSystFixed_JesJerUncOn//stage1_output/{year}/"
        file = f"{rerecoPowheg_load_path}/ggh_powhegPS"
        print(f"file: {file}")
    
        
        rerecoPowheg_events_data = dak.from_parquet(f"{file}/*.parquet")
        rerecoPowheg_events_data = filterRegion(rerecoPowheg_events_data, region="signal")
        rerecoPowheg_events_data = applyGGH_cutV1(rerecoPowheg_events_data)
        rerecoPowheg_events_data = ak.zip({field: rerecoPowheg_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(rerecoPowheg_events_data))
        # print((rerecoPowheg_events_data))
        rerecoPowheg_events.append(rerecoPowheg_events_data)
        # print(rerecoPowheg_events_data)
        # raise ValueError

    
    rerecoPowheg_events = ak.concatenate(rerecoPowheg_events, axis=0)
    # print(f"rerecoPowheg_events len: {len(rerecoPowheg_events)}")
    # print(f"rerecoPowheg_events sum wgts: {ak.sum(rerecoPowheg_events.wgt_nominal)}")
    

    
    # add rereco amc 
    rerecoAmc_events = []
    for year in years:
        rerecoAmc_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//rereco_yun_Dec05_btagSystFixed_JesJerUncOn//stage1_output/{year}/"
        file = f"{rerecoAmc_load_path}/ggh_amcPS"
        print(f"file: {file}")
    
        
        rerecoAmc_events_data = dak.from_parquet(f"{file}/*.parquet")
        
        rerecoAmc_events_data = filterRegion(rerecoAmc_events_data, region="signal")
        rerecoAmc_events_data = applyGGH_cutV1(rerecoAmc_events_data)
        rerecoAmc_events_data = ak.zip({field: rerecoAmc_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(rerecoAmc_events_data))
        # print((rerecoAmc_events_data))
        rerecoAmc_events.append(rerecoAmc_events_data)
        # print(rerecoAmc_events_data)
        # raise ValueError
        
    rerecoAmc_events = ak.concatenate(rerecoAmc_events, axis=0)
    # print(f"rerecoAmc_events len: {len(rerecoAmc_events)}")
    # print(f"rerecoAmc_events sum wgts: {ak.sum(rerecoAmc_events.wgt_nominal)}")
    # raise ValueError
    
    ul_events = []
    # ul_years = ["2017", "2018"]
    for year in years:
        ul_load_path =f"/depot/cms/users/yun79/hmm/copperheadV1clean//V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix/stage1_output/{year}/f1_0"
        file = f"{ul_load_path}/ggh_powhegPS"
        print(f"file: {file}")
        
        ul_events_data = dak.from_parquet(f"{file}/*/*.parquet")
        
        ul_events_data = filterRegion(ul_events_data, region="signal")
        ul_events_data = applyGGH_cutV1(ul_events_data)
        ul_events_data = ak.zip({field: ul_events_data[field] for field in V1_fields_2compute}).compute()
        print(len(ul_events_data))
        ul_events.append(ul_events_data)

        
    ul_events = ak.concatenate(ul_events, axis=0)
    print(f"ul_events len: {len(ul_events)}")
    print(f"ul_events sum wgts: {ak.sum(ul_events.wgt_nominal)}")
    

    rerecoPowheg_events = addEtaCategories(rerecoPowheg_events)
    ul_events = addEtaCategories(ul_events)
    rerecoAmc_events = addEtaCategories(rerecoAmc_events)


    out_table = pd.DataFrame()
    possible_eta_categories = generate_eta_categories()

    # for eta_cat in possible_eta_categories:
    #     rerecoPowheg_dimuon_mass, rerecoPowheg_wgt = filterEtaCat(rerecoPowheg_events, eta_cat)
    #     rerecoAmc_dimuon_mass, rerecoAmc_wgt = filterEtaCat(rerecoAmc_events, eta_cat)
    #     ul_dimuon_mass, ul_wgt = filterEtaCat(ul_events, eta_cat)

    #     # now plot
    #     label = f"rerecoPowheg_etacat{eta_cat}"
    #     save_filename = f"plots/gghMC_{label}.pdf"
    #     rerecoPowheg_sigma, rerecoPowheg_chi2_dof = fitPlot_ggh(rerecoPowheg_dimuon_mass, rerecoPowheg_wgt, label, save_filename)

    #     label = f"ulPowheg_etacat{eta_cat}"
    #     save_filename = f"plots/gghMC_{label}.pdf"
    #     ul_sigma, ul_chi2_dof = fitPlot_ggh(ul_dimuon_mass, ul_wgt, label, save_filename)

    #     label = f"rerecoAmc_etacat{eta_cat}"
    #     save_filename = f"plots/gghMC_{label}.pdf"
    #     rerecoAmc_sigma, rerecoAmc_chi2_dof = fitPlot_ggh(rerecoAmc_dimuon_mass, rerecoAmc_wgt, label, save_filename)

        
    #     out_dict = {
    #         "Eta Category" : [eta_cat],
    #         "Rereco Powheg Sigma" : [rerecoPowheg_sigma],
    #         "Rereco AMC Sigma" : [rerecoAmc_sigma],
    #         "UL Sigma" : [ul_sigma],
    #         "Rereco Powheg Chi2 Dof" : [rerecoPowheg_chi2_dof],
    #         "Rereco AMC Chi2 Dof" : [rerecoAmc_chi2_dof],
    #         "UL Chi2 Dof" : [ul_chi2_dof],
    #     }
    #     # add the computed values
    #     out_table = pd.concat([out_table, pd.DataFrame(out_dict)], ignore_index=True)
    
    # out_table.to_csv("RerecoUl_etaCat_table.csv")



    # 
    plotMuonEtas(rerecoAmc_events, rerecoPowheg_events)






