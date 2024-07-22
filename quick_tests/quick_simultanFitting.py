import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt


from quickSMFtest_functions import MakeBWZ_Redux

if __name__ == "__main__":
    """
    loading stage1 output data and evalauting BDT is deletegated to run_stage2.py
    """
    load_path = "/work/users/yun79/stage2_output/test/processed_events_data.parquet"
    processed_eventsData = ak.from_parquet(load_path)

    
    print("events loaded!")
    
    # comence roofit fitting for each subcategory 
    n_subCats = 5
    poly_order_by_cat = {
        0:3,
        1:3,
        2:2,
        3:2,
        4:2,
    }
    subCatColorMap = {
        "subCat0":rt.kBlue,
        "subCat1":rt.kGreen,
    }
    dof = 3 # degrees of freedom for the core-functions. This should be same for all the functions
    smf_coeffStartVals = {
        0: {
            1: 0.1, # smf val start with index one bc zeroth order coeff is assumed to be zero for RooChebychev
            2: 0.01,
            3: -0.019,
        },
        1: {
            1: -0.05,
            2: 0.02,
            3: -0.01,
        },
    }
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    nbins = 800 # Bin size = 50 MeV -> line 1762 of RERECO AN
    mass_name = f"mh_ggh"
    mass =  rt.RooRealVar(mass_name,mass_name,120,110,150)
    mass.setBins(nbins)
    # defin fit range
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only

    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    subCat_dataHists = {}
    model_dict = {}
    misc_dict = {} # to keep python from erasing variables for Roofit to run properly

    # intialize the core functions 
    BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, dof)
    
    # for cat_ix in range(5):
    for cat_ix in [0,1]:
        subCat_name = f"subCat{cat_ix}"
        sample.defineType(subCat_name)
        # generate
        subCat_filter = (processed_eventsData["subCategory_idx"] == cat_ix)
        subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
        subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
        roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
        roo_datasetData.SetName(subCat_name+"_Dataset_Data")
        roo_histData = rt.RooDataHist(subCat_name+"_DataHist_Data",f"binned version of roo_datasetData of subcat {cat_ix}", rt.RooArgSet(mass), roo_datasetData)
        print(f"roo_datasetData.GetName(): {roo_datasetData.GetName()}")

        smfVarList = []
        smf_order= poly_order_by_cat[cat_ix]
        print(f"smf_order: {smf_order}")
        for ix in range(smf_order): 
            name = subCat_name+f"SMF_Coeff{ix+1}"
            start_val = smf_coeffStartVals[cat_ix][ix+1]
            smf_coeff = rt.RooRealVar(name, name,  start_val, -10, 10)
            smfVarList.append(smf_coeff)
    
        name = subCat_name + "_SMF"
        polynomial_model = rt.RooChebychev(name, name, mass, smfVarList)
        name = subCat_name+f"_SMFx{BWZ_Redux.GetName()}"
        final_BWZ_Redux = rt.RooProdPdf(name, name, [polynomial_model,BWZ_Redux])
        # final_BWZ_Redux.Print("v")
        
        # add the information to relevant dictionaries
        model_dict[subCat_name] = final_BWZ_Redux
        subCat_dataHists[subCat_name] = roo_histData
        misc_dict[roo_datasetData.GetName()] = roo_datasetData
        misc_dict[polynomial_model.GetName()] = polynomial_model
        for smf_coeff in smfVarList:
            misc_dict[smf_coeff.GetName()] = smf_coeff
        # misc_dict.update(params_redux)

    print(f"misc_dict: {misc_dict}")
    # Construct combined dataset
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import=subCat_dataHists,
    ) 
    # combData.Print("v")
    # sample.Print("v")
    # combHist = rt.RooDataHist("CombDataHist_Data",f"CombDataHist_Data", rt.RooArgSet(mass), combData)
    # combHist.Print("v")

    # Construct a simultaneous pdf using category sample as index: associate model
    simPdf = rt.RooSimultaneous("simPdf", "simultaneous pdf", model_dict, sample)
    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, )

    # plot 
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
    roo_datasetData.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
    for model_name, model in model_dict.items():
        color = subCatColorMap[model_name]
        model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model.GetName(), LineColor=color)
        legend.AddEntry(frame.getObject(int(frame.numItems())-1),model.GetName(), "L")

    frame.Draw()
    legend.Draw()        
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"./quick_plots/simultaneousPlotTest.pdf")
        
        