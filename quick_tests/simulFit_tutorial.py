import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt

if __name__ == "__main__":
    load_path = "./processed_events_data.parquet"
    processed_eventsData = ak.from_parquet(load_path)
    print("events loaded!")
    
    # Create model for physics sample
    # -------------------------------------------------------------
    # Create observables
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only
    
    # # Construct signal pdf
    # mean = rt.RooRealVar("mean", "mean", 0, -8, 8)
    # sigma = rt.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
    # gx = rt.RooGaussian("gx", "gx", mass, mean, sigma)
    # intialize the core functions 
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, -0.0146,-0.02,0.03)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.000111,-0.001,0.001)
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 0.462,-5.0,5.0)
    
    name = "Subcat0_BWZ_Redux_dof_3"
    BWZ_Redux_cat0 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
     
    # Construct background pdf
    a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.1, -1, 1)
    a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", 0.004, -1, 1)
    a3_subCat0 = rt.RooRealVar("a3_subCat0", "a3_subCat0", 0.004, -1, 1)
    px = rt.RooChebychev("px", "px", mass, [a0_subCat0, a1_subCat0, a3_subCat0])
     
    # Construct composite pdf
    model = rt.RooProdPdf("model", "model", [BWZ_Redux_cat0, px])
     
    # Create model for control sample
    # --------------------------------------------------------------
     
    # Construct signal pdf.
    # NOTE that sigma is shared with the signal sample model
    # mean_ctl = rt.RooRealVar("mean_ctl", "mean_ctl", -3, -8, 8)
    # gx_ctl = rt.RooGaussian("gx_ctl", "gx_ctl", mass, mean_ctl, sigma)
    name = "Subcat1_BWZ_Redux_dof_3"
    BWZ_Redux_cat1 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    gx_ctl = BWZ_Redux_cat1
    # gx_ctl = BWZ_Redux_cat0
    
    # Construct the background pdf
    a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", -0.1, -1, 1)
    a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", 0.5, -0.1, 1)
    a3_subCat1 = rt.RooRealVar("a3_subCat1", "a3_subCat1", 0.5, -0.1, 1)
    px_ctl = rt.RooChebychev("px_ctl", "px_ctl", mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              a3_subCat1
                             ])
     
    # Construct the composite model
    model_ctl = rt.RooProdPdf("model_ctl", "model_ctl", [gx_ctl, px_ctl])
     
    # Generate events for both samples
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData["subCategory_idx"] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    data = roo_datasetData_subCat0

    subCat_filter = (processed_eventsData["subCategory_idx"] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    data_ctl = roo_datasetData_subCat1

    # # Generate 1000 events in x and y from model
    # data = model.generate({x}, 1000)
    # data_ctl = model_ctl.generate({x}, 2000)
     
    # Create index category and join samples
    # ---------------------------------------------------------------------------
     
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    sample.defineType("physics")
    sample.defineType("control")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            "physics": data, 
            "control": data_ctl
        },
    )
     
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    # Construct a simultaneous pdf using category sample as index: associate model
    # with the physics state and model_ctl with the control state
    simPdf = rt.RooSimultaneous("simPdf", "simultaneous pdf", 
                                  {
                                      "physics": model, 
                                       "control": model_ctl
                                  }, sample
                                 )
     
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    # Perform simultaneous fit of model to data and model_ctl to data_ctl
    start = time.time()

    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", PrintLevel=-1, Save=True)
    end = time.time()
    
    fitResult.Print()
    print(f"runtime: {end-start} seconds")

    # # do plotting
    # name = "Canvas"
    # canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    # canvas.cd()
    
    # frame = mass.frame()
    # legend = rt.TLegend(0.65,0.55,0.9,0.7)


    # # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
    # roo_datasetData_subCat1.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
    # model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model.GetName(), LineColor=rt.kGreen)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),model.GetName(), "L")
    # model_ctl.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model_ctl.GetName(), LineColor=rt.kBlue)
    # legend.AddEntry(frame.getObject(int(frame.numItems())-1),model_ctl.GetName(), "L")

    # frame.Draw()
    # legend.Draw()        
    # canvas.Update()
    # canvas.Draw()
    # canvas.SaveAs(f"./quick_plots/simultaneousPlotTestFromTutorial.pdf")

