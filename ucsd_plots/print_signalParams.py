import ROOT as rt
import pandas as pd



mass_name = "mh_ggh"
mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
fit_range = "hiSB,loSB"
plot_range = "full"
nbins = 800
mass.setBins(nbins)

df_total  = pd.DataFrame(columns=['Name', 'Value', 'Uncertainty'])
cat_ix = 0
for cat_ix in range(5):
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    # UCSD
    ws = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")["w"]
    # ws.Print("v")
    ucsd_pdf = ws.obj(f"ggH_cat{cat_ix}_ggh_pdf")
    cat0_hist =  ws.obj(f"data_ggH_cat{cat_ix}_ggh_m125")  
    purdue_ws = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")["w"]
    purdue_pdf = purdue_ws.obj(f"ggH_cat{cat_ix}_ggh_pdf")
    frame = mass.frame()
    
    cat0_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
    ucsd_pdf.plotOn(frame, LineColor=rt.kGreen)
    purdue_pdf.plotOn(frame, LineColor=rt.kBlue)

    frame.Draw()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"pdf_comparison_{cat_ix}.pdf")



# ws.obj(f"ggH_cat{cat_ix}_ggh_spline_norm").Print("w")
# MH = ws.obj(f"MH")
# MH.Print("w")
# print(f"MH.isConstant(): {MH.isConstant()}")
    # # Purdue
    # ws = rt.TFile(f"./my_workspace/workspace_sig_cat{cat_ix}_ggh.root")["w"]
    # MH = ws.obj(f"MH")
    # MH.Print("w")
    # print(f"{cat_ix} MH.isConstant(): {MH.isConstant()}")
    # param_np = ws.obj(f"CMS_hmm_peak_cat{cat_ix}_ggh")
    # param_np.Print("w")
    # print(f"{cat_ix} param_np.isConstant(): {param_np.isConstant()}")
    # # raise ValueError
    