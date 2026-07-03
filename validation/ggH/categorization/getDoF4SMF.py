import awkward as ak
import dask_awkward as dak
import argparse
import os
import numpy as np
from collections import OrderedDict
from modules.selection import filterRegion
from modules.GoF_utils import get_fStats_n_pVal, chi2_ndf_manual
from modules.utils import logger
import ROOT
from modules.fit_functions import plot_6_26, getShapeModifierHist, getRatioHist, getUnityHistBand, zero_yield_in_range
import pandas as pd

def tranformBDT_score(computed_zip):
    """
    simple function that changes the range from [0,1] to [-1,-1]
    """
    score_name = "BDT_score"
    BDT_score = computed_zip[score_name]
    computed_zip[score_name] = (BDT_score-0.5)*2
    return computed_zip

def fillSampleValues(events, sample_dict, sample_groups, sample: str):
    sample_name = sample.lower()
    # find which sample group sample_name belongs to
    sample_group = next((key for key, values in sample_groups.items() if sample_name in values), None)
    print(f"sample_group: {sample_group}")
    if sample_group in sample_dict.keys():
        sample_info = sample_dict[sample_group]
        fields2load = sample_info.keys() # dimuon_mass, wgt_nominal
        
        # compute in parallel fields to load
        computed_zip = ak.zip({
            field : events[field] for field in fields2load
        }).compute()
        computed_zip = tranformBDT_score(computed_zip)
        # print(f"computed_zip.BDT_score min: {np.min(computed_zip.BDT_score)}")
        # print(f"computed_zip.BDT_score max: {np.max(computed_zip.BDT_score)}")
        # add the computed fields to sample_dict 
        for field in fields2load:
            sample_dict[sample_group][field].append(
                ak.to_numpy(computed_zip[field])
            )
    else:
        print(f"sample {sample_group} not present in sample_dict!")

    return sample_dict
        
    # if sample.lower() == "data":
    #     full_load_path = load_path+f"*data.parquet" 
    # elif sample.lower() == "ggh":
    #     full_load_path = load_path+f"*sigMC_ggh.parquet" 
    # elif sample.lower() == "vbf":
    #     full_load_path = load_path+f"*sigMC_vbf.parquet" 
    # elif sample.lower() == "dy":
    #     full_load_path = load_path+f"*bkgMC_dy.parquet" 
    # elif sample.lower() == "ewk":
    #     full_load_path = load_path+f"*bkgMC_ewk.parquet" 
    # elif sample.lower() == "tt":
    #     full_load_path = load_path+f"*bkgMC_tt.parquet" 
    # elif sample.lower() == "st":
    #     full_load_path = load_path+f"*bkgMC_st.parquet" 
    # elif sample.lower() == "ww":
    #     full_load_path = load_path+f"*bkgMC_ww.parquet" 
    # elif sample.lower() == "wz":
    #     full_load_path = load_path+f"*bkgMC_wz.parquet" 
    # elif sample.lower() == "zz":
    #     full_load_path = load_path+f"*bkgMC_zz.parquet" 


def getDataDict(data_sample_dict, plot_var, apply_blind=True):
    data_val = np.concatenate(data_sample_dict[plot_var], axis=0)
    data_wgt = np.concatenate(sample_dict["data"]["wgt_nominal"], axis=0)

    if apply_blind:
        dimuon_mass = np.concatenate(data_sample_dict["dimuon_mass"], axis=0)
        h_peak = (dimuon_mass > 115) & (dimuon_mass < 135)
        blind_filter = ~h_peak
        data_val = data_val[blind_filter]
        data_wgt = data_wgt[blind_filter]

    data_dict = {
        "values" : data_val,
        "weights": data_wgt
    }
    return data_dict


def getBinnedRooHistMass(x, x_name, x_arr, hist_name):
    roo_dataset = ROOT.RooDataSet.from_numpy({x_name: x_arr}, [x]) # it's data, so no need for weights
    roo_histData = ROOT.RooDataHist(hist_name,hist_name, ROOT.RooArgSet(mass), roo_dataset)
    return roo_histData

def applyFitNGetResults(core_func, x, fit_hist, fit_range, nFit_params, chi2_regions = [(110,115), (135,150)], device="cpu"):
    _ = core_func.fitTo(fit_hist, ROOT.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True)
    fit_result = core_func.fitTo(fit_hist, ROOT.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    
    chi2, NDF = chi2_ndf_manual(core_func, fit_hist, x, chi2_regions, nFit_params)
    return core_func, fit_result, chi2, NDF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-label",
    "--label",
    dest="label",
    default="",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="plots",
    action="store",
    help="string value production category we're working on",
    )
    parser.add_argument(
    "-samp",
    "--samples",
    dest="samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of samples to process for stage2. Current valid inputs are data, signal and DY",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="label",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "-base",
    "--base_path",
    dest="base_path",
    default="/depot/cms/users/yun79/hmm/copperheadV1clean",
    action="store",
    help="",
    )
    args = parser.parse_args()
    if len(args.samples) == 0:
        print("samples list is zero!")
        raise ValueError
    year = args.year
    if year == "all":
        year_param = "*"
    elif year == "2016":
        year_param = "2016*"
    else:
        year_param = year
    load_path =f"{args.base_path}/{args.label}/{args.category}/stage2_output/{year_param}/"
    # events = dak.from_parquet(f"{load_path}/*data.parquet")
    # print(events.fields)
    print(f"load_path : {load_path}")
    print(f"args.samples: {args.samples}")

    # lumi_dict = {
    #     "2018" : 59.83,
    #     "2017" : 41.48,
    #     "2016postVFP": 19.50,
    #     "2016preVFP": 16.81,
    #     "2016": 36.3,
    #     # "all" : 137, # Run2
    #     "2022preEE": "7.9804",
    #     "2022postEE": "26.6717",
    #     "2023": "17.7940",
    #     "2023BPix": "9.4510",
    #     "2024": "108.9600",
    #     "all": "170.8571", # 2022 - 2024
    # }
    # lumi_val = lumi_dict[year]

    possible_samples = ["data"]
    full_save_path = f"{args.save_path}/{args.label}_x_{args.category}/{args.year}_{args.region}/SMF_fTest"
    os.makedirs(full_save_path, exist_ok=True)
    
    for sample in args.samples:
        if sample.lower() == "data":
            full_load_path = load_path+f"*data*.parquet" 
        else:
            print(f"Warning: {sample} is unsupported sample!")
            continue
        print(f"full_load_path: {full_load_path}")
    
        # -----------------------------------------------
        # Load events and filter to subcat
        # -----------------------------------------------
        
        events = dak.from_parquet(full_load_path)
        # apply h-sideband selection and ggH channel category
        # _, events = filterRegion(events, region="h-sidebands")
        _, events = filterRegion(events, region="signal")
        fields2load = [
            "dimuon_mass",
            "wgt_nominal",
            'subCategory_idx',
        ]
        processed_events = ak.zip({ # compute the fields
            field : events[field] for field in fields2load
        }).compute()
        print(processed_events)
        
        # Create observables and set fit setting
        mass_name = "mh_ggh"
        mass = ROOT.RooRealVar(mass_name, mass_name, 120, 110, 150)
        nbins = 800
        mass.setBins(nbins)
        mass.setRange("hiSB", 135, 150 )
        mass.setRange("loSB", 110, 115 )
        mass.setRange("h_peak", 115, 135 )
        mass.setRange("full", 110, 150 )
        fit_range = "hiSB,loSB" 
        plot_range="full"
        device = "cpu"
        
        

        # -----------------------------------------------
        # Fit core function of choice over h-sidebands
        # -----------------------------------------------
        # Initialize core function
        name = f"BWZ_Redux_a_coeff"
        a_coeff = ROOT.RooRealVar(name,name, 3.9611e-02,-0.5,0.5) # this converges to -1.2561e-03
        name = f"BWZ_Redux_b_coeff"
        b_coeff = ROOT.RooRealVar(name,name, -9.9358e-05,-0.5,0.5) # this converges to  2.1729e-05,
        name = f"BWZ_Redux_c_coeff"
        c_coeff = ROOT.RooRealVar(name,name, 1.9978e+00,1,2.5) # this converges to 1.7082e+00
        core_func = ROOT.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff)
        corefunc_nFit_params = 3
        
        # get histogram to fit to
        mass_arr  = ak.to_numpy(processed_events["dimuon_mass"]) # convert to numpy for ROOT.RooDataSet
        hist_name = "allBdtCatDimuMass"
        roo_histData_allBdtCat = getBinnedRooHistMass(mass, mass_name, mass_arr, hist_name)
        
        
        # fit
        chi2_regions = [(110,115), (135,150)]
        core_func, fit_result, chi2, NDF = applyFitNGetResults(core_func, mass, roo_histData_allBdtCat, fit_range, corefunc_nFit_params, chi2_regions=chi2_regions, device=device)
        print()
        chi2ndf = chi2/NDF
        fit_result.Print()
        logger.info(f"core_func: {core_func}")
        logger.info(f"chi2ndf: {chi2ndf}")


        
        # validation Plot
        subCat_dataHists = [
        roo_histData_allBdtCat,
        ]
        core_funcs = {
            "BWZRedux" : core_func
        }
        for coreFuncName, core_func in core_funcs.items():
            save_fname = f"{full_save_path}/fig6_26_{coreFuncName}"
            pdf_l = [
                core_func
            ]
            plot_6_26(mass, subCat_dataHists, pdf_l, fit_result, save_fname, coreFuncName=coreFuncName, unblind=False, applyBkgCompName=False)

        # -----------------------------------------------
        # Obtain dimuon mass roohists all the BDT categories
        # -----------------------------------------------
        nBdtCats = len(np.unique(processed_events.subCategory_idx))
        logger.info(f"nBdtCats: {nBdtCats}")
        rooHist_ByCat = {}
        for ix in range(nBdtCats):
            all_mass_arr = ak.to_numpy(processed_events["dimuon_mass"])
            cat_filter = processed_events.subCategory_idx == ix
            mass_arr  = all_mass_arr[cat_filter] # convert to numpy for ROOT.RooDataSet
            roo_hist = getBinnedRooHistMass(mass, mass_name, mass_arr, hist_name)
            rooHist_ByCat[f"subCat{ix}"] = roo_hist
        

        # quick validation
        for cat_name, roo_hist in rooHist_ByCat.items():
            logger.info(f"roo_hist sum entries {cat_name}: {roo_hist.sumEntries()}")
            frame = mass.frame()
            roo_hist.plotOn(frame, ROOT.RooFit.CutRange(fit_range))
            # roo_hist.plotOn(frame)
            # Draw
            c = ROOT.TCanvas("c", "c", 800, 600)
            frame.Draw()
            c.SaveAs(f"{full_save_path}/{cat_name}.png")
            
            
        total_sum = np.sum([roo_hist.sumEntries() for roo_hist in rooHist_ByCat.values()])
        logger.info(f"rooHist_ByCat total sum entries: {total_sum}")
        logger.info(f"roo_histData_allBdtCat sum entries: {roo_histData_allBdtCat.sumEntries()}")


        # -----------------------------------------------
        # Obtain ratio plot over all the BDT categories
        # -----------------------------------------------
        SMF_hists = {}
        for cat_name, roo_hist in rooHist_ByCat.items():
            # roo_hist_shapModifier = getShapeModifierHist(mass, roo_histData_allBdtCat, roo_hist, normalize=True, nbins=nbins)
            # roo_hist_shapModifier = getShapeModifierHist(mass, roo_histData_allBdtCat, roo_hist, normalize=True, nbins=100)
            roo_hist_shapModifier = getShapeModifierHist(mass, roo_histData_allBdtCat, roo_hist, normalize=True, nbins=nbins)
            SMF_hists[cat_name] = roo_hist_shapModifier


        # quick validation
        for cat_name, roo_hist in SMF_hists.items():
            frame = mass.frame()
            roo_hist.plotOn(frame, ROOT.RooFit.CutRange(fit_range))
            # roo_hist.plotOn(frame)
            # Draw
            c = ROOT.TCanvas("c", "c", 800, 600)
            frame.Draw()
            c.SaveAs(f"{full_save_path}/{cat_name}_smfHist.pdf")

            

        # -----------------------------------------------
        # Do f-test to obtain the best dof for SMF polynomial
        # -----------------------------------------------
        # use chi2_ndf_manual() for blineded chi2

        # iterate for each bdt cat
        max_dof = 4
        # max_dof = 2
        df_dict = {
            "bdt_cat" : [],
            "chi2" : [],
            "NDF" : [],
            "poly_dof" : [],
            
        }
        for cat_name, roo_hist_cat in SMF_hists.items():
            # intialize the Chebychev polynomials
            coeff_lists = [] # keep a list of coeff so they don't get deleted
            polynomial_dict = {}
            fit_dict = {}
            for dof in range(1, max_dof+1):
                # Draw
                canvas = ROOT.TCanvas("c", "c", 800, 600)
    
                # Define upper and lower pads
                pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 1, 1.0)
                pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 1, 0.3)
                
                # Adjust margins
                pad1.SetBottomMargin(0)  # Upper plot does not need bottom margin
                pad2.SetTopMargin(0)     # Lower plot does not need top margin
                pad2.SetBottomMargin(0.3)
        
                pad1.SetTicks(2, 2)
                pad2.SetTicks(2, 2)
                pad1.Draw() # value plot
                pad2.Draw() # ratio plot
        
                # Top pad start
                pad1.cd()
                # coeffs = ROOT.RooArgList()
                coeffs = []
                # Create 'dof' floating coefficients
                for i in range(dof):
                    c = ROOT.RooRealVar(
                        f"{cat_name}_c{i+1}_dof{dof}",
                        f"{cat_name}_c{i+1}_dof{dof}",
                        0.01,      # initial value
                        -10.0,     # min
                        10.0       # max
                    )
                    coeffs.append(c)
            
                poly_model = ROOT.RooChebychev(
                    f"{cat_name}_cheb_dof{dof}",
                    f"{cat_name}_Chebychev order {dof}",
                    mass,
                    coeffs
                )
                coeff_lists.append(coeffs)
                
            
                poly_model, fit_result, chi2, NDF = applyFitNGetResults(poly_model, mass, roo_hist_cat, fit_range, dof, chi2_regions=chi2_regions, device=device)
                fit_result.Print()
                logger.info(f"chi2/NDF: {chi2/NDF}")
                # raise ValueError
                polynomial_dict[dof] = poly_model
                fit_dict[dof] = [chi2, NDF]

                # add in the data
                df_dict["bdt_cat"].append(cat_name) 
                df_dict["chi2"].append(chi2) 
                df_dict["NDF"].append(NDF) 
                df_dict["poly_dof"].append(dof) 

                # ------------------------
                # quick validation plot with pull plot
                # ------------------------
                frame = mass.frame()
                roo_hist_cat.plotOn(frame, ROOT.RooFit.CutRange(fit_range))
                
                roo_hist_cat.plotOn(frame, Invisible=True) # plot invisible data before pdf
                poly_model.plotOn(frame, ROOT.RooFit.NormRange(fit_range), ROOT.RooFit.Range(plot_range), LineColor=ROOT.kRed)
                
                frame.Draw()


                # Bottom pad start
                pad2.cd()
                #-----------------------------------
                # plot again but with the order changed to obtain the correct pull histogram
                #-----------------------------------
                pull_frame = mass.frame()
                roo_hist_cat.plotOn(pull_frame, Invisible=True) # plot invisible data before pdf
                poly_model.plotOn(pull_frame, ROOT.RooFit.NormRange(fit_range), ROOT.RooFit.Range(plot_range), LineColor=ROOT.kRed)
                roo_hist_cat.plotOn(pull_frame, ROOT.RooFit.CutRange(fit_range))
                pullHist = pull_frame.pullHist()


                #-----------------------------------
                # with pull hist, do pull plot
                #-----------------------------------
                ratio_frame= mass.frame()
                ratio_frame.addPlotable(pullHist, "P") 
                ratio_frame.Draw()


                # ---------------------------------------------------------------------
                # FIXME: this is code for fig 6.23-like bottom pad. maybe use it in the future. If not, just delete it
                
                # x = mass
                # roo_hist_cat_th1 = roo_hist_cat.createHistogram(x.GetName())
                # ratio_hist = getRatioHist(x, roo_hist_cat_th1, poly_model)

                # ratio_hist.GetYaxis().SetRangeUser(0.9, 1.1)
                # ratio_hist.SetTitle("")
                # ratio_hist.GetYaxis().SetTitle("Data/Pred")
                # ratio_hist.GetXaxis().SetTitleSize(0.08)
                # ratio_hist.GetXaxis().SetLabelSize(0.08)
                # ratio_hist.GetXaxis().SetTitle("m_{\mu\mu} [GeV]")
                # h_band = getUnityHistBand(x, poly_model, fit_result, ratio_hist)

                # # start draw
                # h_band.Draw("E2") # draw h band first
                # # change style to add a straight red line
                # h_band_line = h_band.Clone("h_bandClone")
                # h_band_line.SetLineColor(ROOT.kRed)
                # h_band_line.SetLineWidth(2)
                # h_band_line.SetFillStyle(0)  # No fill
                # h_band_line.Draw("HIST L SAME")

                # blinded = True
                # if blinded:
                #     blinded_x_min, blinded_x_max = 115, 135
                #     ratio_hist = zero_yield_in_range(ratio_hist, blinded_x_min, blinded_x_max)
                # ratio_hist.Draw("E1 SAME")
                # ---------------------------------------------------------------------



                
                # save plots
                # ------------------------
                canvas.SaveAs(f"{full_save_path}/{cat_name}_dof{dof}_smfFit.png")
                canvas.SaveAs(f"{full_save_path}/{cat_name}_dof{dof}_smfFit.pdf")
                
                    
            logger.info(f"polynomial_dict: {polynomial_dict}")
            logger.info(f"fit_dict: {fit_dict}")

        df = pd.DataFrame(df_dict)
        df["chi2ndf"] = df["chi2"]/ df["NDF"]
        logger.info(f"df: {df}")

        # obtain the p_values of current dof+1
        df["f_statistic"] = np.nan
        df["p_value"] = np.nan
        for cat in np.unique(df["bdt_cat"]):
            cat_filter = df["bdt_cat"] == cat
            df_filtered = df[cat_filter]
            logger.info(f"df_filtered: {df_filtered}")
            for dof in range(1, max_dof):
                # get low
                dof_filter = df_filtered["poly_dof"] == dof
                chi2_low = df_filtered["chi2"][dof_filter].to_numpy()
                ndf_low = df_filtered["NDF"][dof_filter].to_numpy()
                # get high
                dof_filter = df_filtered["poly_dof"] == dof+1
                chi2_high = df_filtered["chi2"][dof_filter].to_numpy()
                ndf_high = df_filtered["NDF"][dof_filter].to_numpy()
                assert(len(chi2_low)==1)
                f_statistic, p_value = get_fStats_n_pVal(chi2_low, chi2_high, ndf_low, ndf_high)
                # logger.info(f"chi2_low: {chi2_low}")
                # logger.info(f"ndf_low: {ndf_low}")
                logger.info(f"p_value: {p_value}")
                
                # add it to the columns
                dof_filter = df_filtered["poly_dof"] == dof
                df_filtered["f_statistic"][dof_filter] = f_statistic
                df_filtered["p_value"][dof_filter] = p_value
            df[cat_filter] =  df_filtered
            p_threshold = 0.05
            df["should increases dof by one"] = df["p_value"] < p_threshold # is p value is less than threshold, dof+1 is better
            logger.info(f"final df: {df}")
            df.to_csv(f"{full_save_path}/f_test.csv")
            
# chi2_high = fit_func_high.GetChisquare()
#                 ndf_high = fit_func_high.GetNDF()
        
#                 # delta_chi2 = chi2_low - chi2_high
#                 # delta_dof = -(ndf_high - ndf_low) # Negative sign because the order_high is greater than order_low
#                 # f_statistic = (delta_chi2 / chi2_high) * (ndf_high / delta_dof) if delta_dof != 0 and chi2_high != 0 else 0
#                 # p_value = 1 - f.cdf(f_statistic, delta_dof, ndf_high)
#                 f_statistic, p_value = get_fStats_n_pVal(chi2_low, chi2_high, ndf_low, ndf_high)
        
#                 # Log results
#                 if ndf_low == 0 or ndf_high == 0:
#                     logger.error("NDF is zero!")
#                     logger.debug(f"Order {order_low}: χ² = {chi2_low:.2f}, NDF = {ndf_low}")
#                     logger.debug(f"Order {order_high}: χ² = {chi2_high:.2f}, NDF = {ndf_high}")
#                 else:
#                     logger.debug(f"Order {order_low}: χ² = {chi2_low:.2f}, NDF = {ndf_low}, χ²/NDF = {chi2_low/ndf_low:.3f}")
#                     logger.debug(f"Order {order_high}: χ² = {chi2_high:.2f}, NDF = {ndf_high}, χ²/NDF = {chi2_high/ndf_high:.3f}")
#                 logger.debug(f"F-statistic: {f_statistic:.3f}, p-value: {p_value:.5f}")
        
#                 # save_histogram(hist_SF, fit_func_high, order_high, year, njet, target_nbins, outtext)
        
#                 # Decision based on p-value
#                 if p_value < 0.05:
#                     logger.info(f"Significant improvement with polynomial order {order_high} over {order_low}.")
#                     outTextFile.write(f"{year} njet{njet} {target_nbins} bins: Higher-order {order_high} polynomial significantly improves the fit over {order_low}. chi2_low: {chi2_low/ndf_low} vs chi2_high: {chi2_high/ndf_high}\n")
#                     outTextFile_keys.write(f"{year} {njet} {target_nbins} {order_high} {order_low}\n")
