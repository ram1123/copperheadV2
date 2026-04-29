import ROOT
import numpy as np
import argparse
import glob
ROOT.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import os

def extractBiasNPlot(in_index,out_index):
    plot_dir = "plots"
    in_filelist = glob.glob(f"slurmJobs/*in{in_index}_out{out_index}*/fitDiagnosticsbias*.root")
    
    # out_fname = f"{plot_dir}/{args.out_file}.pdf"
    out_fname = f"{plot_dir}/bias_in{in_index}_out{out_index}.pdf"

    print(f"in_filelist: {len(in_filelist)}")
    # print(f"in_filelist: {(in_filelist)}")
    # raise ValueError
    r_truth = 1

    truth_function = "exp"
    fit_function = "poly"

    name = "truth_%s_fit_%s" % (truth_function, fit_function)


    hist_pull = ROOT.TH1F("pull_%s" % name, "Pull distribution: truth=%s, fit=%s" % (truth_function, fit_function), 80, -5, 5)
    # hist_pull = ROOT.TH1F("pull_%s" % name, "Pull distribution: truth=%s, fit=%s" % (truth_function, fit_function), 1000, -5, 5)
    hist_pull.GetXaxis().SetTitle("Pull = (r_{fit}-r_{truth})/#sigma_{fit}")
    hist_pull.GetYaxis().SetTitle("Entries")
    bias_values = []
    for fname in in_filelist:
        # Open file with fits
        try:
            f = ROOT.TFile(fname)
            t = f.Get("tree_fit_sb")
            # t.Print("V")

            

            sigma_values = np.array([])

            for i_toy in range(t.GetEntries()):
                # Best-fit value
                t.GetEntry(i_toy)
                r_fit = getattr(t, "r")
                rHiErr = getattr(t, "rHiErr")
                rLoErr = getattr(t, "rLoErr")
                bias_fit = (r_fit-1)/(0.5*(rHiErr+rLoErr))
                hist_pull.Fill(bias_fit)
                bias_values.append(bias_fit)
        except Exception as e:
            print(f"Error: {e}")
            continue
    canv = ROOT.TCanvas()
    hist_pull.Draw()

    # Fit Gaussian to pull distribution
    ROOT.gStyle.SetOptFit(111)
    hist_pull.Fit("gaus")

    canv.SaveAs(out_fname)
    print(f"bias_values: {bias_values}")
    return np.mean(bias_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    # "--in_file",
    # dest="in_file",
    # default=None,
    # action="store",
    # help="save path to store stage1 output files",
    # )
    # parser.add_argument(
    # "--in_index",
    # dest="in_index",
    # default=None,
    # action="store",
    # help="save path to store stage1 output files",
    # )
    # parser.add_argument(
    # "--out_index",
    # dest="out_index",
    # default=None,
    # action="store",
    # help="save path to store stage1 output files",
    # )
    # parser.add_argument(
    # "--out_file",
    # dest="out_file",
    # default=None,
    # action="store",
    # help="save path to store stage1 output files",
    # )
    # n_fitfunction_candidates = 7
    n_fitfunction_candidates = 8
    corePdf_indexes = [1]
    fit_function_indexes = range(n_fitfunction_candidates)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # fit_function_indexes = [3]
    results = []
    for in_index in fit_function_indexes:
        for out_index in corePdf_indexes:
            mean_bias = extractBiasNPlot(in_index,out_index)
            results.append({
                "in_index": in_index,
                "out_index": out_index,
                "mean_bias": mean_bias
            })
    # Create DataFrame
    df = pd.DataFrame(results, columns=["in_index", "out_index", "mean_bias"])

    print(df)
    df.to_csv("mean_bias_inIndex.csv")

    df_outIndexSort = df.sort_values(by="out_index", ascending=True).reset_index(drop=True)
    df_outIndexSort.to_csv("mean_bias_outIndex.csv")

    # index map for reference
    cat_map = {
        0 : "BWZRedux",
        1 : "BwzGamma",
        2 : "BWZxBern",
        3 : "sumExp",
        4 : "PowerLaw",
        5 : "FEWZxBern",
        6 : "LandauxBern",
        7 : "Polynomial",
    }