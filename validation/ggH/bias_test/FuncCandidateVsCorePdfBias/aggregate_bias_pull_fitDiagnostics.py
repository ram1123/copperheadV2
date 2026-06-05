import ROOT
import numpy as np
import argparse
import glob
ROOT.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import os

CAT_MAP = {
    0: "BWZRedux",
    1: "BwzGamma",
    2: "BWZxBern",
    3: "sumExp",
    4: "PowerLaw",
    5: "FEWZxBern",
    6: "LandauxBern",
    7: "Polynomial",
}

def load_r_truth():
    cfg_path = "bias_truth_r.txt"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as handle:
            return float(handle.read().strip())
    return float(os.environ.get("BIAS_EXPECT_SIGNAL", "0"))

def extractBiasNPlot(in_index,out_index):
    plot_dir = "plots"
    in_filelist = glob.glob(f"slurmJobs/*in{in_index}_out{out_index}*/fitDiagnosticsbias*.root")
    
    # out_fname = f"{plot_dir}/{args.out_file}.pdf"
    out_fname = f"{plot_dir}/bias_in{in_index}_out{out_index}.pdf"

    print(f"in_filelist: {len(in_filelist)}")
    # print(f"in_filelist: {(in_filelist)}")
    # raise ValueError
    r_truth = load_r_truth()

    truth_function = CAT_MAP.get(in_index, f"idx{in_index}")
    fit_function = CAT_MAP.get(out_index, f"idx{out_index}")

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
                sigma_fit = 0.5 * (rHiErr + rLoErr)
                if sigma_fit <= 0:
                    continue
                bias_fit = (r_fit - r_truth) / sigma_fit
                hist_pull.Fill(bias_fit)
                bias_values.append(bias_fit)
        except Exception as e:
            print(f"Error: {e}")
            continue
    canv = ROOT.TCanvas()
    hist_pull.Draw()

    # Fit Gaussian to pull distribution
    ROOT.gStyle.SetOptFit(111)
    fit_result = hist_pull.Fit("gaus", "QS")

    canv.SaveAs(out_fname)
    print(f"bias_values: {bias_values}")
    gaus = hist_pull.GetFunction("gaus")
    gaus_mean = float(gaus.GetParameter(1)) if gaus else np.nan
    raw_mean = float(np.mean(bias_values)) if bias_values else np.nan
    return raw_mean, gaus_mean

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
    selected_index_file = "selected_truth_function_indices.txt"
    if os.path.exists(selected_index_file):
        with open(selected_index_file, "r") as handle:
            entries = handle.read().strip().split()
        if entries:
            fit_function_indexes = [int(entry) for entry in entries]
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # fit_function_indexes = [3]
    results = []
    for in_index in fit_function_indexes:
        for out_index in corePdf_indexes:
            mean_bias_raw, mean_bias_gaus = extractBiasNPlot(in_index,out_index)
            results.append({
                "in_index": in_index,
                "out_index": out_index,
                "mean_bias_raw": mean_bias_raw,
                "mean_bias_gaus": mean_bias_gaus
            })
    # Create DataFrame
    df = pd.DataFrame(results, columns=["in_index", "out_index", "mean_bias_raw", "mean_bias_gaus"])

    print(df)
    df.to_csv("mean_bias_inIndex.csv", index=False)

    df_outIndexSort = df.sort_values(by="out_index", ascending=True).reset_index(drop=True)
    df_outIndexSort.to_csv("mean_bias_outIndex.csv", index=False)

    # index map for reference
    print(CAT_MAP)
