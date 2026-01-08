import ROOT
import ROOT as rt
import matplotlib.pyplot as plt
import numpy as np

def chi2_ndf_manual(pdf, data_hist, x, regions, nfloat):
    chi2 = 0.0
    nbin = 0
    data_hist = data_hist.createHistogram("data_hist", x)
    for i in range(1, data_hist.GetNbinsX()+1):
        xc = data_hist.GetXaxis().GetBinCenter(i)

        # check if bin is in any allowed region
        if not any(lo <= xc <= hi for (lo, hi) in regions):
            continue

        obs = data_hist.GetBinContent(i)
        err = data_hist.GetBinError(i)
        if err <= 0:
            print(f"NOTE: bin {i} has error zero. Skipping!")
            continue

        x.setVal(xc)
        exp = pdf.getVal(ROOT.RooArgSet(x)) * data_hist.Integral() * data_hist.GetBinWidth(i)

        chi2 += (obs - exp)**2 / err**2
        nbin += 1

    # print(f"chi2_ndf_manual data_hist.GetNbinsX(): {data_hist.GetNbinsX()}")
    # print(f"chi2_ndf_manual nbin: {nbin}")
    ndf = nbin - nfloat
    return chi2, ndf

def getNewRangeHist(hist2copy: rt.TH1D, new_hist_name: str, xlow_new: float, xhigh_new: float):
    """
    """
    # get binwise content
    # Initialize result dictionary
    bin_dict = {"xlow": [], "xhigh": [], "content": []}
    
    # Fill dictionary with bin info
    hist = hist2copy
    for i in range(1, hist.GetNbinsX() + 1):
        xlow = hist.GetBinLowEdge(i)
        xhigh = xlow + hist.GetBinWidth(i)
        content = hist.GetBinContent(i)
        if (xlow >= xlow_new) and (xhigh <= xhigh_new):
            bin_dict["xlow"].append(xlow)
            bin_dict["xhigh"].append(xhigh)
            bin_dict["content"].append(content)
        
    # print(bin_dict)
    new_nbins = len(bin_dict["content"])

    new_hist = rt.TH1D(new_hist_name, new_hist_name, new_nbins, xlow_new, xhigh_new)
    for i in range(1, new_hist.GetNbinsX() + 1):
        content = bin_dict["content"][i-1]
        new_hist.SetBinContent(i, content)

    # # sanity check
    # for i in range(1, new_hist.GetNbinsX() + 1):
    #     xlow = new_hist.GetBinLowEdge(i)
    #     xhigh = xlow + new_hist.GetBinWidth(i)
    #     content = new_hist.GetBinContent(i)
    #     print(f"{new_hist_name} bin{i} {(xlow, xhigh, content)}")
    # print(f"new_nbins: {new_nbins}")
    # raise ValueError
    return new_hist

def getGOF_KS(x: rt.RooRealVar, data: rt.RooDataHist, pdf: rt.RooAbsPdf, cat_name:str, save_path:str):
    """
    Get KS value for specific value
    """
    # raise ValueError
    nbins = x.getBins()
    var_name = x.GetName()
    # # Generate toy dataset
    # data = pdf.generate(rt.RooArgSet(x), 1000)
    hist_data_orig = data.createHistogram(var_name).Clone("clone")# clone it just in case
    # Create a histogram of the PDF
    hist_pdf_orig = pdf.createHistogram(var_name, nbins)


    plot_line_width = 0.5
    # -------------------------------------------
    # Sanity check: plot the pdf and cdf of signal region PDF
    # -------------------------------------------
    # data_counts = np.array([hist_data_orig.GetBinContent(i) for i in range(1, hist_data_orig.GetNbinsX()+1)])
    pdf_counts = np.array([hist_pdf_orig.GetBinContent(i) for i in range(1, hist_pdf_orig.GetNbinsX()+1)])
    # print(f"data_counts: {data_counts}")
    # print(f"pdf_counts: {pdf_counts}")
    pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
    
    bin_centers = []
    hist = hist_pdf_orig
    for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
        center = hist.GetBinCenter(i)
        bin_centers.append(center)
    bin_centers = np.array(bin_centers)
    plt.plot(bin_centers, pdf_cdf, label='PDF CDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    if save_path != "":
        plt.savefig(f"{save_path}/GoF_cdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()
    
    # Draw the normalized pdf histogram:
    plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF')
    plt.xlabel('mass')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    if save_path != "":
        plt.savefig(f"{save_path}/GoF_pdfs_SignalFitRange_{cat_name}.pdf")
    plt.clf()


    # -------------------------------------------
    # Do KS test
    # -------------------------------------------
    return_dict = {}
    for test_range_name in ["loSB", "hiSB"]:
        xlow, xhigh = list(x.getRange(test_range_name))
        data_hist_new = getNewRangeHist(hist_data_orig, f"data_hist_{test_range_name}", xlow, xhigh)
        pdf_hist_new = getNewRangeHist(hist_pdf_orig, f"pdf_hist_{test_range_name}", xlow, xhigh)
        data_counts = np.array([data_hist_new.GetBinContent(i) for i in range(1, data_hist_new.GetNbinsX()+1)])
        pdf_counts = np.array([pdf_hist_new.GetBinContent(i) for i in range(1, pdf_hist_new.GetNbinsX()+1)])
        # print(f"data_counts: {data_counts}")
        # print(f"pdf_counts: {pdf_counts}")
        # nevents=np.sum(data_counts)
        data_cdf = np.cumsum(data_counts) / np.sum(data_counts)
        pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
        ks_statistic = np.max(np.abs(data_cdf - pdf_cdf))
        print(f"ks_statistic {cat_name} {test_range_name}: {ks_statistic}")
        nevents = data_hist_new.Integral()
        
        return_dict[test_range_name] = {
            "ks_statistic": ks_statistic,
            "nevents" : nevents,
                                       }
        
    
        # Draw the cdf histogram
        # Extract bin centers (excluding underflow/overflow)
        bin_centers = []
        hist = data_hist_new
        for i in range(1, hist.GetNbinsX() + 1):  # 1-based bin index
            center = hist.GetBinCenter(i)
            bin_centers.append(center)
        bin_centers = np.array(bin_centers)
        plt.plot(bin_centers, data_cdf, label='Data CDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_cdf, label='PDF CDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        if save_path != "":
            plt.savefig(f"{save_path}/GoF_cdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()

        # Draw the normalized pdf histogram:
        plt.plot(bin_centers, data_counts/np.sum(data_counts), label='Data PDF', linewidth=plot_line_width)
        plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='PDF PDF', linewidth=plot_line_width)
        plt.xlabel('mass')
        plt.ylabel('')
        # plt.title('Simple NumPy Plot')
        plt.legend()
        plt.grid(True)
        if save_path != "":
            plt.savefig(f"{save_path}/GoF_pdfs_{test_range_name}_{cat_name}.pdf")
        plt.clf()
        
    return return_dict