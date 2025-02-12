import ROOT as rt
import ROOT 
import numpy as np
import matplotlib.pyplot as plt

def plotROC(background_yields, signal_yields, label:str):
    # Compute cumulative sums (assuming a cut-based selection, e.g., loosest to tightest cut)
    signal_cumsum = np.cumsum(signal_yields)
    background_cumsum = np.cumsum(background_yields)
    # Compute total signal and background
    total_signal = np.sum(signal_yields)
    total_background = np.sum(background_yields)
    
    # Compute efficiencies
    signal_efficiency = signal_cumsum / total_signal
    background_efficiency = background_cumsum / total_background
    print(f"{label} signal_efficiency : {signal_efficiency}")
    print(f"{label} background_efficiency : {background_efficiency}")
    plt.plot(signal_efficiency, background_efficiency, marker='o', linestyle='-', label=f"{label}")



ucsd_bkg_yields = []
ucsd_sig_yields = []

for cat_ix in range(5):
    ws = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"]
    bkg_yield = ws.obj(f"data_cat{cat_ix}_ggh").sumEntries()
    ucsd_bkg_yields.append(bkg_yield)

    ws = rt.TFile(f"../ucsd_workspace/workspace_sig_cat{cat_ix}_ggh.root")["w"]
    sig_yield = ws.obj(f"data_ggH_cat{cat_ix}_ggh_m125").sumEntries()
    ucsd_sig_yields.append(sig_yield)


ucsd_bkg_yields = np.array(ucsd_bkg_yields)
ucsd_sig_yields = np.array(ucsd_sig_yields)

print(f"ucsd_bkg_yields : {ucsd_bkg_yields}")
print(f"ucsd_sig_yields : {ucsd_sig_yields}")
print(f"ucsd_bkg_yields percentage: {ucsd_bkg_yields/sum(ucsd_bkg_yields)}")
print(f"ucsd_sig_yields percentage: {ucsd_sig_yields/sum(ucsd_sig_yields)}")

purdue_bkg_yields = []
purdue_sig_yields = []

for cat_ix in range(5):
    ws = rt.TFile(f"my_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"]
    bkg_yield = ws.obj(f"data_cat{cat_ix}_ggh").sumEntries()
    purdue_bkg_yields.append(bkg_yield)

    ws = rt.TFile(f"my_workspace/workspace_sig_cat{cat_ix}_ggh.root")["w"]
    sig_yield = ws.obj(f"data_ggH_cat{cat_ix}_ggh").sumEntries()
    purdue_sig_yields.append(sig_yield)


purdue_bkg_yields = np.array(purdue_bkg_yields)
purdue_sig_yields = np.array(purdue_sig_yields)

print(f"purdue_bkg_yields : {purdue_bkg_yields}")
print(f"purdue_sig_yields : {purdue_sig_yields}")
print(f"purdue_bkg_yields percentage: {purdue_bkg_yields/sum(purdue_bkg_yields)}")
print(f"purdue_sig_yields percentage: {purdue_sig_yields/sum(purdue_sig_yields)}")


# add old yields for reference
old_purdue_cum_sum_sig_eff = [0.29999875, 0.64999815, 0.79999898, 0.94999867, 1.    	]
old_purdue_cum_sum_bkg_eff = [0.49970419, 0.81569565, 0.91304606, 0.98740517, 1.    	]


# Plot ROC curve
plt.figure(figsize=(7, 5))
# add old plots for ref
# plt.plot(old_purdue_cum_sum_sig_eff, old_purdue_cum_sum_bkg_eff, marker='o', linestyle='-', label=f"Purdue Old", color="green")
# plot the rest
plotROC(ucsd_bkg_yields, ucsd_sig_yields, "UCSD")
plotROC(purdue_bkg_yields, purdue_sig_yields, "Purdue New")

plt.ylabel("Background Efficiency")
plt.yscale("log")
plt.xlabel("Signal Efficiency")
plt.title("ROC Curve")

plt.grid()
plt.legend()
plt.savefig("quickROC_curve.png")
plt.savefig("quickROC_curve.pdf")


# zoom in for cat3 and 4 portion
plt.yscale('linear')
plt.xlim(0.8, 1.0)
plt.ylim(0.9,1.0)
plt.savefig("quickROC_curve_zoom.png")
plt.savefig("quickROC_curve_zoom.pdf")