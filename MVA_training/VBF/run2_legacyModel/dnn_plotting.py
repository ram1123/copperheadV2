import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import torch
import glob, os

import csv

import logging
from modules.utils import logger

plt.style.use(hep.style.CMS)

import ROOT as R

R.gROOT.SetBatch(True)
R.gStyle.SetOptStat(0)
R.TH1.AddDirectory(False)
R.gErrorIgnoreLevel = R.kError


def safe_weighted_auc(y_true, y_score, sample_weight=None):
    # 1) Build weighted ROC without dropping points
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight,
                            drop_intermediate=False)
    # 2) Numerics guard: clip into [0,1]
    fpr = np.clip(fpr, 0.0, 1.0)
    tpr = np.clip(tpr, 0.0, 1.0)
    # 3) Enforce monotonic non-decreasing FPR
    #    (cummax fixes tiny decreases like 1.00005895 -> 1.00000000)
    fpr = np.maximum.accumulate(fpr)
    # 4) Deduplicate x (FPR) to keep trapezoid stable
    keep = np.r_[True, np.diff(fpr) > 0]
    fpr, tpr = fpr[keep], tpr[keep]
    # 5) Trapezoid rule
    return float(np.trapz(tpr, fpr))


def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.pdf"):
    """
    Plot training and validation loss vs. epoch.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        save_path (str): Path to save the plot. If None, shows the plot.
    """
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    hep.cms.text("Private Work", loc=0)
    plt.grid(True)
    plt.savefig(save_path)
    logger.info(f"[plot_loss_curves] Saved loss curve to {save_path}")
    plt.close()


def plotPrecisionRecall(score_dict, plt_save_path):
    """
    Plot Precision-Recall curve for the given score dictionary.
    """
    fig, ax_main = plt.subplots()
    status = "Private Work 2018"
    CenterOfMass = "13"
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, ax=ax_main)

    for stage, output_dict in score_dict.items():
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]

        precision, recall, _ = precision_recall_curve(
            label_total, pred_total, sample_weight=wgt_total
        )
        ax_main.plot(recall, precision, label=f"{stage}")

    ax_main.set_xlabel("Recall")
    ax_main.set_ylabel("Precision")
    ax_main.set_title("Precision-Recall Curve")
    ax_main.legend()
    plt.savefig(plt_save_path)
    plt.clf()
    plt.close(fig)  # Close the figure to free memory


def plotConfusionMatrix(score_dict, plt_save_path):
    """
    Plot confusion matrix for the given score dictionary.
    """
    # Try importing seaborn, fallback to sklearn if unavailable
    try:
        import seaborn as sns
    except ImportError:
        from sklearn.metrics import ConfusionMatrixDisplay

        sns = None

    # status = "Private Work 2018"
    # CenterOfMass = "13"

    for stage, output_dict in score_dict.items():
        fig, ax_main = plt.subplots()
        # hep.cms.label(data=True, loc=0, ax=ax_main)
        pred_total = output_dict["prediction"]
        label_total = output_dict["label"]
        wgt_total = output_dict["weight"]

        # Convert predictions to binary (0 or 1)
        pred_binary = (pred_total > 0.5).astype(int)

        cm = confusion_matrix(label_total, pred_binary, sample_weight=wgt_total)
        if sns:
            sns.heatmap(cm, annot=True, cmap="Blues", ax=ax_main)
        else:
            ConfusionMatrixDisplay(cm).plot(ax=ax_main, cmap="Blues", values_format="d")
        ax_main.set_title(f"Confusion Matrix - {stage}")
        ax_main.set_xlabel("Predicted")
        ax_main.set_ylabel("True")

        plt.savefig(plt_save_path.replace(".pdf", f"_{stage}.pdf"))
        plt.clf()
        plt.close(fig)  # Close the figure to free memory


def plot_auc_and_loss(history, save_path):

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ep = history["epoch"]
    # Loss (left axis)
    ax1.plot(ep, history["train_loss"], marker="o", label="Train loss")
    ax1.plot(ep, history["val_loss"], marker="s", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    # AUC (right axis)
    ax2 = ax1.twinx()
    ax2.plot(ep, history["train_auc"], marker="^", linestyle="--", label="Train AUC")
    ax2.plot(ep, history["val_auc"], marker="v", linestyle="--", label="Val AUC")
    ax2.set_ylabel("AUC")
    # one legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"[plot_auc_and_loss] Saved {save_path}")


def plot_significance(history, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["significance"], marker="o", label="S/√B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Significance (S/√B)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"[plot_significance] Saved {save_path}")


def plot_lr(history, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["lr"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"[plot_lr] Saved {save_path}")


def _eff_n_from_weights(w):
    # Kish effective sample size
    w = np.asarray(w, dtype=float)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def _weighted_empirical_cdf(x, w):
    # returns (sorted_x, cdf) with weights normalized
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    csum = np.cumsum(ws)
    tot = csum[-1] if len(csum) else 1.0
    cdf = csum / tot
    return xs, cdf


def _ks_weighted(x1, w1, x2, w2):
    # weighted 2-sample KS with asymptotic p-value using effective n
    x = np.concatenate([x1, x2])
    xs = np.unique(np.sort(x))

    # build CDFs on combined grid
    def cdf_on_grid(xa, wa, grid):
        xs, cdf = _weighted_empirical_cdf(xa, wa)
        # right-continuous step function
        idx = np.searchsorted(xs, grid, side="right") - 1
        idx = np.clip(idx, -1, len(xs) - 1)
        out = np.where(idx >= 0, cdf[idx], 0.0)
        return out

    F1 = cdf_on_grid(x1, w1, xs)
    F2 = cdf_on_grid(x2, w2, xs)
    D = np.max(np.abs(F1 - F2)) if len(xs) else 0.0

    # effective n (Kish) per sample -> two-sample effective n
    n1eff = _eff_n_from_weights(w1)
    n2eff = _eff_n_from_weights(w2)
    neff = (n1eff * n2eff) / (n1eff + n2eff) if (n1eff + n2eff) > 0 else 0.0

    # asymptotic p-value (Kolmogorov distribution)
    if neff <= 0 or D <= 0:
        pval = 1.0
    else:
        x = (np.sqrt(neff) + 0.12 + 0.11 / np.sqrt(neff)) * D
        # 2 * sum_{k=1..∞} (-1)^{k-1} exp(-2 k^2 x^2)
        terms = [
            np.exp(-2 * (k * k) * (x * x)) * (1 if (k % 2) == 1 else -1)
            for k in range(1, 50)
        ]
        pval = max(0.0, min(1.0, 2.0 * np.sum(terms)))
    return D, pval


def plot_overtraining_KS_ROOT(
    score_train, y_train, w_train, score_val, y_val, w_val, save_path, nbins=30
):
    # Hist overlaid + KS p-values per class
    can = R.TCanvas("cKS", "cKS", 800, 650)
    frame = R.TH1F("frame", ";DNN score;Normalized entries", nbins, 0.0, 1.0)
    frame.SetStats(0)
    frame.Draw()

    # helper to fill weighted, normalized TH1
    def make_hist(name, arr, w):
        h = R.TH1F(name, name, nbins, 0.0, 1.0)
        for a, ww in zip(arr, w):
            h.Fill(float(a), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        h.SetLineWidth(2)
        return h

    # SIG
    tr_sig = y_train == 1
    va_sig = y_val == 1
    h_tr_sig = make_hist("h_tr_sig", score_train[tr_sig], w_train[tr_sig])
    h_tr_sig.SetLineColor(R.kBlue)
    h_va_sig = make_hist("h_va_sig", score_val[va_sig], w_val[va_sig])
    h_va_sig.SetLineColor(R.kBlue + 2)
    h_va_sig.SetLineStyle(2)
    # BKG
    tr_bkg = y_train == 0
    va_bkg = y_val == 0
    h_tr_bkg = make_hist("h_tr_bkg", score_train[tr_bkg], w_train[tr_bkg])
    h_tr_bkg.SetLineColor(R.kRed)
    h_va_bkg = make_hist("h_va_bkg", score_val[va_bkg], w_val[va_bkg])
    h_va_bkg.SetLineColor(R.kRed + 2)
    h_va_bkg.SetLineStyle(2)

    # Draw
    for h in [h_tr_sig, h_va_sig, h_tr_bkg, h_va_bkg]:
        h.Draw("hist same")
    leg = R.TLegend(0.55, 0.70, 0.88, 0.88)
    leg.AddEntry(h_tr_sig, "Signal train", "l")
    leg.AddEntry(h_va_sig, "Signal val", "l")
    leg.AddEntry(h_tr_bkg, "Bkg train", "l")
    leg.AddEntry(h_va_bkg, "Bkg val", "l")

    # KS (weighted)
    D_sig, p_sig = _ks_weighted(
        score_train[tr_sig], w_train[tr_sig], score_val[va_sig], w_val[va_sig]
    )
    D_bkg, p_bkg = _ks_weighted(
        score_train[tr_bkg], w_train[tr_bkg], score_val[va_bkg], w_val[va_bkg]
    )

    txt = R.TPaveText(0.15, 0.74, 0.52, 0.88, "NDC")
    txt.SetFillColor(0)
    txt.SetBorderSize(0)
    txt.AddText(f"KS(sig): D={D_sig:.3f}, p={p_sig:.3g}")
    txt.AddText(f"KS(bkg): D={D_bkg:.3f}, p={p_bkg:.3g}")

    leg.Draw()
    txt.Draw()
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

def _weighted_brier(labels, probs, weights):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    weights = np.asarray(weights)
    return np.sqrt(np.average((probs - labels) ** 2, weights=weights))  # RMSE form


def plot_calibration_ROOT(probs, labels, weights, save_path, n_bins=10):
    """
    DOC:
    """
    # bin by predicted prob, compute observed frequency (weighted)
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_cent = 0.5 * (edges[:-1] + edges[1:])
    x_pts, y_pts = [], []
    for i in range(n_bins):
        sel = (probs >= edges[i]) & (probs < edges[i + 1])
        wsum = np.sum(weights[sel])
        if wsum <= 0:
            x_pts.append(bin_cent[i])
            y_pts.append(0.0)
            continue
        pred_mean = np.average(probs[sel], weights=weights[sel])
        obs_rate = np.average(labels[sel], weights=weights[sel])
        x_pts.append(pred_mean)
        y_pts.append(obs_rate)

    brier = _weighted_brier(labels, probs, weights)

    can = R.TCanvas("cCal", "cCal", 700, 650)
    frame = R.TH2F(
        "f", ";Predicted probability;Observed positive rate", 10, 0, 1, 10, 0, 1
    )
    frame.SetStats(0)
    frame.Draw()
    # diagonal
    line = R.TLine(0, 0, 1, 1)
    line.SetLineStyle(2)
    line.Draw()
    # points
    gr = R.TGraph(len(x_pts), np.array(x_pts, dtype="f"), np.array(y_pts, dtype="f"))
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(1.0)
    gr.Draw("P same")

    txt = R.TPaveText(0.15, 0.80, 0.55, 0.88, "NDC")
    txt.SetFillColor(0)
    txt.SetBorderSize(0)
    txt.AddText(f"Brier (RMSE): {brier:.4f}")
    txt.Draw()
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

def _scan_thresholds(probs, labels, weights, n=200):
    thr = np.linspace(0, 1, n)
    TPR = np.zeros(n)
    FPR = np.zeros(n)
    PREC = np.zeros(n)
    SSQRTB = np.zeros(n)
    y = np.asarray(labels)
    p = np.asarray(probs)
    w = np.asarray(weights)
    wS = np.sum(w[y == 1])
    wB = np.sum(w[y == 0])
    for i, t in enumerate(thr):
        sel = p >= t
        TP = np.sum(w[sel & (y == 1)])
        FP = np.sum(w[sel & (y == 0)])
        FN = np.sum(w[~sel & (y == 1)])
        TN = np.sum(w[~sel & (y == 0)])

        TPR[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR[i] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        PREC[i] = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        S = TP
        B = FP
        SSQRTB[i] = S / np.sqrt(B) if B > 0 else 0.0
    return thr, TPR, FPR, PREC, SSQRTB


def plot_threshold_scan_ROOT(probs, labels, weights, save_path):
    thr, TPR, FPR, PREC, SSB = _scan_thresholds(probs, labels, weights, n=300)
    # right-axis scaling for S/sqrt(B)
    ssb_max = np.max(SSB) if np.max(SSB) > 0 else 1.0
    SSB_scaled = SSB / ssb_max

    can = R.TCanvas("cThr", "cThr", 820, 650)
    frame = R.TH2F("f", ";Threshold;Rate", 10, 0, 1, 10, 0, 1)
    frame.SetStats(0)
    frame.Draw()

    def mkgraph(y, color):
        g = R.TGraph(len(thr), np.array(thr, dtype="f"), np.array(y, dtype="f"))
        g.SetLineColor(color)
        g.SetLineWidth(2)
        return g

    gTPR = mkgraph(TPR, R.kGreen + 2)
    gFPR = mkgraph(FPR, R.kRed + 1)
    gPREC = mkgraph(PREC, R.kBlue + 1)
    gSSB = mkgraph(SSB_scaled, R.kMagenta + 1)

    for g in [gTPR, gFPR, gPREC, gSSB]:
        g.Draw("L same")

    leg = R.TLegend(0.42, 0.30, 0.68, 0.48)
    leg.AddEntry(gTPR, "TPR (signal eff)", "l")
    leg.AddEntry(gFPR, "FPR (bkg eff)", "l")
    leg.AddEntry(gPREC, "Precision", "l")
    leg.AddEntry(gSSB, "(S/#sqrt{B})/max", "l")
    leg.Draw()

    # right axis for S/#sqrt{B}
    right = R.TGaxis(1.0, 0.0, 1.0, 1.0, 0.0, ssb_max, 510, "+L")
    right.SetTitle("S/#sqrt{B}")
    right.SetTitleOffset(1.2)
    right.Draw()

    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

def _wmean(x, w):
    w = np.asarray(w); x = np.asarray(x)
    s = np.sum(w);   return np.sum(w * x) / s if s > 0 else 0.0

def _wcov(x, y, w):
    mx, my = _wmean(x, w), _wmean(y, w)
    s = np.sum(w)
    return (np.sum(w * (x - mx) * (y - my)) / s) if s > 0 else 0.0

def _wstd(x, w):
    v = _wcov(x, x, w)
    return np.sqrt(v) if v > 0 else 0.0


def plot_score_feature_corr_ROOT_heatmap(abs_rhos, save_path, topk=30):
    out = sorted(abs_rhos, key=lambda t: t[1], reverse=True)[: min(topk, len(abs_rhos))]
    n = len(out)
    if n == 0:
        return

    can = R.TCanvas("cCorr", "cCorr", 900, 100 + 25 * n)
    R.gStyle.SetPaintTextFormat(".2f")

    nx = 2  # critical: not 1
    h = R.TH2F("h", ";|#rho(score, feature)|;Feature", nx, 0.0, 1.0, n, 0, n)
    for i, (f, absv, _) in enumerate(out, start=1):
        h.SetBinContent(2, i, absv)  # fill the second x-bin
        h.GetYaxis().SetBinLabel(i, f)

    h.SetMinimum(0.0)
    h.SetMaximum(1.0)
    h.Draw("COLZ")  # 1st pass: color
    h.Draw("TEXT0 SAME")  # 2nd pass: text (separate call avoids painter crash)
    can.Update()
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

def plot_score_feature_corr_ROOT_bar(
    abs_rhos, save_path, topk=30, title="|#rho(score, feature)|"
):
    """
    abs_rhos: list of (feature, |rho|, rho) already sorted (desc by |rho|) or unsorted.
    """
    out = sorted(abs_rhos, key=lambda t: t[1], reverse=True)[: min(topk, len(abs_rhos))]
    n = len(out)
    if n == 0:
        return

    can = R.TCanvas("cCorrBar", "cCorrBar", 900, 100 + 28 * n)
    R.gStyle.SetPadLeftMargin(0.38)
    R.gStyle.SetPadRightMargin(0.08)

    # Horizontal bars via TH2F (n rows, 2 columns), fill the right column with |rho|
    h = R.TH2F("hCorrBar", f";{title};Feature", 100, 0.0, 1.0, n, 0, n)
    for i, (f, absv, _) in enumerate(out, start=1):
        h.SetBinContent(h.GetXaxis().FindBin(absv), i, 1.0)  # just to drive "box" fill
        h.GetYaxis().SetBinLabel(i, f)

    # draw axes
    ax = R.TH2F("frame", "", 10, 0.0, 1.0, n, 0, n)
    ax.GetXaxis().SetTitle(title)
    for i, (f, _, _) in enumerate(out, start=1):
        ax.GetYaxis().SetBinLabel(i, f)
    ax.Draw()

    # draw bars with TBox to avoid painter quirks
    boxes = []
    for i, (f, absv, _) in enumerate(out, start=1):
        y1 = i - 1
        y2 = i
        b = R.TBox(0.0, y1 + 0.15, absv, y2 - 0.15)
        b.SetFillColor(R.kAzure + 1)
        b.SetLineColor(R.kAzure + 1)
        b.Draw("same")
        boxes.append(b)

    can.Update()
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

def _roc_weighted(probs, labels, weights, n=300):
    thr = np.linspace(0, 1, n)
    tpr = np.zeros(n)
    fpr = np.zeros(n)
    y = np.asarray(labels)
    p = np.asarray(probs)
    w = np.asarray(weights)
    for i, t in enumerate(thr):
        sel = p >= t
        TP = np.sum(w[sel & (y == 1)])
        FN = np.sum(w[~sel & (y == 1)])
        FP = np.sum(w[sel & (y == 0)])
        TN = np.sum(w[~sel & (y == 0)])
        tpr[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr[i] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return thr, tpr, fpr


def _mk_hist(arr, w, nb=30):
    h = R.TH1F(R.TUUID().AsString(), "", nb, 0.0, 1.0)
    for a, ww in zip(arr, w):
        h.Fill(float(a), float(ww))
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineWidth(2)
    return h


def plot_score_shapes_and_roc_by_category_ROOT(
    probs, labels, weights, categories, save_prefix, nbins=30
):
    probs = np.asarray(probs)
    y = np.asarray(labels)
    w = np.asarray(weights)
    cats = np.asarray(categories)
    for cat in np.unique(cats):
        sel = cats == cat
        if np.sum(sel) == 0:
            continue

        # shapes
        can1 = R.TCanvas("cSh", "cSh", 800, 650)
        h_sig = _mk_hist(probs[sel & (y == 1)], w[sel & (y == 1)], nb=nbins)
        h_sig.SetLineColor(R.kBlue)
        h_bkg = _mk_hist(probs[sel & (y == 0)], w[sel & (y == 0)], nb=nbins)
        h_bkg.SetLineColor(R.kRed)
        frame = R.TH1F("f", ";DNN score;Normalized entries", nbins, 0, 1)
        frame.SetMaximum(1.1 * max(h_sig.GetMaximum(), h_bkg.GetMaximum()))
        frame.Draw()
        h_sig.Draw("hist same")
        h_bkg.Draw("hist same")
        leg = R.TLegend(0.60, 0.75, 0.88, 0.88)
        leg.AddEntry(h_sig, "Signal", "l")
        leg.AddEntry(h_bkg, "Background", "l")
        leg.SetHeader(str(cat))
        leg.Draw()
        can1.SaveAs(f"{save_prefix}_shapes_{cat}.pdf")

        # ROC
        thr, tpr, fpr = _roc_weighted(probs[sel], y[sel], w[sel], n=300)
        can2 = R.TCanvas("cROC", "cROC", 700, 650)
        frame2 = R.TH2F("f2", ";FPR;TPR", 10, 0, 1, 10, 0, 1)
        frame2.SetStats(0)
        frame2.Draw()
        g = R.TGraph(len(fpr), np.array(fpr, dtype="f"), np.array(tpr, dtype="f"))
        g.SetLineWidth(2)
        g.SetLineColor(R.kBlack)
        g.Draw("L same")
        diag = R.TLine(0, 0, 1, 1)
        diag.SetLineStyle(2)
        diag.Draw()
        lab = R.TLatex()
        lab.SetNDC()
        lab.DrawLatex(0.18, 0.92, f"Category: {cat}")
        can2.SaveAs(f"{save_prefix}_roc_{cat}.pdf")
        can1.Close(); can2.Close()
        del can1; del can2
        R.gROOT.GetListOfCanvases().Clear()
        R.gROOT.GetListOfSpecials().Clear()
        R.gDirectory.GetList().Clear()

def plot_cumulative_SSB_per_process_ROOT(
    probs, labels, weights, processes, signal_processes, save_path, n=400
):
    probs = np.asarray(probs)
    y = np.asarray(labels)
    w = np.asarray(weights)
    proc = np.asarray(processes)
    thr = np.linspace(0, 1, n)

    # curves for ALL signal, and (optionally) each signal subprocess
    curves = {}

    def ssb_over_thr(sig_mask):
        SSB = np.zeros(n)
        for i, t in enumerate(thr):
            sel = probs >= t
            S = np.sum(w[sel & sig_mask])
            B = np.sum(w[sel & ~sig_mask])
            SSB[i] = S / np.sqrt(B) if B > 0 else 0.0
        return SSB

    # all-signal mask: either by 'labels' or by `process in signal_processes`
    sig_mask_all = np.isin(proc, signal_processes)
    curves["ALL signal"] = ssb_over_thr(sig_mask_all)

    # optional: individual signal processes
    for sp in signal_processes:
        curves[sp] = ssb_over_thr(proc == sp)

    # plot
    can = R.TCanvas("cSSB", "cSSB", 820, 650)
    frame = R.TH2F(
        "f",
        ";Score cut (>=); S / #sqrt{B}",
        10,
        0,
        1,
        10,
        0,
        max(1.0, max([np.max(v) for v in curves.values()])),
    )
    frame.Draw()

    colors = [
        R.kBlack,
        R.kBlue,
        R.kRed,
        R.kGreen,
        R.kMagenta,
        R.kOrange,
    ]
    leg = R.TLegend(0.58, 0.68, 0.88, 0.88)
    for (name, yv), col in zip(curves.items(), colors):
        g = R.TGraph(len(thr), np.array(thr, dtype="f"), np.array(yv, dtype="f"))
        g.SetLineWidth(2)
        g.SetLineColor(col)
        g.Draw("L same")
        leg.AddEntry(g, name, "l")

    # mark best working point for ALL
    y_all = curves["ALL signal"]
    imax = int(np.argmax(y_all))
    best_t = float(thr[imax])
    best_v = float(y_all[imax])
    l1 = R.TLine(best_t, 0, best_t, best_v)
    l1.SetLineStyle(2)
    l1.Draw()
    mark = R.TMarker(best_t, best_v, 20)
    mark.SetMarkerSize(1.1)
    mark.Draw()
    txt = R.TPaveText(0.15, 0.78, 0.55, 0.88, "NDC")
    txt.SetFillColor(0)
    txt.SetBorderSize(0)
    txt.AddText(f"Best ALL: cut={best_t:.3f}, S/#sqrt{{B}}={best_v:.2f}")
    txt.Draw()

    leg.Draw()
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

### Permutation importance

@torch.no_grad()
def _predict_probs(model, X, device):
    model.eval()
    tt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(tt) if getattr(model, "returns_logits", False) else model(tt)
    # if your forward returns logits, set model.returns_logits=True and uncomment next line:
    # probs = torch.sigmoid(logits).cpu().numpy().ravel()
    probs = logits.detach().cpu().numpy().ravel()
    return probs


def permutation_importance_auc(
    model, df, features, labels, weights, device, n_repeats=1, subsample=None
):
    """Returns list of (feat, dAUC, AUC_perm_mean). Uses weighted AUC."""
    X = df[features].values
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)

    if subsample is not None and subsample < len(df):
        idx = np.random.choice(len(df), subsample, replace=False)
        X, y, w = X[idx], y[idx], w[idx]

    p_base = _predict_probs(model, X, device)
    auc_base = safe_weighted_auc(y, p_base, sample_weight=w)

    results = []
    rng = np.random.default_rng()
    for f in features:
        j = features.index(f)
        auc_perm_l = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])  # permute one column
            pp = _predict_probs(model, Xp, device)
            aucp = safe_weighted_auc(y, pp, sample_weight=w)
            auc_perm_l.append(aucp)
        auc_perm = float(np.mean(auc_perm_l))
        dAUC = float(auc_base - auc_perm)
        results.append((f, dAUC, auc_perm))
    # sort by dAUC desc
    results.sort(key=lambda t: t[1], reverse=True)
    return auc_base, results


def plot_perm_importance_ROOT(
    results, save_path, topk=20, title="Permutation importance (#Delta AUC)"
):
    top = results[: min(topk, len(results))]
    n = len(top)
    can = R.TCanvas("cPI", "cPI", 900, 100 + 25 * n)
    h = R.TH2F(
        "h",
        f";#Delta AUC;Feature",
        1,
        0.0,
        max(1e-4, max(r[1] for r in top) * 1.15),
        n,
        0,
        n,
    )
    for i, (f, dAUC, _) in enumerate(top, start=1):
        h.SetBinContent(1, i, dAUC)
        h.GetYaxis().SetBinLabel(i, f)
    h.GetXaxis().SetBinLabel(1, "#Delta AUC")
    h.SetTitle(title)
    h.Draw("COLZ TEXT")
    can.SetLeftMargin(0.33)
    can.SetRightMargin(0.12)
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

### Partial dependence (PDP) for key variables
def partial_dependence_curve(
    model,
    df,
    features,
    target_feature,
    weights,
    device,
    grid="quantile",
    nbins=15,
    subsample=20000,
):
    """Returns grid_vals, pdp_vals (weighted mean score at each grid value)."""
    Xfull = df[features].values
    w = np.asarray(weights, dtype=float)

    if subsample and subsample < len(df):
        idx = np.random.choice(len(df), subsample, replace=False)
        Xfull, w = Xfull[idx], w[idx]
    j = features.index(target_feature)
    xj = Xfull[:, j]

    if grid == "quantile":
        grid_vals = np.unique(np.quantile(xj, np.linspace(0.02, 0.98, nbins)))
    else:
        lo, hi = np.min(xj), np.max(xj)
        grid_vals = np.linspace(lo, hi, nbins)

    pdp_vals = []
    for v in grid_vals:
        X = Xfull.copy()
        X[:, j] = v
        p = _predict_probs(model, X, device)
        pdp_vals.append(np.average(p, weights=w))
    return grid_vals, np.array(pdp_vals)


def plot_pdp_ROOT(grid, pdp, xlabel, save_path, ylabel="Mean DNN score"):
    can = R.TCanvas("cPDP", "cPDP", 750, 600)
    frame = R.TH2F(
        "f",
        "",
        10,
        float(grid.min()),
        float(grid.max()),
        10,
        0.0,
        max(0.001, 1.05 * float(pdp.max())),
    )
    frame.GetXaxis().SetTitle(xlabel)
    frame.GetYaxis().SetTitle(ylabel)
    frame.Draw()
    g = R.TGraph(len(grid), np.array(grid, dtype="f"), np.array(pdp, dtype="f"))
    g.SetLineWidth(2)
    g.Draw("L same")
    can.SaveAs(save_path)
    can.Close()
    del can
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

### Weight distribution (linear + log)
def plot_weight_distribution_ROOT(weights, save_prefix, nbins=60):
    w = np.asarray(weights, dtype=float)

    # linear
    c1 = R.TCanvas("cWlin","cWlin",750,600);
    h1 = R.TH1F("hWlin",";Event weight;Events", nbins, float(np.min(w)), float(np.max(w)))
    for ww in w: h1.Fill(float(ww))
    h1.Draw("hist")
    c1.SaveAs(f"{save_prefix}_weights_linear.pdf")

    # logy (with safe x-range)
    c2 = R.TCanvas("cWlog","cWlog",750,600); c2.SetLogy(1)
    xmin = np.percentile(w, 0.1); xmax = np.percentile(w, 99.9)
    h2 = R.TH1F("hWlog",";Event weight;Events", nbins, float(xmin), float(xmax))
    for ww in w:
        if xmin <= ww <= xmax: h2.Fill(float(ww))
    h2.Draw("hist")
    c2.SaveAs(f"{save_prefix}_weights_log.pdf")
    c1.Close()
    c2.Close()
    del c1
    del c2
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()

### Yield tabel after cut
def yield_table_after_cut(probs, labels, weights, processes, score_cut, save_prefix):
    p = np.asarray(probs)
    y = np.asarray(labels)
    w = np.asarray(weights)
    proc = np.asarray(processes)

    sel = p >= score_cut
    rows = []
    procs = list(np.unique(proc))
    for pr in procs:
        m = sel & (proc == pr)
        Y = float(np.sum(w[m]))
        S2 = float(np.sum(w[m] ** 2))
        err = np.sqrt(S2)
        rows.append((pr, Y, err))

    # totals
    m_sig = sel & (y == 1)
    m_bkg = sel & (y == 0)
    S = float(np.sum(w[m_sig]))
    S_err = np.sqrt(float(np.sum(w[m_sig] ** 2)))
    B = float(np.sum(w[m_bkg]))
    B_err = np.sqrt(float(np.sum(w[m_bkg] ** 2)))
    ssb = S / np.sqrt(B) if B > 0 else 0.0

    # text table
    lines = ["Process, Yield, StatErr"]
    lines += [f"{pr}, {Y:.6g}, {err:.6g}" for pr, Y, err in rows]
    lines += [
        f"TOTAL_SIG, {S:.6g}, {S_err:.6g}",
        f"TOTAL_BKG, {B:.6g}, {B_err:.6g}",
        f"S/sqrt(B) @ cut={score_cut:.3f}: {ssb:.4f}",
    ]
    txtpath = f"{save_prefix}_yields_cut_{score_cut:.3f}.txt"
    with open(txtpath, "w") as f:
        f.write("\n".join(lines))

    # CSV
    csvpath = f"{save_prefix}_yields_cut_{score_cut:.3f}.csv"
    with open(csvpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Process", "Yield", "StatErr"])
        for pr, Y, err in rows:
            writer.writerow([pr, Y, err])
        writer.writerow(["TOTAL_SIG", S, S_err])
        writer.writerow(["TOTAL_BKG", B, B_err])
        writer.writerow([f"S/sqrt(B)@{score_cut:.3f}", ssb, ""])
    return {
        "S": S,
        "B": B,
        "S_err": S_err,
        "B_err": B_err,
        "ssb": ssb,
        "txt": txtpath,
        "csv": csvpath,
    }

## Save per-fold artifacts
# --- Save per-fold CV artifacts for later aggregation ---
# Weighted ROC (reuse your earlier helper or this minimal one)
def _roc_weighted(probs, labels, weights, n=300):
    thr = np.linspace(0,1,n)
    tpr = np.zeros(n); fpr = np.zeros(n)
    y = np.asarray(labels); p = np.asarray(probs); w = np.asarray(weights)
    for i,t in enumerate(thr):
        sel = (p >= t)
        TP = np.sum(w[ sel & (y==1)]); FN = np.sum(w[~sel & (y==1)])
        FP = np.sum(w[ sel & (y==0)]); TN = np.sum(w[~sel & (y==0)])
        tpr[i] = TP/(TP+FN) if (TP+FN)>0 else 0.0
        fpr[i] = FP/(FP+TN) if (FP+TN)>0 else 0.0
    return fpr, tpr


### One shot after all folds: CV consistency plots

def _mk_hist_score(pred, lab, w, which, nb=30):
    """which='sig' or 'bkg'"""
    sel = (lab == 1) if which == "sig" else (lab == 0)
    h = R.TH1F(R.TUUID().AsString(), "", nb, 0.0, 1.0)
    for p, ww in zip(pred[sel], w[sel]):
        h.Fill(float(p), float(ww))
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineWidth(2)
    return h


def cv_consistency_plots_ROOT(parent_save_path, fold_dirs, nbins=30):
    """
    fold_dirs: list like [f"{save_path}/fold0", ...]
    Produces:
      - CV_ROC_overlay.pdf
      - CV_AUC_summary.pdf
      - CV_ScoreShapes_signal.pdf
      - CV_ScoreShapes_background.pdf
    """
    # ---------- load ----------
    aucs = []
    roc_graphs = []
    sig_hists, bkg_hists = [], []
    colors = [
        R.kBlue + 1,
        R.kRed + 1,
        R.kGreen + 2,
        R.kMagenta + 2,
        R.kOrange + 7,
        R.kCyan + 1,
    ]

    for i, fd in enumerate(fold_dirs):
        zpath = os.path.join(fd, "cv_artifacts.npz")
        if not os.path.exists(zpath):
            print(f"[cv] missing {zpath}, skipping")
            continue
        data = np.load(zpath, allow_pickle=True)
        aucs.append(float(data["auc"]))
        fpr, tpr = data["fpr"], data["tpr"]
        pred, lab, w = data["pred"], data["label"], data["weight"]

        # ROC graph
        g = R.TGraph(len(fpr), np.array(fpr, dtype="f"), np.array(tpr, dtype="f"))
        g.SetLineColor(colors[i % len(colors)])
        g.SetLineWidth(2)
        roc_graphs.append(("fold{}".format(i), g))

        # score shapes
        hs = _mk_hist_score(pred, lab, w, "sig", nb=nbins)
        hb = _mk_hist_score(pred, lab, w, "bkg", nb=nbins)
        hs.SetLineColor(colors[i % len(colors)])
        hb.SetLineColor(colors[i % len(colors)])
        sig_hists.append(("fold{}".format(i), hs))
        bkg_hists.append(("fold{}".format(i), hb))

    # ---------- ROC overlay ----------
    can = R.TCanvas("cROCcv", "cROCcv", 750, 650)
    fr = R.TH2F("fr", ";FPR;TPR", 10, 0, 1, 10, 0, 1)
    fr.Draw()
    diag = R.TLine(0, 0, 1, 1)
    diag.SetLineStyle(2)
    diag.Draw()
    leg = R.TLegend(0.62, 0.38, 0.88, 0.58)
    for name, g in roc_graphs:
        g.Draw("L same")
        leg.AddEntry(g, name, "l")
    leg.Draw()
    can.SaveAs(os.path.join(parent_save_path, "CV_ROC_overlay.pdf"))

    # ---------- AUC summary (bars + mean±RMS) ----------
    if len(aucs):
        canA = R.TCanvas("cAUC", "cAUC", 760, 620)
        n = len(aucs)
        h = R.TH1F("hA", ";fold;AUC", n, 0, n)
        for i, a in enumerate(aucs, start=1):
            h.SetBinContent(i, a)
        h.SetBarWidth(0.8)
        h.SetFillColor(R.kAzure + 1)
        h.Draw("bar2")
        mean = float(np.mean(aucs))
        rms = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)
        line = R.TLine(0, mean, n, mean)
        line.SetLineColor(R.kRed + 1)
        line.SetLineWidth(2)
        line.Draw()
        box = R.TPaveText(0.62, 0.78, 0.88, 0.88, "NDC")
        box.SetFillColor(0)
        box.SetBorderSize(0)
        box.AddText(f"mean AUC = {mean:.4f}")
        box.AddText(f"RMS = {rms:.4f}")
        box.Draw()
        canA.SaveAs(os.path.join(parent_save_path, "CV_AUC_summary.pdf"))

    # ---------- Score shapes overlays ----------
    # Signal
    canS = R.TCanvas("cSsig", "cSsig", 750, 650)
    frS = R.TH1F("frS", ";DNN score;Normalized entries", nbins, 0, 1)
    frS.SetMaximum(
        max([h.GetMaximum() for _, h in sig_hists] + [0]) * 1.15 if sig_hists else 1.0
    )
    frS.Draw()
    legS = R.TLegend(0.22, 0.68, 0.48, 0.88)
    for name, h in sig_hists:
        h.Draw("hist same")
        legS.AddEntry(h, name, "l")
    legS.SetHeader("Signal")
    legS.Draw()
    canS.SaveAs(os.path.join(parent_save_path, "CV_ScoreShapes_signal.pdf"))

    # Background
    canB = R.TCanvas("cSbkg", "cSbkg", 750, 650)
    frB = R.TH1F("frB", ";DNN score;Normalized entries", nbins, 0, 1)
    frB.SetMaximum(
        max([h.GetMaximum() for _, h in bkg_hists] + [0]) * 1.15 if bkg_hists else 1.0
    )
    frB.Draw()
    legB = R.TLegend(0.62, 0.68, 0.88, 0.88)
    for name, h in bkg_hists:
        h.Draw("hist same")
        legB.AddEntry(h, name, "l")
    legB.SetHeader("Background")
    legB.Draw()
    canB.SaveAs(os.path.join(parent_save_path, "CV_ScoreShapes_background.pdf"))
    canA.Close()
    canB.Close()
    canS.Close()
    del canA
    del canB
    del canS
    R.gROOT.GetListOfCanvases().Clear()
    R.gROOT.GetListOfSpecials().Clear()
    R.gDirectory.GetList().Clear()
