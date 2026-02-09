import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SUFFIX = ""  # "" or "_AfterImpute"  # optional suffix for output filenames

def plot_before_after_scaling(x, w, x_mean, x_std, feature_names, outdir, nshow=51):
    x_scaled = (x - x_mean) / x_std

    nshow = min(nshow, len(feature_names))
    ncols = 3
    nrows = (nshow + ncols - 1) // ncols

    # -----------------------------
    # 1) Linear scale figure
    # -----------------------------
    fig_lin, axes_lin = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes_lin = axes_lin.flatten()

    # -----------------------------
    # 2) Log scale figure
    # -----------------------------
    fig_log, axes_log = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes_log = axes_log.flatten()

    for i in range(nshow):
        bins_raw = 50
        bins_scaled = np.linspace(-5, 5, 60)

        # ---------- Linear ----------
        ax = axes_lin[i]
        ax.hist(
            x[:, i],
            bins=bins_raw,
            weights=w,
            density=True,
            histtype="step",
            label="raw",
            linewidth=1.5,
        )
        ax.hist(
            x_scaled[:, i],
            bins=bins_scaled,
            weights=w,
            density=True,
            histtype="step",
            label="scaled",
            linewidth=1.5,
        )
        ax.set_title(feature_names[i])
        ax.legend(fontsize=9)
        # add mean and std info
        mean_raw = np.average(x[:, i], weights=w)
        std_raw = np.sqrt(np.average((x[:, i] - mean_raw) ** 2, weights=w))
        mean_scaled = np.average(x_scaled[:, i], weights=w)
        std_scaled = np.sqrt(np.average((x_scaled[:, i] - mean_scaled) ** 2, weights=w))
        ax.text(0.95, 0.95,
                f"raw: μ={mean_raw:.2f}, σ={std_raw:.2f}\n"
                f"scaled: μ={mean_scaled:.2f}, σ={std_scaled:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right')

        # ---------- Log ----------
        ax = axes_log[i]
        ax.hist(
            x[:, i],
            bins=bins_raw,
            weights=w,
            density=True,
            histtype="step",
            label="raw",
            linewidth=1.5,
        )
        ax.hist(
            x_scaled[:, i],
            bins=bins_scaled,
            weights=w,
            density=True,
            histtype="step",
            label="scaled",
            linewidth=1.5,
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)  # avoid log(0)
        ax.set_title(feature_names[i])
        ax.legend(fontsize=9)
        # add mean and std info
        mean_raw = np.average(x[:, i], weights=w)
        std_raw = np.sqrt(np.average((x[:, i] - mean_raw) ** 2, weights=w))
        mean_scaled = np.average(x_scaled[:, i], weights=w)
        std_scaled = np.sqrt(np.average((x_scaled[:, i] - mean_scaled) ** 2, weights=w))
        ax.text(0.95, 0.95,
                f"raw: μ={mean_raw:.2f}, σ={std_raw:.2f}\n"
                f"scaled: μ={mean_scaled:.2f}, σ={std_scaled:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right')

    # Turn off unused pads
    for ax in axes_lin[nshow:]:
        ax.axis("off")
    for ax in axes_log[nshow:]:
        ax.axis("off")



    fig_lin.suptitle("Scaling validation (linear scale)", fontsize=16)
    fig_log.suptitle("Scaling validation (log scale)", fontsize=16)

    fig_lin.savefig(f"{outdir}/scaling_before_after_linear{SUFFIX}.pdf")
    fig_log.savefig(f"{outdir}/scaling_before_after_log{SUFFIX}.pdf")

    plt.close(fig_lin)
    plt.close(fig_log)


def plot_scaled_mean_std(x_scaled, w, feature_names, outdir):
    """Sanity check: Check that mean shoudl be around ZERO and
    standard deviation should be around 1. If std is ZERO that means
    its a constant features.

    Args:
        x_scaled (_type_): _description_
        w (_type_): _description_
        feature_names (_type_): _description_
        outdir (_type_): _description_
    """
    mu = np.average(x_scaled, axis=0, weights=w)
    var = np.average((x_scaled - mu) ** 2, axis=0, weights=w)
    sigma = np.sqrt(var)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(mu, "o")
    ax1.axhline(0, color="k", ls="--")
    ax1.set_ylabel("Weighted mean")

    ax2.plot(sigma, "o")
    ax2.axhline(1, color="k", ls="--")
    ax2.set_ylabel("Weighted std")
    ax2.set_xlabel("Feature index")

    fig.suptitle("Scaled feature statistics", fontsize=14)
    fig.savefig(f"{outdir}/scaling_mean_std{SUFFIX}.pdf")
    plt.close(fig)


def plot_corr_before_after(x, x_scaled, outdir):
    """Scaling should not change the correlation between features.

    Correlation should look identical before and after scaling. If not,
    that implies a bug in the scaling code.

    Args:
        x (_type_): _description_
        x_scaled (_type_): _description_
        outdir (_type_): _description_
    """
    corr_raw = np.corrcoef(x.T)
    corr_scaled = np.corrcoef(x_scaled.T)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_raw, vmin=-1, vmax=1, cmap="coolwarm", ax=axs[0])
    axs[0].set_title("Raw features")

    sns.heatmap(corr_scaled, vmin=-1, vmax=1, cmap="coolwarm", ax=axs[1])
    axs[1].set_title("Scaled features")

    fig.savefig(f"{outdir}/scaling_corr_check{SUFFIX}.pdf")
    plt.close(fig)


def plot_scaled_outliers(x_scaled, feature_names, outdir):
    """Check for outliers in scaled features.
    If max |z| > 10, that indicates a potential outlier in that feature.
    Args:
        x_scaled (_type_): _description_
        feature_names (_type_): _description_
        outdir (_type_): _description_
    """
    max_abs = np.max(np.abs(x_scaled), axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(max_abs, "o")
    ax.axhline(10, color="r", ls="--", label="|z| = 10")
    ax.set_ylabel("max |z|")
    ax.set_xlabel("Feature index")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(f"{outdir}/scaling_outliers{SUFFIX}.pdf")
    plt.close(fig)
