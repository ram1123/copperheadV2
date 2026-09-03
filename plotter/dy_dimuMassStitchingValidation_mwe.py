"""Minimal working example: DY stitching validation for 2017.

Stacks the dimuon mass spectrum of the mass-binned `dyTo2Mu_M-100to200_MiNNLO`
sample on top of the inclusive `dyTo2Mu_M-50_MiNNLO` sample, read from the
stage1 compacted parquets, so the overlap region (100 < m(mumu) < 200 GeV) can
be inspected for the usual stitching double counting / gaps.

Both samples are drawn with the plotting package used by run_plotter.py
(src/lib/histogram/plotting.py): as a filled stack (M-50 at the bottom,
M-100to200 on top) with the stack total overlaid in error-bar mode, i.e. as
points carrying the combined sum-w2 statistical uncertainty.

Two figures are produced per entry in PLOT_RANGES (full 50-500 GeV spectrum,
plus a 70-150 GeV zoom around the Z peak and the stitching boundary): a linear
one under the configured filename and a log-scale one with a `_log` suffix. The
parquets are read once and re-histogrammed per range.

Run from the repository root:
    python plotter/dy_dimuMassStitchingValidation_mwe.py
"""

import logging
import sys
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml
from matplotlib.colors import ListedColormap

# Allow `python plotter/dy_dimuMassStitchingValidation_mwe.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils import logger
from src.lib.histogram.plotting import getHistAndErrs

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
YEAR = "2017"
LOAD_PATH = Path(
    "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
    "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo/stage1_output/"
    f"{YEAR}/compacted"
)

# label -> sample directory name under LOAD_PATH.
# Stacking order follows this dict: the first entry is the bottom of the stack,
# so M-100to200 ends up stacked on top of M-50.
SAMPLES = {
    "DY M-50 MiNNLO": "dyTo2Mu_M-50_MiNNLO",
    "DY M-100to200 MiNNLO": "dyTo2Mu_M-100to200_MiNNLO",
}

VAR = "dimuon_mass"
WGT = "wgt_nominal"

SAVE_DIR = Path("validation/figs/dy_diMuMassStitching")

# (xmin, xmax, nbins, output filename) -- each entry gives a linear plot under
# this filename plus a log-scale one with a `_log` suffix
# nbins is chosen per range so the bin edges stay on a common grid rather than
# always being 100: 4.5 GeV bins over 50-500, 0.8 GeV over 70-150, and 0.5 GeV
# over the 95-115 zoom -- 0.5 GeV divides the 4.5 GeV bins exactly, so every
# edge of the 95-115 plot (95.0, 95.5, ..., 115.0) lies on the full-range grid,
# with 95.0, 99.5, 104.0, 108.5 and 113.0 being shared bin edges.
PLOT_RANGES = [
    (50.0, 500.0, 100, f"dy_stitchingValidation_{YEAR}.pdf"),
    (70.0, 150.0, 100, f"dy_stitchingValidation_{YEAR}_70to150.pdf"),
    (95.0, 115.0, 40, f"dy_stitchingValidation_{YEAR}_95to115.pdf"),
    # finer 0.1 GeV version of the same zoom; 0.1 still divides both 0.5 and
    # 4.5 GeV, so these edges remain a subdivision of the two plots above
    (95.0, 115.0, 200, f"dy_stitchingValidation_{YEAR}_95to115_nbins200.pdf"),
]

STATUS = "Preliminary"
CM_ENERGY = 13


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_lumi(year: str) -> float:
    """Integrated luminosity in fb-1, same source as validation_plotter_unified.py."""
    with open(Path("configs") / "parameters" / "lumi.yaml", "r") as f:
        lumi_config = yaml.safe_load(f)
    lumi = lumi_config.get("integrated_lumis", {}).get(year, 0.0)
    if lumi == 0.0:
        raise ValueError(f"lumi for year {year} is not defined!")
    return round(lumi / 1000.0, 1)  # pb-1 -> fb-1


def load_sample(sample_dir: Path) -> tuple:
    """Read (values, weights) for VAR/WGT out of the compacted parquets.

    Only the two needed columns are read, so the on-disk size of the sample is
    irrelevant -- parquet prunes everything else.
    """
    ddf = dd.read_parquet(str(sample_dir / "*" / "*.parquet"), columns=[VAR, WGT])
    df = ddf.compute()
    values = df[VAR].to_numpy(dtype=np.float64)
    weights = df[WGT].to_numpy(dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights)
    if (~finite).any():
        logger.warning(f"{sample_dir.name}: dropping {(~finite).sum()} non-finite entries")
    return values[finite], weights[finite]


def plot_stack(samples_arrs: dict, xmin: float, xmax: float, nbins: int, save_path: Path):
    """Stacked m(mumu) plots for one range, with the stack total in error-bar mode.

    Histograms are filled once and drawn twice: linear scale under `save_path`,
    log scale under the same name with a `_log` suffix -- the same naming
    convention plotDataMC_compare uses via run_plotter.py.
    """
    binning = np.linspace(xmin, xmax, nbins + 1)

    labels, hist_l, hist_w2_l = [], [], []
    for label, (values, weights) in samples_arrs.items():
        np_hist, np_hist_err = getHistAndErrs(binning, values, weights)
        labels.append(label)
        hist_l.append(np_hist)
        hist_w2_l.append(np_hist_err**2)
        logger.info(f"  {label}: yield in [{xmin}, {xmax}] = {np_hist.sum():.2f}")

    stack_total = np.sum(np.asarray(hist_l), axis=0)
    stack_total_err = np.sqrt(np.sum(np.asarray(hist_w2_l), axis=0))
    logger.info(f"  Stack total: yield in [{xmin}, {xmax}] = {stack_total.sum():.2f}")

    plt.style.use(hep.style.CMS)
    petroff10 = ListedColormap(
        ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
         "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    )
    colors = petroff10.colors

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_save_path = save_path.with_name(save_path.stem + "_log" + save_path.suffix)

    for log_scale, out_path in ((False, save_path), (True, log_save_path)):
        fig, ax_main = plt.subplots()

        # filled stack, in SAMPLES order (first entry at the bottom). sort=None keeps
        # that order instead of mplhep's default yield-based sorting.
        hep.histplot(
            hist_l,
            bins=binning,
            stack=True,
            histtype="fill",
            label=labels,
            sort=None,
            color=colors[: len(hist_l)],
            ax=ax_main,
        )
        # stack total in error-bar mode, carrying the combined stat. uncertainty
        hep.histplot(
            stack_total,
            bins=binning,
            xerr=True,
            yerr=stack_total_err,
            histtype="errorbar",
            color="black",
            label="Stack total (stat. unc.)",
            ax=ax_main,
        )

        ax_main.set_xlim(xmin, xmax)
        ax_main.set_xlabel(r"$m_{\mu\mu}$ (GeV)")
        ax_main.set_ylabel("Events")
        if log_scale:
            ax_main.set_yscale("log")
        ax_main.legend(loc="best")
        hep.cms.label(
            data=False, loc=0, text=STATUS, com=CM_ENERGY, lumi=get_lumi(YEAR), ax=ax_main
        )

        plt.savefig(out_path)
        plt.close(fig)
        logger.info(f"  Saved {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # read once, re-histogram per range
    samples_arrs = {}
    for label, sample_name in SAMPLES.items():
        sample_dir = LOAD_PATH / sample_name
        logger.info(f"Reading {sample_dir}")
        values, weights = load_sample(sample_dir)
        samples_arrs[label] = (values, weights)
        logger.info(f"{label}: {len(values)} raw events")

    for xmin, xmax, nbins, fname in PLOT_RANGES:
        logger.info(f"Plotting [{xmin}, {xmax}] GeV with {nbins} bins")
        plot_stack(samples_arrs, xmin, xmax, nbins, SAVE_DIR / fname)


if __name__ == "__main__":
    main()
