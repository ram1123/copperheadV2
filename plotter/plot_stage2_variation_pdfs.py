"""Plot each shape-systematic up/down variation against the nominal DNN score
template, for every sample/year/region found under a Stage-2 histogram
directory, and save the comparisons as PDFs (one multi-page PDF per
year/sample, one page per region + systematic).

Run from the repository root with:

    pixi run -e default python \
        scripts/plot_stage2_variation_pdfs.py \
        --hist-path /path/to/stage2_histograms/score_<label> \
        --outdir plots/stage2_variation_validation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_vbf_hist_validation import (  # noqa: E402
    RUN2_YEARS,
    extract_arrays,
    load_histogram,
    parse_years,
    score_axis_name,
    variation_names,
)
# save_postfix = "Jul14_2026_100nTrialsFoldsAll_Max70bins_systFix" 
save_postfix = "Jul15_2026_100nTrialsFoldsAll_Max70bins_nominal_dnn_features_for_systs" 
DEFAULT_HIST_PATH = Path(
    "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
    "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo/"
    "stage2_histograms/"
    "score_Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo_"
    f"{save_postfix}"
)
DEFAULT_OUTDIR = Path(f"plots/stage2_variation_validation/{save_postfix}")
DEFAULT_REGIONS = ("h-peak", "h-sidebands")
CHANNEL = "vbf"


def fold_flow(values, n_regular_bins):
    """Fold underflow/overflow entries (if present) into the first/last bin."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == n_regular_bins:
        return values
    folded = values[1:-1].copy()
    folded[0] += values[0]
    folded[-1] += values[-1]
    return folded


def discover_systematics(available_labels):
    """Return sorted base systematic names with an '_up' and/or '_down' entry."""
    bases = set()
    for label in available_labels - {"nominal"}:
        if label.endswith("_up"):
            bases.add(label[: -len("_up")])
        elif label.endswith("_down"):
            bases.add(label[: -len("_down")])
    return sorted(bases)


def draw_comparison(ax_main, ax_ratio, edges, nominal, up, down, syst_name):
    ax_main.stairs(
        nominal, edges, label=f"nominal (\N{GREEK CAPITAL LETTER SIGMA}={nominal.sum():.3g})",
        color="black", linewidth=1.6,
    )
    if up is not None:
        ax_main.stairs(
            up, edges, label=f"{syst_name} up (\N{GREEK CAPITAL LETTER SIGMA}={up.sum():.3g})",
            color="crimson", linewidth=1.3,
        )
    if down is not None:
        ax_main.stairs(
            down, edges, label=f"{syst_name} down (\N{GREEK CAPITAL LETTER SIGMA}={down.sum():.3g})",
            color="royalblue", linewidth=1.3,
        )
    ax_main.set_ylabel("Weighted events")
    if np.any(nominal > 0):
        ax_main.set_yscale("log")
    ax_main.legend(fontsize=8, loc="best")

    nominal_safe = np.where(nominal != 0, nominal, np.nan)
    ax_ratio.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    if up is not None:
        ax_ratio.stairs(up / nominal_safe, edges, color="crimson", linewidth=1.3)
    if down is not None:
        ax_ratio.stairs(down / nominal_safe, edges, color="royalblue", linewidth=1.3)
    ax_ratio.set_ylabel("var / nominal")
    ax_ratio.set_xlabel("Transformed DNN score")
    ax_ratio.set_ylim(0.5, 1.5)


def plot_sample_year(histogram, sample, year, pdf, include_flow, regions):
    score_name = score_axis_name(histogram)
    available_labels = variation_names(histogram)
    systematics = discover_systematics(available_labels)
    if not systematics:
        return 0

    n_pages = 0
    for region in regions:
        selection = {"region": region, "channel": CHANNEL}
        try:
            edges, nominal_values, _ = extract_arrays(
                histogram, "nominal", selection, include_flow
            )
        except Exception as exc:
            print(f"  [skip] {sample} {year} {region}: could not extract nominal ({exc})")
            continue

        n_regular_bins = len(edges) - 1
        nominal_folded = fold_flow(nominal_values, n_regular_bins)
        if not np.any(nominal_folded > 0):
            print(f"  [skip] {sample} {year} {region}: nominal histogram is empty")
            continue

        for syst in systematics:
            up_label, down_label = f"{syst}_up", f"{syst}_down"
            has_up = up_label in available_labels
            has_down = down_label in available_labels
            if not has_up and not has_down:
                continue

            up_folded = down_folded = None
            if has_up:
                _, up_values, _ = extract_arrays(histogram, up_label, selection, include_flow)
                up_folded = fold_flow(up_values, n_regular_bins)
            if has_down:
                _, down_values, _ = extract_arrays(histogram, down_label, selection, include_flow)
                down_folded = fold_flow(down_values, n_regular_bins)

            fig, (ax_main, ax_ratio) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(7, 6),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
                constrained_layout=True,
            )
            draw_comparison(ax_main, ax_ratio, edges, nominal_folded, up_folded, down_folded, syst)
            fig.suptitle(f"{sample} | {year} | region={region}", fontsize=11)
            fig.text(0.01, 0.005, f"score axis: {score_name}", fontsize=6, color="gray")
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
    return n_pages


def main(hist_path, outdir, years, samples, include_flow, regions):
    outdir.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    for year in years:
        year_dir = hist_path / year
        if not year_dir.is_dir():
            print(f"Skipping {year}: directory not found at {year_dir}")
            continue

        sample_list = samples or sorted(
            p.name[: -len("_hist.pkl")] for p in year_dir.glob("*_hist.pkl")
        )
        for sample in sample_list:
            pkl_path = year_dir / f"{sample}_hist.pkl"
            if not pkl_path.is_file():
                print(f"Skipping {sample} ({year}): no histogram at {pkl_path}")
                continue

            histogram = load_histogram(pkl_path)
            out_path = outdir / year / f"{sample}_variation_validation.pdf"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(out_path) as pdf:
                n_pages = plot_sample_year(histogram, sample, year, pdf, include_flow, regions)

            if n_pages:
                print(f"Wrote {n_pages} page(s) -> {out_path}")
                total_pages += n_pages
            else:
                out_path.unlink(missing_ok=True)
                print(f"No shape systematics to plot for {sample} ({year}); skipped {out_path}")

    print(f"\nDone. {total_pages} comparison page(s) written under {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot each up/down shape-systematic variation against the nominal "
            "DNN score template for Stage-2 histograms, saved as one "
            "multi-page PDF per (year, sample)."
        )
    )
    parser.add_argument(
        "--hist-path",
        type=Path,
        default=DEFAULT_HIST_PATH,
        help=(
            "Base directory containing Stage-2 histograms, with one "
            f"subdirectory per year (default: {DEFAULT_HIST_PATH})."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory to write PDF files under (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--years",
        type=parse_years,
        default=RUN2_YEARS,
        metavar="YEAR[,YEAR...]",
        help=f"Comma-separated Run-2 years to plot (default: {','.join(RUN2_YEARS)}).",
    )
    parser.add_argument(
        "--samples",
        default=None,
        metavar="SAMPLE[,SAMPLE...]",
        help="Comma-separated sample names to plot (default: every *_hist.pkl found per year).",
    )
    parser.add_argument(
        "--regions",
        default=",".join(DEFAULT_REGIONS),
        metavar="REGION[,REGION...]",
        help=f"Comma-separated regions to plot (default: {','.join(DEFAULT_REGIONS)}).",
    )
    parser.add_argument(
        "--flow",
        dest="include_flow",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fold underflow/overflow bins into the first/last visible bin (default: True).",
    )
    args = parser.parse_args()
    samples = [s.strip() for s in args.samples.split(",")] if args.samples else None
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    main(args.hist_path, args.outdir, args.years, samples, args.include_flow, regions)
