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
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from modules.sample_config import get_bkg_sig_dicts  # noqa: E402
from stage2_vbf_hist_validation import (  # noqa: E402
    RUN2_YEARS,
    extract_arrays,
    load_histogram,
    parse_years,
    score_axis_name,
    variation_names,
)
# save_postfix = "Jul14_2026_100nTrialsFoldsAll_Max70bins_systFix" 
save_postfix = "Jul23_2026_aiAgentSystemTest" 

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

#: Grey dotted guides drawn in the ratio panel alongside the solid y = 1 line.
#: Both sit inside the panel's fixed (0.5, 1.5) range.
RATIO_GUIDES = (0.75, 1.25)

#: Which MC samples are signal and which are background is defined once, in the
#: analysis sample config -- not duplicated here. A hardcoded list would silently
#: go stale the first time a process is added or swapped in the yaml.
DEFAULT_SAMPLES_YAML = Path(__file__).resolve().parent.parent / "configs/samples/samples.yaml"


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


def load_sample_roles(samples_yaml, year):
    """(signal, background) process-name sets for `year`, from samples.yaml.

    Resolution is delegated to modules.sample_config -- the same code Stage-2 and
    Stage-3 use -- rather than reimplemented here, so these plots can never
    disagree with the histograms they describe about which process is which.
    (An earlier version of this function did reimplement it, and got the
    per-year lookup wrong: samples.yaml writes numeric years unquoted, so they
    load as int, and a str lookup silently fell through to the group defaults,
    dropping both DY samples and EWK from the background total.
    `_resolve_group` normalises the keys and has no such hole.)
    """
    bkg, sig, _ = get_bkg_sig_dicts(samples_yaml, year=str(year))
    signal_names = {p for procs in sig.values() for p in procs}
    background_names = {p for procs in bkg.values() for p in procs}

    overlap = signal_names & background_names
    if overlap:
        raise ValueError(
            f"{samples_yaml}: {sorted(overlap)} appear as both signal and "
            f"background for year {year}; the totals would double count."
        )
    return signal_names, background_names


def classify_samples(sample_list, signal_names, background_names):
    """Split discovered sample names into (signal, background, unclassified).

    Membership comes from samples.yaml, so anything the config does not know
    about -- observed data, or a process someone added to Stage-2 without adding
    it to the config -- lands in `unclassified` and is reported rather than being
    quietly folded into a total.
    """
    signal, background, unclassified = [], [], []
    for name in sorted(sample_list):
        if name in signal_names:
            signal.append(name)
        elif name in background_names:
            background.append(name)
        else:
            unclassified.append(name)
    return signal, background, unclassified


def folded_arrays(histogram, label, selection, include_flow):
    """(edges, values) with underflow/overflow folded into the end bins."""
    edges, values, _ = extract_arrays(histogram, label, selection, include_flow)
    return edges, fold_flow(values, len(edges) - 1)


def sum_group(histograms, label, selection, include_flow):
    """Sum `label` across samples; fall back to each sample's nominal if absent.

    Returns ``(edges, summed, used, fell_back)``.

    A process that does not carry a given shape systematic keeps its nominal
    shape in the fit, so its nominal is what must enter the up/down sums --
    dropping it instead would make the total's ratio panel show a variation that
    no fit ever sees. `fell_back` records which samples that applied to, and the
    caller prints it on the page rather than leaving it implicit.
    """
    edges = total = None
    used, fell_back = [], []
    for name, histogram in histograms.items():
        labels = variation_names(histogram)
        use_label = label
        if label not in labels:
            if "nominal" not in labels:
                continue
            use_label = "nominal"
            fell_back.append(name)
        try:
            sample_edges, values = folded_arrays(
                histogram, use_label, selection, include_flow
            )
        except Exception as exc:                       # region absent for this sample
            print(f"  [skip] {name}: {label} in {selection} ({exc})")
            continue
        if edges is None:
            edges = sample_edges
        elif len(sample_edges) != len(edges) or not np.allclose(sample_edges, edges):
            raise ValueError(
                f"sample {name} has a different score binning from the rest of the "
                "group; the totals would be meaningless. Were these histograms "
                "produced with the same dnn_binning.yaml?"
            )
        total = values if total is None else total + values
        used.append(name)
    return edges, total, used, fell_back


def write_metadata(outdir, hist_path, samples_yaml, year, entry):
    """Record which MC samples went into each total, as
    `<outdir>/<year>/metadata.txt` -- in the year directory, beside the PDFs it
    describes, so a plot and its provenance never get separated.

    The on-page footer says the same thing, but a plain-text file next to the
    PDFs is what makes a plot directory self-describing months later: which
    processes the sums contain, which the sample config did not know about, and
    which histogram directory it all came from.
    """
    lines = [
        f"Stage-2 variation validation plots -- sample composition of the totals ({year})",
        "=" * 78,
        f"histograms : {hist_path}",
        f"sample cfg : {samples_yaml}",
        f"year       : {year}",
        f"generated  : {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Signal/background membership is resolved for this year from the sample",
        "config above; anything it does not list (e.g. observed data) is excluded",
        "from both totals. total_background and total_signal sum the WEIGHTED",
        "EVENTS of the samples listed below, bin by bin.",
        "",
    ]

    # Dumped verbatim from load_sample_roles(samples_yaml, year), before any
    # matching against what is on disk. If a sample is missing from a total,
    # these two blocks say whether the config never listed it or the histogram
    # directory did not have it.
    for role in ("signal", "background"):
        names = sorted(entry[f"{role}_names"])
        lines.append(f"{role}_names from the sample config ({len(names)}):")
        lines += [f"    {name}" for name in names] or ["    (none)"]
        lines.append("")

    for group in ("signal", "background"):
        members = entry[group]
        lines.append(
            f"total_{group} -- summed into the plots "
            f"({len(members)} of {len(entry[f'{group}_names'])} config entries "
            "present in this histogram directory):"
        )
        lines += [f"    {name}" for name in members] or ["    (none)"]
        missing = sorted(set(entry[f"{group}_names"]) - set(members))
        if missing:
            lines.append(f"  listed in the config but absent here ({len(missing)}):")
            lines += [f"    {name}" for name in missing]
        lines.append("")

    if entry["unclassified"]:
        lines.append(f"excluded, not in the sample config ({len(entry['unclassified'])}):")
        lines += [f"    {name}" for name in entry["unclassified"]]
        lines.append("")

    if entry.get("nominal_totals"):
        lines.append("nominal yield per region (sum over the group's samples):")
        for (group, region), value in sorted(entry["nominal_totals"].items()):
            lines.append(f"    total_{group:<10} {region:<14} {value:12.4f}")

    path = Path(outdir) / str(year) / "metadata.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def format_sample_list(used, fell_back, width=110):
    """The provenance line printed under each total plot."""
    import textwrap

    lines = textwrap.wrap(
        f"summed MC samples ({len(used)}): " + ", ".join(used), width=width
    )
    if fell_back:
        lines += textwrap.wrap(
            f"nominal used for (no such systematic): {', '.join(fell_back)}",
            width=width,
        )
    return "\n".join(lines)


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
    # +-25% reference lines, so the size of a variation can be read off the panel
    # without measuring against the axis ticks.
    for guide in RATIO_GUIDES:
        ax_ratio.axhline(guide, color="grey", linewidth=0.8, linestyle=":")
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
            edges, nominal_values = folded_arrays(
                histogram, "nominal", selection, include_flow
            )
        except Exception as exc:
            print(f"  [skip] {sample} {year} {region}: could not extract nominal ({exc})")
            continue

        n_regular_bins = len(edges) - 1
        nominal_folded = nominal_values
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
                _, up_folded = folded_arrays(histogram, up_label, selection, include_flow)
            if has_down:
                _, down_folded = folded_arrays(histogram, down_label, selection, include_flow)

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


def plot_total_group(histograms, group_name, year, pdf, include_flow, regions):
    """One page per region + systematic for the SUM over a group of MC samples."""
    if not histograms:
        return 0

    score_name = score_axis_name(next(iter(histograms.values())))
    systematics = discover_systematics(
        set().union(*(variation_names(h) for h in histograms.values()))
    )
    if not systematics:
        return 0

    n_pages = 0
    for region in regions:
        selection = {"region": region, "channel": CHANNEL}
        edges, nominal, nominal_used, _ = sum_group(
            histograms, "nominal", selection, include_flow
        )
        if nominal is None or not np.any(nominal > 0):
            print(f"  [skip] {group_name} {year} {region}: empty nominal sum")
            continue

        for syst in systematics:
            up_edges, up, _, up_fell = sum_group(
                histograms, f"{syst}_up", selection, include_flow
            )
            down_edges, down, _, down_fell = sum_group(
                histograms, f"{syst}_down", selection, include_flow
            )
            if up is None and down is None:
                continue

            fig, (ax_main, ax_ratio) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(7, 6.6),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
                constrained_layout=True,
            )
            draw_comparison(ax_main, ax_ratio, edges, nominal, up, down, syst)
            fig.suptitle(f"{group_name} | {year} | region={region}", fontsize=11)
            fig.text(
                0.01,
                -0.02,
                format_sample_list(nominal_used, sorted(set(up_fell) | set(down_fell))),
                fontsize=6,
                color="dimgray",
                va="top",
                transform=fig.transFigure,
            )
            fig.text(0.01, 0.005, f"score axis: {score_name}", fontsize=6, color="gray")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
    return n_pages


def main(hist_path, outdir, years, samples, include_flow, regions,
         totals=True, samples_yaml=DEFAULT_SAMPLES_YAML):
    outdir.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    metadata = {}
    for year in years:
        year_dir = hist_path / year
        if not year_dir.is_dir():
            print(f"Skipping {year}: directory not found at {year_dir}")
            continue

        sample_list = samples or sorted(
            p.name[: -len("_hist.pkl")] for p in year_dir.glob("*_hist.pkl")
        )
        loaded = {}
        for sample in sample_list:
            pkl_path = year_dir / f"{sample}_hist.pkl"
            if not pkl_path.is_file():
                print(f"Skipping {sample} ({year}): no histogram at {pkl_path}")
                continue

            histogram = load_histogram(pkl_path)
            loaded[sample] = histogram
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

        if not totals:
            continue

        signal_names, background_names = load_sample_roles(samples_yaml, year)
        signal, background, unclassified = classify_samples(
            loaded, signal_names, background_names
        )
        if unclassified:
            print(f"  Not in {Path(samples_yaml).name} for {year}, excluded from both "
                  f"totals: {', '.join(unclassified)}")
        metadata[year] = {
            # Straight from load_sample_roles(), before anything downstream
            # touches them: what the sample config itself says for this year.
            "signal_names": signal_names,
            "background_names": background_names,
            # What was actually found in the histogram directory and summed.
            "signal": signal,
            "background": background,
            "unclassified": unclassified,
            "nominal_totals": {},
        }
        for group_name, members in (("total_background", background),
                                    ("total_signal", signal)):
            if not members:
                print(f"No {group_name} samples found for {year}; skipped")
                continue
            group = {name: loaded[name] for name in members}
            role = group_name[len("total_"):]
            for region in regions:
                try:
                    _, nominal, _, _ = sum_group(
                        group, "nominal", {"region": region, "channel": CHANNEL},
                        include_flow,
                    )
                except ValueError as exc:
                    print(f"  [skip] {group_name} {year} {region}: {exc}")
                    continue
                if nominal is not None:
                    metadata[year]["nominal_totals"][(role, region)] = float(nominal.sum())
            out_path = outdir / year / f"{group_name}_variation_validation.pdf"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(out_path) as pdf:
                n_pages = plot_total_group(
                    group, group_name, year, pdf, include_flow, regions
                )
            if n_pages:
                print(f"Wrote {n_pages} page(s) -> {out_path}  "
                      f"[{len(members)} samples: {', '.join(members)}]")
                total_pages += n_pages
            else:
                out_path.unlink(missing_ok=True)
                print(f"No shape systematics to plot for {group_name} ({year})")

        meta_path = write_metadata(
            outdir, hist_path, samples_yaml, year, metadata[year]
        )
        print(f"Wrote sample composition -> {meta_path}")

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
    parser.add_argument(
        "--totals",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Also write total_background and total_signal PDFs, summing the "
            "weighted events of every background / signal MC sample. Each page "
            "lists the samples that went into the sum (default: True)."
        ),
    )
    parser.add_argument(
        "--samples-yaml",
        type=Path,
        default=DEFAULT_SAMPLES_YAML,
        help=(
            "Analysis sample config defining which processes are signal and which "
            "are background, resolved per year. Anything not listed there (e.g. "
            f"data) is excluded from both totals (default: {DEFAULT_SAMPLES_YAML})."
        ),
    )
    args = parser.parse_args()
    samples = [s.strip() for s in args.samples.split(",")] if args.samples else None
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    main(args.hist_path, args.outdir, args.years, samples, args.include_flow, regions,
         totals=args.totals, samples_yaml=args.samples_yaml)
