"""Compare per-sample Run-2 DNN histograms with Stage-2 reference outputs.

Run from the repository root with:

    pixi run -e default python \
        scripts/stage2_vbf_hist_validation.py

The test and Stage-2 reference files are both stored as one pickle per sample.
"""

import argparse
from itertools import product
from pathlib import Path
import pickle

import numpy as np


RUN2_YEARS = ("2018", "2017", "2016postVFP", "2016preVFP")
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9
DEFAULT_FLOW = True
# TEST_HISTOGRAM_BASE = Path(
#     # "quick_tests/histograms/trained_dnn_groups/with_variations"
#     "quick_tests/histograms/trained_dnn_groups/nominal"
# )
# TEST_HISTOGRAM_BASE = Path(
#     "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
#     "Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/"
#     "stage2_histograms/"
#     "score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc_"
#     # "Jun22_2026_stage2PR_test"
#     "Jun22_2026_stage2PR_test_vbf_filter_study_NoSyst"
# )
# REFERENCE_HISTOGRAM_BASE = Path(
#     "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
#     "Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/"
#     "stage2_histograms/"
#     "score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc_"
#     "Jun11_2026_50nTrialsFoldsAll_Max57bins_NoSyst"
#     # "Jun20_2026_50nTrialsFoldsAll_Max57bins"
# )

TEST_HISTOGRAM_BASE = Path(
    "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
    "Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo/"
    "stage2_histograms/"
    "score_Run2_NanoV15_forVBFChannel_July06_2026_jetUncRedo_Jul04_2026_50nTrialsFoldsAll_Max70bins_NoSyst"
)
REFERENCE_HISTOGRAM_BASE = Path(
    "/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/"
    "Run2_NanoV15_forVBFChannel_June26_2026_jetUnc/"
    "stage2_histograms/"
    "score_Run2_NanoV15_forVBFChannel_June26_2026_jetUnc_Jul04_2026_50nTrialsFoldsAll_Max70bins_NoSyst"
)

SAMPLES = (
    "data",
    # "dyTo2L_M-50_aMCatNLO",
    "dyTo2Mu_M-50_MiNNLO",
    "dyTo2Mu_M-100to200_MiNNLO",
    "dy_VBF_filter",
    "ttjets_dl",
    "ttjets_sl",
    "ewk_zlljj",
    "ww_2l2nu",
    "wz_1l1nu2q",
    "wz_2l2q",
    "wz_3lnu",
    "ggh_powhegPS",
    "vbf_powheg_dipole",
)


def parse_years(years_text):
    """Parse a comma-separated Run-2 year list while preserving its order."""
    years = tuple(dict.fromkeys(year.strip() for year in years_text.split(",") if year.strip()))
    if not years:
        raise argparse.ArgumentTypeError("at least one year must be provided")
    invalid = [year for year in years if year not in RUN2_YEARS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported year(s): {', '.join(invalid)}; "
            f"choose from {', '.join(RUN2_YEARS)}"
        )
    return years


def load_histogram(path):
    with path.open("rb") as histogram_file:
        return pickle.load(histogram_file)


def score_axis_name(histogram):
    names = [name for name in histogram.axes.name if name.startswith("score")]
    if len(names) != 1:
        raise ValueError(
            f"Expected one score axis, found {names} in {histogram.axes.name}"
        )
    return names[0]


def variation_names(histogram):
    """Return available variation labels, treating old histograms as nominal."""
    if "variation" not in histogram.axes.name:
        return {"nominal"}
    return set(histogram.axes["variation"])


def side_by_side_axes(test_histogram, reference_histogram):
    """Return common non-score axes that should not be projected away."""
    skipped_axes = {"variation", "val_sumw2", score_axis_name(test_histogram)}
    reference_axes = set(reference_histogram.axes.name)
    return [
        axis_name
        for axis_name in test_histogram.axes.name
        if axis_name in reference_axes and axis_name not in skipped_axes
    ]


def side_by_side_selections(test_histogram, reference_histogram):
    """Build one selection per common label on each non-score categorical axis."""
    axis_names = side_by_side_axes(test_histogram, reference_histogram)
    if not axis_names:
        return [{}]

    labels_by_axis = []
    for axis_name in axis_names:
        common_labels = sorted(
            set(test_histogram.axes[axis_name]) & set(reference_histogram.axes[axis_name])
        )
        if not common_labels:
            raise ValueError(
                f"No common labels for axis {axis_name!r}: "
                f"test={list(test_histogram.axes[axis_name])}, "
                f"reference={list(reference_histogram.axes[axis_name])}"
            )
        labels_by_axis.append(common_labels)

    return [
        dict(zip(axis_names, labels, strict=True))
        for labels in product(*labels_by_axis)
    ]


def format_selection(selection):
    if not selection:
        return "inclusive"
    return ", ".join(f"{axis}={label}" for axis, label in selection.items())


def score_bin_labels(edges, include_flow):
    labels = [f"[{low}, {high})" for low, high in zip(edges[:-1], edges[1:])]
    if include_flow:
        return [f"underflow (< {edges[0]})", *labels, f"overflow (>= {edges[-1]})"]
    return labels


def extract_arrays(histogram, variation="nominal", axis_selection=None, include_flow=True):
    """Project one variation's values and stored sumw2 onto the score axis."""
    score_name = score_axis_name(histogram)
    common_selection = dict(axis_selection or {})
    if "variation" in histogram.axes.name:
        common_selection["variation"] = variation
    if "val_sumw2" not in histogram.axes.name:
        projected = histogram[common_selection].project(score_name)
        return (
            projected.axes[score_name].edges,
            projected.values(flow=include_flow),
            projected.variances(flow=include_flow),
        )

    value_histogram = histogram[
        {**common_selection, "val_sumw2": "value"}
    ].project(score_name)
    sumw2_histogram = histogram[
        {**common_selection, "val_sumw2": "sumw2"}
    ].project(score_name)
    return (
        value_histogram.axes[score_name].edges,
        value_histogram.values(flow=include_flow),
        sumw2_histogram.values(flow=include_flow),
    )


def arrays_differ(test_arrays, reference_arrays, rtol, atol):
    """Check edges, weighted values, and stored sumw2 with tight tolerances."""
    return any(
        not np.allclose(test, reference, rtol=rtol, atol=atol, equal_nan=True)
        for test, reference in zip(test_arrays, reference_arrays)
    )


def print_comparison(
    sample,
    test_arrays,
    reference_arrays,
    variation="nominal",
    axis_selection=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    include_flow=True,
):
    test_edges, test_values, test_variances = test_arrays
    reference_edges, reference_values, reference_variances = reference_arrays

    print(f"\n{'=' * 80}")
    print(f"Sample: {sample}")
    print(f"Selection: {format_selection(axis_selection or {})}")
    print(f"Variation: {variation}")
    print(f"Bin edges: {test_edges.tolist()}")
    print(f"Flow bins included: {include_flow}")
    if not np.array_equal(test_edges, reference_edges):
        print("Result: NOT COMPARED because the score bin edges differ")
        print(f"Test edges:      {test_edges.tolist()}")
        print(f"Reference edges: {reference_edges.tolist()}")
        return

    difference = test_values - reference_values
    relative_difference = np.divide(
        difference,
        reference_values,
        out=np.full_like(difference, np.nan, dtype=np.float64),
        where=reference_values != 0,
    )

    print(f"Score bins:          {score_bin_labels(test_edges, include_flow)}")
    print(f"Test values:         {test_values.tolist()}")
    print(f"Reference values:    {reference_values.tolist()}")
    print(f"Difference (test-ref): {difference.tolist()}")
    print(f"Relative difference: {relative_difference.tolist()}")
    print(f"Test variances:      {test_variances.tolist()}")
    print(f"Reference sumw2:     {reference_variances.tolist()}")
    print(f"Test integral:       {np.sum(test_values):.12g}")
    print(f"Reference integral:  {np.sum(reference_values):.12g}")
    print(f"Integral difference: {np.sum(difference):.12g}")
    print(f"Maximum abs. bin difference: {np.max(np.abs(difference)):.12g}")
    finite_relative = np.abs(relative_difference[np.isfinite(relative_difference)])
    if finite_relative.size:
        print(f"Maximum abs. relative difference: {np.max(finite_relative):.12g}")
    print(f"Values exactly equal: {np.array_equal(test_values, reference_values)}")
    print(
        f"Values close (rtol={rtol:g}, atol={atol:g}): "
        f"{np.allclose(test_values, reference_values, rtol=rtol, atol=atol)}"
    )


def compare_year(
    year,
    test_histogram_base=TEST_HISTOGRAM_BASE,
    reference_histogram_base=REFERENCE_HISTOGRAM_BASE,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    print_all=False,
    include_flow=DEFAULT_FLOW,
):
    """Compare all available samples for one Run-2 year."""
    test_directory = Path(test_histogram_base) / year
    reference_directory = Path(reference_histogram_base) / year
    if not test_directory.is_dir():
        print(f"Skipping {year}: test directory does not exist: {test_directory}")
        return 0, 0
    if not reference_directory.is_dir():
        print(
            f"Skipping {year}: reference directory does not exist: "
            f"{reference_directory}"
        )
        return 0, 0

    print(f"\n{'#' * 80}")
    print(f"Year: {year}")
    print(f"Test histograms: {test_directory}")
    print(f"Reference histograms: {reference_directory}")
    compared_samples = 0
    failed_histograms = 0
    for sample in SAMPLES:
        test_path = test_directory / f"{sample}_hist.pkl"
        reference_path = reference_directory / f"{sample}_hist.pkl"
        if not test_path.is_file():
            print(f"\nSkipping {sample}: no test histogram at {test_path}")
            continue
        if not reference_path.is_file():
            print(f"\nSkipping {sample}: no reference histogram at {reference_path}")
            continue

        test_histogram = load_histogram(test_path)
        reference_histogram = load_histogram(reference_path)

        common_variations = sorted(
            (
                variation_names(test_histogram) & variation_names(reference_histogram)
            )
            - {"nominal"}
        )
        selections = side_by_side_selections(test_histogram, reference_histogram)
        # raise ValueError(f"selections: {selections}")
        for selection in selections:
            nominal_test_arrays = extract_arrays(
                test_histogram,
                axis_selection=selection,
                include_flow=include_flow,
            )
            nominal_reference_arrays = extract_arrays(
                reference_histogram,
                axis_selection=selection,
                include_flow=include_flow,
            )
            nominal_differs = arrays_differ(
                nominal_test_arrays,
                nominal_reference_arrays,
                rtol,
                atol,
            )
            if nominal_differs or print_all:
                print_comparison(
                    sample,
                    nominal_test_arrays,
                    nominal_reference_arrays,
                    axis_selection=selection,
                    rtol=rtol,
                    atol=atol,
                    include_flow=include_flow,
                )
            failed_histograms += nominal_differs

            different_variations = 0
            for variation in common_variations:
                reference_arrays = extract_arrays(
                    reference_histogram,
                    variation,
                    selection,
                    include_flow,
                )
                test_arrays = extract_arrays(
                    test_histogram,
                    variation,
                    selection,
                    include_flow,
                )
                variation_differs = arrays_differ(
                    test_arrays,
                    reference_arrays,
                    rtol,
                    atol,
                )
                if variation_differs or print_all:
                    print_comparison(
                        sample,
                        test_arrays,
                        reference_arrays,
                        variation,
                        selection,
                        rtol,
                        atol,
                        include_flow,
                    )
                if variation_differs:
                    different_variations += 1
                    failed_histograms += 1
            if common_variations and (different_variations or print_all):
                print(
                    f"Differing common variations for {sample} "
                    f"({format_selection(selection)}): "
                    f"{different_variations}/{len(common_variations)}"
                )
        compared_samples += 1

    print(f"\n{year} samples compared: {compared_samples}")
    print(f"{year} histogram comparisons outside tolerance: {failed_histograms}")
    return compared_samples, failed_histograms


def main(
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    print_all=False,
    years=RUN2_YEARS,
    test_histogram_path=TEST_HISTOGRAM_BASE,
    reference_histogram_path=REFERENCE_HISTOGRAM_BASE,
    include_flow=DEFAULT_FLOW,
):
    print(f"Comparison tolerances: rtol={rtol:g}, atol={atol:g}")
    print(f"Print all histogram comparisons: {print_all}")
    print(f"Include underflow/overflow bins: {include_flow}")
    total_groups = 0
    total_failures = 0
    years_compared = 0
    for year in years:
        compared_samples, failed_histograms = compare_year(
            year,
            test_histogram_path,
            reference_histogram_path,
            rtol,
            atol,
            print_all,
            include_flow,
        )
        total_groups += compared_samples
        total_failures += failed_histograms
        years_compared += compared_samples > 0

    if not years_compared:
        raise FileNotFoundError("No matching Run-2 histogram directories were found")
    print(f"\nRun-2 years compared: {years_compared}/{len(years)}")
    print(f"Total year/sample comparisons: {total_groups}")
    print(f"Total histogram comparisons outside tolerance: {total_failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative comparison tolerance (default: {DEFAULT_RTOL:g}).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute comparison tolerance (default: {DEFAULT_ATOL:g}).",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print passing histogram comparisons as well as failures.",
    )
    parser.add_argument(
        "--flow",
        dest="include_flow",
        default=DEFAULT_FLOW,
        action=argparse.BooleanOptionalAction,
        help=(
            "Include underflow and overflow bins in value and sumw2 comparisons "
            f"(default: {DEFAULT_FLOW})."
        ),
    )
    parser.add_argument(
        "--years",
        type=parse_years,
        default=RUN2_YEARS,
        metavar="YEAR[,YEAR...]",
        help=(
            "Comma-separated Run-2 years to compare "
            f"(default: {','.join(RUN2_YEARS)})."
        ),
    )
    parser.add_argument(
        "--test-histogram-path",
        type=Path,
        default=TEST_HISTOGRAM_BASE,
        help=(
            "Base directory containing test histograms, with one subdirectory "
            f"per year (default: {TEST_HISTOGRAM_BASE})."
        ),
    )
    parser.add_argument(
        "--reference-histogram-path",
        type=Path,
        default=REFERENCE_HISTOGRAM_BASE,
        help=(
            "Base directory containing reference histograms, with one "
            f"subdirectory per year (default: {REFERENCE_HISTOGRAM_BASE})."
        ),
    )
    arguments = parser.parse_args()
    if arguments.rtol < 0 or arguments.atol < 0:
        parser.error("--rtol and --atol must be non-negative")
    main(
        arguments.rtol,
        arguments.atol,
        arguments.print_all,
        arguments.years,
        arguments.test_histogram_path,
        arguments.reference_histogram_path,
        arguments.include_flow,
    )
