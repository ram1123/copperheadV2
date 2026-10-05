# Stage-3-only pseudo-years whose stage-2 histograms are summed into one set of templates
# (PC guideline: 2025+2026 is one year). Stage-1/2 always run on the component years.
MERGED_YEARS = {
    "2025_2026": ("2025", "2026"),
}


def component_years(year: str) -> tuple:
    """Stage-2 years whose histograms make up `year`; a plain year is its own component."""
    return MERGED_YEARS.get(str(year), (str(year),))


def classify_year(year: str) -> dict:
    run2 = any(x in year for x in ["2016", "2017", "2018", "RERECO"])
    run3 = any(x in year for x in ["22", "23", "24","25"])
    return {"run2": run2, "run3": run3}


def is_run2(year) -> bool:
    """
    Accepts:
      - '2016preVFP', '2016postVFP', '2017', '2018'
      - 2016, 2017, 2018 (int)
    """
    if isinstance(year, int):
        return year in (2016, 2017, 2018)

    if isinstance(year, str):
        return year.startswith("2016") or year in ("2017", "2018")

    raise TypeError(f"Unsupported year type: {year} ({type(year)})")


def is_run3(year) -> bool:
    """ if not run2, then run3 """
    return not is_run2(year)
