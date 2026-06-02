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
        return year.startswith(("2016", "2017", "2018"))

    raise TypeError(f"Unsupported year type: {year} ({type(year)})")


def is_run3(year) -> bool:
    """ if not run2, then run3 """
    return not is_run2(year)
