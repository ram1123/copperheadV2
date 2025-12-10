def classify_year(year: str) -> dict:
    run2 = any(x in year for x in ["2016", "2017", "2018", "RERECO"])
    run3 = any(x in year for x in ["22", "23", "24","25"])
    return {"run2": run2, "run3": run3}
