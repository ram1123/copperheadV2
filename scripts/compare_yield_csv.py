import pandas as pd
import numpy as np


df_b = pd.read_csv(
    "yield_Run2_NanoV15_Feb23_2026_noQGLSF_2017.csv"
)
df_a = pd.read_csv(
    "yield_Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_2017.csv"
)

KEYS = ["sample", "category", "region", "year"]

suffixes = ("_V12", "_V15")
df = df_a.merge(
    df_b,
    on=KEYS,
    suffixes=suffixes,
    how="inner",  # only compare common rows
)

# print(df.head(10))


def percent_diff(a, b):
    return np.where(a != 0, 100.0 * (b - a) / a, np.nan)


df["raw_events_diff_pct"] = percent_diff(df[f"raw_events{suffixes[0]}"], df[f"raw_events{suffixes[1]}"])

df["yield_diff_pct"] = percent_diff(df[f"yield{suffixes[0]}"], df[f"yield{suffixes[1]}"])

df["raw_events_diff_pct"] = percent_diff(df[f"raw_events{suffixes[0]}"], df[f"raw_events{suffixes[1]}"])

df["yield_diff_pct"] = percent_diff(df[f"yield{suffixes[0]}"], df[f"yield{suffixes[1]}"])

cols = KEYS + [
    f"raw_events{suffixes[0]}",
    f"raw_events{suffixes[1]}",
    "raw_events_diff_pct",
    f"yield{suffixes[0]}",
    f"yield{suffixes[1]}",
    "yield_diff_pct",
]

df_cmp = df[cols].sort_values(["year", "category", "region", "sample"])

df_cmp.to_csv("yield_comparison.csv", index=False)


# print only for vbf category, remove NaN rows
df_vbf = df_cmp[df_cmp["category"] == "vbf"].dropna()
print(df_vbf)


for year, g in df_vbf.groupby("year", sort=False):
    print("\n" + "=" * 100)
    print(f"YEAR: {year}")
    print("=" * 100)
    print(g.to_string(index=False))
