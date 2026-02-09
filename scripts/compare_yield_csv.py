import pandas as pd
import numpy as np


df_a = pd.read_csv(
    "yield_Run3_nanoAODv12_02Feb_FilterJets_2022preEE_2022postEE_2023_2023BPix_2024.csv"
)
df_b = pd.read_csv(
    "yield_Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV_2022preEE_2022postEE_2023_2023BPix_2024.csv"
)

KEYS = ["sample", "category", "region", "year"]

df = df_a.merge(
    df_b,
    on=KEYS,
    suffixes=("_A", "_B"),
    how="inner",  # only compare common rows
)

# print(df.head(10))


def percent_diff(a, b):
    return np.where(a != 0, 100.0 * (b - a) / a, np.nan)


df["raw_events_diff_pct"] = percent_diff(df["raw_events_A"], df["raw_events_B"])

df["yield_diff_pct"] = percent_diff(df["yield_A"], df["yield_B"])

df["raw_events_diff_pct"] = percent_diff(df["raw_events_A"], df["raw_events_B"])

df["yield_diff_pct"] = percent_diff(df["yield_A"], df["yield_B"])

cols = KEYS + [
    "raw_events_A",
    "raw_events_B",
    "raw_events_diff_pct",
    "yield_A",
    "yield_B",
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
