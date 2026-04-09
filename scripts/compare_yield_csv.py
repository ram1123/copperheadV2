import pandas as pd
import numpy as np


configs = {
    "legacy": {
        "file": "yield_Run3_nanoAODv12_FilterJetsHorn30GeV_Feb23_tightPassLepVeto_NoJER_AddVars_v2_2022postEE.csv",
        "suffix": "_legacy",
    },
    "JetIDFix": {
        "file": "yield_Run3_nanoAODv12_FilterJetsHorn25GeV_HE30GeV_Apr03_tightPassLepVeto_NoJER_JetIDFix_2022postEE.csv",
        "suffix": "_JetIDFix",
    },
}

KEYS = ["sample", "category", "region", "year"]
labels = list(configs.keys())
suffixes = (configs[labels[0]]["suffix"], configs[labels[1]]["suffix"])

dfs = {k: pd.read_csv(v["file"]) for k, v in configs.items()}


df = dfs[labels[0]].merge(
    dfs[labels[1]],
    on=KEYS,
    suffixes=suffixes,
    how="inner",
)

print(df.head())


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
