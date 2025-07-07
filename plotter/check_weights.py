#!/usr/bin/env python3
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

# years = ["2016","2017","2018"]
years = ["2018"]
# sample = "dy_VBF_filter"
sample = "dy_VBF_filter_NewZWgt"
# sample = "dy_M-100To200_MiNNLO"
# sample = "dy_M-50_MiNNLO"
# adjust to wherever your DYVBF parquet folders live
LOAD_PATH = "/depot/cms/users/shar1172/hmm/copperheadV1clean/" \
            "Run2_nanoAODv12_08June/stage1_output/{year}/f1_0/" \
            f"{sample}/*/*.parquet"

# fraction to sample (so we don't accidentally pull 10s of GB into memory)
SAMPLE_FRAC = 1.0

outdir = "plots_separate_weights"
os.makedirs(outdir, exist_ok=True)

# ─── READ & FIND WEIGHTS ────────────────────────────────────────────────────────

# build a glob for all years
paths = [LOAD_PATH.format(year=yr) for yr in years]
print("Reading parquet from:", *paths, sep="\n  ")

# read into a single dask DataFrame
df = dd.read_parquet(paths)

# find every column that begins with "separate"
sep_cols = [c for c in df.columns if c.startswith("separate")]
# add "wgt_nominal" to the list of columns to plot
sep_cols.append("wgt_nominal")
print(f"Found {len(sep_cols)} 'separate*' columns:", sep_cols)

if not sep_cols:
    raise RuntimeError("No columns starting with 'separate' found in DYVBF sample!")

# ─── SAMPLE & COMPUTE ───────────────────────────────────────────────────────────

# do a small random uniform sample on rows, then pull into pandas
pdf = df[sep_cols].sample(frac=SAMPLE_FRAC, random_state=42).compute()

print(f"Sampled ~{len(pdf)} rows ({SAMPLE_FRAC*100:.1f}% of total)")

# ─── PLOT EACH DISTRIBUTION ────────────────────────────────────────────────────

n = len(sep_cols)
ncols = 2
nrows = (n + ncols - 1)//ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 3*nrows), squeeze=False)

for col in sep_cols:
    data = pdf[col].dropna().values
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=50, histtype="step", color="C0")
    plt.title(f"{sample} - {col}")
    plt.xlabel(col)
    plt.ylabel("Entries")
    plt.grid(True, ls="--", alpha=0.3)
    fn = os.path.join(outdir, f"{sample}_{col}.pdf")
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
    print("Saved:", fn)

# for idx, col in enumerate(sep_cols):
#     r = idx // ncols
#     c = idx % ncols
#     ax = axes[r][c]
#     data = pdf[col].dropna().values
#     ax.hist(data, bins=50, histtype="step", color="C{}".format(idx%10))
#     ax.set_title(col)
#     ax.set_xlabel(col)
#     ax.set_ylabel("Entries")
#     ax.grid(True, ls="--", alpha=0.3)
# # turn off any unused subplots
# for idx in range(n, nrows*ncols):
#     axes[idx//ncols][idx%ncols].axis("off")

# plt.tight_layout()
# outpath = os.path.join(outdir, f"{sample}_separate_weights.pdf")
# plt.savefig(outpath, dpi=200)
# print("Saved plot to", outpath)
