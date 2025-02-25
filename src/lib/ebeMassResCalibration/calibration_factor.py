import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from basic_class_for_calibration import get_calib_categories

# --- Step 1. Load the data and compute mass resolution ---
# Read the parquet files into a Dask DataFrame
df = dd.read_parquet("/depot/cms/users/shar1172/hmm/copperheadV1clean/Run2_nanoAODv12_24Feb_BSCorr//stage1_output/2018/f1_0/data_*/*/part*.parquet")

# Compute the per-event energy per muon (assumed to be half the dimuon mass) and the error contributions
df = df.assign(
    muon_E = df['dimuon_mass'] / 2,
    dpt1 = (df['mu1_ptErr'] / df['mu1_pt']) * (df['dimuon_mass'] / 2),
    dpt2 = (df['mu2_ptErr'] / df['mu2_pt']) * (df['dimuon_mass'] / 2)
)

# Combine in quadrature to get the mass resolution (absolute uncertainty on the dimuon mass)
df = df.assign(
    dimuon_ebe_mass_res_calc = np.sqrt(df['dpt1']**2 + df['dpt2']**2)
)

# Optionally compute the relative mass resolution using muon_E (which is dimuon_mass/2)
df = df.assign(
    # dimuon_ebe_mass_res_rel_calc = df['dimuon_ebe_mass_res_calc'] / df['muon_E']
    dimuon_ebe_mass_res_rel_calc = df['dimuon_ebe_mass_res_calc']
)

# Trigger the computation
result = df.compute()
print(result[['dimuon_ebe_mass_res_calc', 'dimuon_ebe_mass_res_rel_calc']].head())


# --- Step 3. For each calibration category, plot a histogram with its median value ---
# Get the dictionary of boolean masks for calibration categories from the result DataFrame.
calib_cats = get_calib_categories(result)

# Loop over each category, compute the median mass resolution, and create a histogram.
for cat_name, mask in calib_cats.items():
    # Select events in this calibration category.
    cat_data = result[mask]
    # If there are no events, skip this category.
    if cat_data.empty:
        print(f"Category {cat_name} has no events, skipping.")
        continue
    # Compute the median mass resolution for this category.
    median_val = cat_data['dimuon_ebe_mass_res_calc'].median()

    # Plot the histogram.
    plt.figure()
    plt.hist(cat_data['dimuon_ebe_mass_res_calc'], bins=100, range=(0, 5.0), color='C0', alpha=0.7)
    plt.xlabel('Dimuon mass resolution (GeV)')
    plt.ylabel('Events')
    plt.title(f"Category {cat_name}\nMedian = {median_val:.4f} GeV")
    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2,
                label=f"Median: {median_val:.4f} GeV")
    plt.legend()
    plt.savefig(f'mass_resolution_{cat_name}.png')
    plt.close()
    print(f"Saved histogram for category {cat_name} (median = {median_val:.4f} GeV)")

    # Save the category index and median value to a CSV file.
    with open('mass_resolution_medians.txt', 'a') as f:
        f.write(f"{cat_name} {median_val}\n")

# Optionally, plot the overall histogram too.
plt.figure()
plt.hist(result['dimuon_ebe_mass_res_calc'], bins=100, range=(0, 5.0), color='C1', alpha=0.7)
overall_median = result['dimuon_ebe_mass_res_calc'].median()
plt.xlabel('Dimuon mass resolution (GeV)')
plt.ylabel('Events')
plt.title(f"Overall Dimuon Mass Resolution\nMedian = {overall_median:.4f} GeV")
plt.axvline(overall_median, color='red', linestyle='dashed', linewidth=2,
            label=f"Median: {overall_median:.4f} GeV")
plt.legend()
plt.savefig('mass_resolution_overall.png')
plt.close()
print(f"Saved overall mass resolution histogram (median = {overall_median:.4f} GeV)")
