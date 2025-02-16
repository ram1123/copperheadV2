import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    
    for iter_idx in range(1, 8):
        load_path = f"iter{iter_idx}_significances.csv"
        sig_df = pd.read_csv(load_path)
        plt.scatter(sig_df["sig_eff"], sig_df["Significance"], color='blue', s=50, alpha=0.7)  # s=50 sets dot size, alpha=0.7 makes them slightly transparent

        full_sig_effs = np.arange(0.01, 1.00, 0.01)
        df_sig_eff = sig_df["sig_eff"]
        # round the two eff to 2 decimal places before comparing them
        full_sig_effs = np.round(full_sig_effs, 2)
        df_sig_eff = np.round(df_sig_eff, 2)
        # Find elements that are NOT common and plot vertical lines on those values
        not_common_effs = np.setxor1d(full_sig_effs, df_sig_eff)

        
        for removed_eff in not_common_effs:
            plt.axvline(x=removed_eff, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'X={removed_eff:.1f}')

        
        plt.xlabel("Signal Efficiency")
        plt.ylabel("Approximate Median Significance")
        plt.savefig(f"iter{load_path}_significances.png")
        plt.clf()

        