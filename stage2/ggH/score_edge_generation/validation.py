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
            plt.axvline(x=removed_eff, color='grey', linestyle='--', linewidth=2, alpha=0.7, label=f'X={removed_eff:.1f}')
            plt.text(removed_eff, plt.ylim()[0] - 0.5, f'{removed_eff:.2f}', ha='center', va='top', fontsize=12, color='red')

        # plot bright red vertical line over the sig eff with max AMS
        max_ix = np.argmax(sig_df["Significance"])
        max_sig_eff = df_sig_eff[max_ix]
        plt.axvline(x=max_sig_eff, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'X={max_sig_eff:.1f}')
        plt.text(max_sig_eff, plt.ylim()[0] - 0.5, f'{max_sig_eff:.2f}', ha='center', va='top', fontsize=12, color='red')

        print(f"not_common_effs: {not_common_effs}")
        print(f"max_sig_eff: {max_sig_eff}")
        # # Update x-ticks to include vertical line positions
        # xticks = list(plt.xticks()[0])  # Get current x-ticks
        # xticks.extend([max_sig_eff])  # Add vertical line positions
        xticks = list(not_common_effs) + [max_sig_eff]
        print(xticks)
        plt.xticks(sorted(xticks))  # Set updated ticks

        
        plt.xlabel("Signal Efficiency")
        plt.ylabel("Approximate Median Significance")
        plt.savefig(f"iter{iter_idx}_significances.png")
        plt.clf()

