import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default=".",
    action="store",
    help="path where the edge validation is saved",
    )
    parser.add_argument(
    "--label",
    dest="label",
    default="",
    action="store",
    help="stage2 run label",
    )
    args = parser.parse_args()
    save_path = f"{args.save_path}/{args.label}"
    os.makedirs(save_path, exist_ok=True)
    
    for iter_idx in range(1, 6):
        load_path = f"{save_path}/iter{iter_idx}_significances.csv"
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
        print(f"xticks: {xticks}")
        plt.xticks(sorted(xticks))  # Set updated ticks

        
        plt.xlabel("Signal Efficiency")
        plt.ylabel("Approximate Median Significance")
        plt.savefig(f"{save_path}/iter{iter_idx}_significances.png")
        plt.clf()


        # -----------------------------
        # Write YAML target yields
        # -----------------------------
        # UPDATE IT: such that it will save key as the label name and in further steps it 
        # should extract target yeilds using the label name instead of "target_yeild". So, 
        # that we keep target yields for each labels.
        if iter_idx != 5: continue

        boundaries = sorted(not_common_effs)

        yields = []
        yields.append(boundaries[0])

        for i in range(1, len(boundaries)):
            yields.append(boundaries[i] - boundaries[i-1])

        yields.append(1.0 - boundaries[-1])

        yields = np.round(yields, 2)

        yaml_data = {
            "target_yields": yields.tolist()
        }

        with open("stage2/ggH/target_yields.yaml", "w") as f:
            f.write(
                f"target_yields:  # {args.label}\n"
            )
            f.write(f"# {boundaries}\n")

            for i, y in enumerate(yields):
                if i == 0:
                    f.write(f"- {y}\n")
                elif i == len(yields) - 1:
                    f.write(f"- {y}  # 1.0 - {boundaries[-1]}\n")
                else:
                    low = int(boundaries[i-1]*100)
                    high = int(boundaries[i]*100)
                    f.write(f"- {y}  # {high}-{low}\n")
