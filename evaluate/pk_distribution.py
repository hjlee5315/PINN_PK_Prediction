import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from src.config import OUTPUT_DIR

NONMEM_POP = {"CL": 0.00762, "V1": 4.27, "Q": 0.0171, "V2": 5.44}
UNITS       = {"CL": "L/h",   "V1": "L",  "Q": "L/h",  "V2": "L"}


# PK parameter distribution (2x2 panel) vs NONMEM population values
def plot_pk_distribution(result_df, title="PK Parameter Distribution",
                         save_name="pk_distribution.png"):
    pat_pk   = result_df.groupby("ID")[["CL", "V1", "Q", "V2"]].first()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=15, y=1.02)

    for ax, param in zip(axes.flatten(), ["CL", "V1", "Q", "V2"]):
        vals = pat_pk[param].values
        ax.hist(vals, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(NONMEM_POP[param], color="red", lw=2.5, linestyle="--",
                   label=f"NONMEM\n{NONMEM_POP[param]:.5f}")
        ax.axvline(np.mean(vals), color="orange", lw=2.5,
                   label=f"PINN mean\n{np.mean(vals):.5f}")
        ax.set_xlabel(f"{param} ({UNITS[param]})", fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title(f"Individual {param}")
        ax.legend(fontsize=8)

        # Scientific notation for small-range parameters
        if param in ["CL", "Q"]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True); fmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved → {OUTPUT_DIR / save_name}")
