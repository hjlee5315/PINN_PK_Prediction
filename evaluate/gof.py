import numpy as np
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR


# Goodness-of-fit and weighted residual plots
def plot_gof_wres(result_df, title="Goodness-of-fit — PINN",
                  save_name="gof_wres.png"):
    df_plot = result_df[result_df["TIME"] > 0].copy()

    obs   = df_plot["DV_obs"].values
    pred  = df_plot["DV_pred"].values
    times = df_plot["TIME"].values

    # Weighted residuals
    residual = obs - pred
    wres = (residual - np.mean(residual)) / (np.std(residual) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(title, fontsize=16, y=1.02, fontweight='bold')

    scatter_gof  = dict(alpha=0.35, s=20, color='steelblue',
                        edgecolors='navy', linewidth=0.3)
    scatter_wres = dict(alpha=0.25, s=8,  color='steelblue',
                        edgecolors='navy', linewidth=0.1)

    # (a) Observed vs Predicted
    ax = axes[0]
    lo = min(obs.min(), pred.min()) * 0.95
    hi = max(obs.max(), pred.max()) * 1.05
    ax.scatter(obs, pred, **scatter_gof, label='Data points')
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.8, label='Identity line')
    xs = np.linspace(lo, hi, 300)
    ax.plot(xs, np.poly1d(np.polyfit(obs, pred, 1))(xs),
            color='#e6550d', lw=1.8, label='Trend line')
    ax.set_xlabel('Observed Concentration (mg/L)', fontsize=11)
    ax.set_ylabel('Predicted Concentration (mg/L)', fontsize=11)
    ax.set_title('(a) Observed versus predicted concentrations',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (b) WRES vs Predicted
    ax = axes[1]
    ax.scatter(pred, wres, **scatter_wres)
    for y, ls, c in [(0, '-', 'black'), (1, ':', 'gray'), (-1, ':', 'gray'),
                     (2, '--', 'gray'),  (-2, '--', 'gray'),
                     (3, '--', 'red'),   (-3, '--', 'red')]:
        ax.axhline(y, color=c, lw=0.8 if abs(y) < 3 else 1.0, linestyle=ls, alpha=0.7)
    ax.plot(np.linspace(pred.min(), pred.max(), 300),
            np.poly1d(np.polyfit(pred, wres, 1))(np.linspace(pred.min(), pred.max(), 300)),
            color='#e6550d', lw=1.8, label='Trend line')
    ax.set_xlabel('Predicted Concentration (mg/L)', fontsize=11)
    ax.set_ylabel('Weighted Residual', fontsize=11)
    ax.set_title('(b) Weighted residuals versus predicted concentrations',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 5); ax.grid(True, alpha=0.25)

    # (c) WRES vs Time
    ax = axes[2]
    ax.scatter(times, wres, **scatter_wres)
    for y, ls, c in [(0, '-', 'black'), (1, ':', 'gray'), (-1, ':', 'gray'),
                     (2, '--', 'gray'),  (-2, '--', 'gray'),
                     (3, '--', 'red'),   (-3, '--', 'red')]:
        ax.axhline(y, color=c, lw=0.8 if abs(y) < 3 else 1.0, linestyle=ls, alpha=0.7)
    ax.plot(np.linspace(times.min(), times.max(), 300),
            np.poly1d(np.polyfit(times, wres, 1))(np.linspace(times.min(), times.max(), 300)),
            color='#e6550d', lw=1.8)
    ax.set_xlabel('Time (hr)', fontsize=11)
    ax.set_ylabel('Weighted Residual', fontsize=11)
    ax.set_title('(c) Weighted residuals versus time',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 5); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved → {OUTPUT_DIR / save_name}")

