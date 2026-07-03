import numpy as np
import torch
import matplotlib.pyplot as plt
from src.config import DEVICE, SEED, OUTPUT_DIR

COV_COLS_PI = ['BW', 'EGFR', 'SEX', 'BPS', 'RAAS']


# Baseline RMSE computation
def _compute_rmse(model, loader, scaler_dv):
    dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32, device=DEVICE)
    dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32, device=DEVICE)
    model.eval()
    sum_sq = torch.tensor(0.0, device=DEVICE)
    n_obs  = torch.tensor(0,   device=DEVICE)

    with torch.no_grad():
        for batch in loader:
            tn   = batch['times_norm'].to(DEVICE)
            an   = batch['amts_norm'].to(DEVICE)
            dm_n = batch['demographics_norm'].to(DEVICE)
            dv_o = batch['dvs_orig'].to(DEVICE)
            mask = batch['masks'].to(DEVICE)

            pred   = model(tn, an, dm_n)
            pred_o = pred * dv_std + dv_mean
            m      = mask.bool()
            sum_sq += ((pred_o[m] - dv_o[m]) ** 2).sum()
            n_obs  += m.sum()

    return torch.sqrt(sum_sq / n_obs.clamp(min=1)).item()


# Covariate permutation feature importance (ΔRMSE)
def permutation_importance(model, loader, scaler_dv,
                           cov_cols=COV_COLS_PI,
                           n_repeats=5, random_state=SEED):
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(random_state)
    model.eval()

    dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32, device=DEVICE)
    dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32, device=DEVICE)

    base_rmse = _compute_rmse(model, loader, scaler_dv)
    print(f"Baseline RMSE: {base_rmse:.4f} mg/L")

    importances = {cov: [] for cov in cov_cols}

    for cov_idx, cov_name in enumerate(cov_cols):
        print(f"  [{cov_idx + 1}/{len(cov_cols)}] Shuffling '{cov_name}' ...")

        for _ in range(n_repeats):
            sum_sq = torch.tensor(0.0, device=DEVICE)
            n_obs  = torch.tensor(0,   device=DEVICE)

            with torch.no_grad():
                for batch in loader:
                    tn   = batch['times_norm'].to(DEVICE)
                    an   = batch['amts_norm'].to(DEVICE)
                    dm_n = batch['demographics_norm'].to(DEVICE)
                    dv_o = batch['dvs_orig'].to(DEVICE)
                    mask = batch['masks'].to(DEVICE)

                    B       = dm_n.shape[0]
                    perm    = torch.randperm(B, generator=rng, device=DEVICE)
                    dm_shuf = dm_n.clone()
                    dm_shuf[:, cov_idx] = dm_n[perm, cov_idx]

                    pred   = model(tn, an, dm_shuf)
                    pred_o = pred * dv_std + dv_mean
                    m      = mask.bool()
                    sum_sq += ((pred_o[m] - dv_o[m]) ** 2).sum()
                    n_obs  += m.sum()

            shuf_rmse = torch.sqrt(sum_sq / n_obs.clamp(min=1)).item()
            importances[cov_name].append(shuf_rmse - base_rmse)

    import pandas as pd
    imp_mean     = {c: float(np.mean(v)) for c, v in importances.items()}
    importance_df = pd.DataFrame({
        'Feature':            list(imp_mean.keys()),
        'Importance (ΔRMSE)': list(imp_mean.values()),
    }).sort_values('Importance (ΔRMSE)', ascending=False).reset_index(drop=True)

    print("\nPermutation Importance (ΔRMSE ↑ = more important):")
    print(importance_df.to_string(index=False))
    return importance_df


# Permutation importance bar chart
def plot_permutation_importance(importance_df,
                                title="(b) Permutation Importance — PINN",
                                save_name="permutation_importance.png"):
    df_plot   = importance_df.sort_values('Importance (ΔRMSE)', ascending=True)
    vals      = df_plot['Importance (ΔRMSE)'].values
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    colors    = plt.cm.Blues(0.35 + 0.55 * norm_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_plot['Feature'], vals, color=colors,
                   edgecolor='grey', linewidth=0.5, height=0.55)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + vals.max() * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=10)

    ax.axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('Feature Importance (Δ RMSE, mg/L)', fontsize=11)
    ax.set_ylabel('Covariate', fontsize=11)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(right=vals.max() * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved → {OUTPUT_DIR / save_name}")
