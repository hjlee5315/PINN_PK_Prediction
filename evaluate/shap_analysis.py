import numpy as np
import torch
import matplotlib.pyplot as plt
from src.config import DEVICE, OUTPUT_DIR

COV_COLS = ['BW', 'EGFR', 'SEX', 'BPS', 'RAAS']

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not installed. Run: pip install shap")


# Build covariate arrays for SHAP analysis
def build_shap_inputs(test_ids, df_original, df_test_n, n_patients=500):
    val_cov_orig = np.stack([
        df_original[df_original['ID'] == pid][COV_COLS].iloc[0].values
        for pid in list(test_ids)[:n_patients]
    ]).astype(np.float32)

    val_cov_norm = np.stack([
        df_test_n[df_test_n['ID'] == pid][COV_COLS].iloc[0].values
        for pid in list(test_ids)[:n_patients]
    ]).astype(np.float32)

    return val_cov_orig, val_cov_norm


# Compute SHAP values for each PK parameter and mean concentration
def compute_shap_values(best_model, test_ids, df_original, df_test_n,
                        scaler_dv, n_bg=100, n_samples=100):
    assert HAS_SHAP, "shap package required"
    best_model.eval()

    val_cov_orig, val_cov_norm = build_shap_inputs(
        test_ids, df_original, df_test_n)
    bg_orig = val_cov_orig[:n_bg]
    bg_norm = val_cov_norm[:n_bg]

    # Representative time/amount sequence for mean concentration SHAP
    sample_pid = list(test_ids)[0]
    sample_seq = df_test_n[df_test_n['ID'] == sample_pid].sort_values('TIME')
    t_seq_rep  = torch.FloatTensor(sample_seq['TIME'].values
                                   ).unsqueeze(0).unsqueeze(-1).to(DEVICE)
    a_seq_rep  = torch.FloatTensor(sample_seq['AMT'].values
                                   ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    def pred_CL(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            CL, *_ = best_model.predict_pk_params(xt)
        return CL.cpu().numpy()

    def pred_V1(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            _, V1, *_ = best_model.predict_pk_params(xt)
        return V1.cpu().numpy()

    def pred_Q(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            _, _, Q, *_ = best_model.predict_pk_params(xt)
        return Q.cpu().numpy()

    def pred_V2(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            _, _, _, V2, *_ = best_model.predict_pk_params(xt)
        return V2.cpu().numpy()

    def pred_mean_conc(x):
        with torch.no_grad():
            B   = x.shape[0]
            L   = t_seq_rep.shape[1]
            t_b = t_seq_rep.expand(B, -1, -1)
            a_b = a_seq_rep.expand(B, -1, -1)
            cov_t   = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            dem_exp = cov_t.unsqueeze(1).expand(-1, L, -1)
            inp     = torch.cat([t_b, a_b, dem_exp], dim=-1)
            c       = best_model.concentration_network(inp)
            dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32,
                                   device=DEVICE)
            dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32,
                                   device=DEVICE)
            conc_o  = c * dv_std + dv_mean
        return conc_o.mean(dim=1).squeeze(-1).cpu().numpy()

    print("[SHAP] CL ...")
    shap_CL = _shap.KernelExplainer(pred_CL, bg_orig
                                     ).shap_values(val_cov_orig, nsamples=n_samples)
    print("[SHAP] V1 ...")
    shap_V1 = _shap.KernelExplainer(pred_V1, bg_orig
                                     ).shap_values(val_cov_orig, nsamples=n_samples)
    print("[SHAP] Q ...")
    shap_Q  = _shap.KernelExplainer(pred_Q,  bg_orig
                                     ).shap_values(val_cov_orig, nsamples=n_samples)
    print("[SHAP] V2 ...")
    shap_V2 = _shap.KernelExplainer(pred_V2, bg_orig
                                     ).shap_values(val_cov_orig, nsamples=n_samples)
    print("[SHAP] Mean Concentration ...")
    shap_conc = _shap.KernelExplainer(pred_mean_conc, bg_norm
                                       ).shap_values(val_cov_norm, nsamples=n_samples)

    return (shap_CL, shap_V1, shap_Q, shap_V2, shap_conc,
            val_cov_orig, val_cov_norm)


# Plot SHAP summary for all PK parameters and mean concentration
def plot_shap_summary(shap_CL, shap_V1, shap_Q, shap_V2, shap_conc,
                      val_cov_orig, val_cov_norm,
                      save_name="shap_summary.png"):
    assert HAS_SHAP

    plot_configs = [
        (shap_CL,   val_cov_orig, "SHAP Summary — CL (L/h)"),
        (shap_V1,   val_cov_orig, "SHAP Summary — V1 (L)"),
        (shap_Q,    val_cov_orig, "SHAP Summary — Q (L/h)"),
        (shap_V2,   val_cov_orig, "SHAP Summary — V2 (L)"),
        (shap_conc, val_cov_norm, "(a) SHAP Summary — Mean Concentration"),
    ]

    for i, (shap_vals, cov_data, title) in enumerate(plot_configs):
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.sca(ax)
        _shap.summary_plot(shap_vals, cov_data, feature_names=COV_COLS,
                           plot_type="dot", show=False, color_bar=True)
        ax.set_title(title, fontsize=16, pad=12)
        ax.set_xlabel("SHAP value", fontsize=11)
        plt.tight_layout()
        name = save_name.replace(".png", f"_{['CL','V1','Q','V2','conc'][i]}.png")
        plt.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved → {OUTPUT_DIR / name}")
