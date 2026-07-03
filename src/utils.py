import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from src.config import DEVICE


# Concordance correlation coefficient
def concordance_correlation(y_true, y_pred):
    mu_t = np.mean(y_true); mu_p = np.mean(y_pred)
    var_t = np.var(y_true); var_p = np.var(y_pred)
    cov   = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return (2 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2)


# Compute prediction metrics (MSE, MAE, RMSE, R², bias, MAPE, Pearson r, CCC)
def compute_metrics(y_pred, y_true):
    p = y_pred.flatten(); t = y_true.flatten()
    mask = t > 0; p, t = p[mask], t[mask]

    mse   = float(np.mean((p - t) ** 2))
    mae   = float(np.mean(np.abs(p - t)))
    rmse  = float(np.sqrt(mse))
    r2    = float(np.corrcoef(p, t)[0, 1] ** 2)
    bias  = float(np.mean((p - t) / t) * 100)
    mape  = float(np.mean(np.abs(p - t) / t) * 100)
    pr, _ = pearsonr(p, t)
    ccc   = concordance_correlation(t, p)

    return {
        "MSE (mg/L)²":            round(mse,  4),
        "MAE (mg/L)":             round(mae,  4),
        "RMSE (mg/L)":            round(rmse, 4),
        "R²":                     round(r2,   4),
        "Prediction Bias (MPE%)": round(bias, 2),
        "MAPE (%)":               round(mape, 2),
        "Pearson r":              round(float(pr), 4),
        "CCC":                    round(ccc,  4),
        "N":                      int(np.sum(mask)),
    }


# Generate predictions DataFrame from a DataLoader
def predict_fold(model, loader, scaler_dv):
    dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32, device=DEVICE)
    dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32, device=DEVICE)
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            tn    = batch['times_norm'].to(DEVICE)
            an    = batch['amts_norm'].to(DEVICE)
            dm_n  = batch['demographics_norm'].to(DEVICE)
            dm_o  = batch['demographics_orig'].to(DEVICE)
            mask  = batch['masks'].cpu().numpy().astype(bool)
            to_np = batch['times_orig'].numpy()
            dv_np = batch['dvs_orig'].numpy()
            pids  = batch['patient_ids']

            pred   = model(tn, an, dm_n)
            conc_o = (pred * dv_std + dv_mean).cpu().numpy()

            CL, V1, Q, V2, *_ = model.predict_pk_params(dm_o)
            CL_np = CL.cpu().numpy()
            V1_np = V1.cpu().numpy()
            Q_np  = Q.cpu().numpy()
            V2_np = V2.cpu().numpy()

            for i, pid in enumerate(pids):
                for t in range(pred.shape[1]):
                    if mask[i, t, 0]:
                        rows.append({
                            "ID":      pid,
                            "TIME":    float(to_np[i, t, 0]),
                            "DV_obs":  float(dv_np[i, t, 0]),
                            "DV_pred": float(conc_o[i, t, 0]),
                            "CL":      float(CL_np[i]),
                            "V1":      float(V1_np[i]),
                            "Q":       float(Q_np[i]),
                            "V2":      float(V2_np[i]),
                        })

    return pd.DataFrame(rows)
