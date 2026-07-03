import time
import pickle
import json
import shutil
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.config import CONFIG, DEVICE, OUTPUT_DIR
from src.dataset import PK_Dataset, collate_fn
from src.model import build_model
from src.loss import PINN_PK_Loss
from src.utils import compute_metrics, predict_fold


# Early stopping with best-weight restore
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-6):
        self.patience     = patience
        self.min_delta    = min_delta
        self.best_loss    = None
        self.counter      = 0
        self.best_weights = None
        self.early_stop   = False

    def __call__(self, val_loss, model):
        if (self.best_loss is None or
                val_loss < self.best_loss - self.min_delta):
            self.best_loss    = val_loss
            self.counter      = 0
            self.best_weights = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.best_weights:
                    model.load_state_dict(self.best_weights)


# Single training epoch
def train_epoch(model, loader, criterion, optimizer, scaler_amt, scaler_dv):
    model.train()
    tot = dat = phy = par = 0

    amt_std  = torch.tensor(scaler_amt.scale_[0], dtype=torch.float32, device=DEVICE)
    amt_mean = torch.tensor(scaler_amt.mean_[0],  dtype=torch.float32, device=DEVICE)

    for batch in loader:
        tn       = batch['times_norm'].to(DEVICE)
        an       = batch['amts_norm'].to(DEVICE)
        dn       = batch['dvs_norm'].to(DEVICE)
        to_      = batch['times_orig'].to(DEVICE)
        dm_n     = batch['demographics_norm'].to(DEVICE)
        mask     = batch['masks'].to(DEVICE)
        obs_mask = batch['obs_mask'].to(DEVICE)

        amt_orig = an * amt_std + amt_mean

        optimizer.zero_grad()
        pred = model(tn, an, dm_n)

        # Combine padding mask and observation mask
        combined_mask = mask * obs_mask
        pred_masked   = pred * combined_mask
        dvs_masked    = dn   * combined_mask

        loss, ld, lp, lr = criterion(
            pred_masked, dvs_masked,
            model, to_, amt_orig, dm_n, scaler_dv)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot += loss.item(); dat += ld.item()
        phy += lp.item();   par += lr.item()

    n = len(loader)
    return tot / n, dat / n, phy / n, par / n


# Single validation epoch
def evaluate_epoch(model, loader, scaler_dv):
    model.eval()
    total_mse = total_mae = n_samples = 0

    dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32, device=DEVICE)
    dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        for batch in loader:
            tn   = batch['times_norm'].to(DEVICE)
            an   = batch['amts_norm'].to(DEVICE)
            dn   = batch['dvs_norm'].to(DEVICE)
            dm_n = batch['demographics_norm'].to(DEVICE)
            mask = batch['masks'].to(DEVICE)

            pred        = model(tn, an, dm_n)
            pred_masked = pred * mask
            dvs_masked  = dn   * mask

            po = pred_masked * dv_std + dv_mean
            ao = dvs_masked  * dv_std + dv_mean
            mb = mask.bool()

            total_mse += (torch.mean((po[mb] - ao[mb]) ** 2).item() * mb.sum().item())
            total_mae += (torch.mean(torch.abs(po[mb] - ao[mb])).item() * mb.sum().item())
            n_samples  += mb.sum().item()

    return total_mse / n_samples, total_mae / n_samples


# 5-Fold cross-validation training loop
def run_cv(df_trainval_n, df_original, folds, scalers):
    scaler_amt = scalers['scaler_amt']
    scaler_dv  = scalers['scaler_dv']

    all_metrics   = []
    all_results   = []
    all_histories = []

    for fold_idx, (tr_ids, va_ids) in enumerate(folds):
        print(f"\n{'═' * 65}")
        print(f" Fold {fold_idx + 1}/{CONFIG['n_folds']} | "
              f"Train {len(tr_ids)} / Val {len(va_ids)} patients")
        print(f"{'═' * 65}")

        tr_norm = df_trainval_n[df_trainval_n['ID'].isin(tr_ids)]
        va_norm = df_trainval_n[df_trainval_n['ID'].isin(va_ids)]
        tr_orig = df_original[df_original['ID'].isin(tr_ids)]
        va_orig = df_original[df_original['ID'].isin(va_ids)]

        train_dl = DataLoader(PK_Dataset(tr_norm, tr_orig),
                              batch_size=CONFIG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
        val_dl   = DataLoader(PK_Dataset(va_norm, va_orig),
                              batch_size=CONFIG["batch_size"], shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

        model     = build_model()
        criterion = PINN_PK_Loss(CONFIG["lambda_data"],
                                 CONFIG["lambda_physics"],
                                 CONFIG["lambda_param"])
        optimizer = optim.AdamW(model.parameters(),
                                lr=CONFIG["lr"],
                                weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"],
            min_lr=CONFIG["lr_min"])
        es = EarlyStopping(CONFIG["es_patience"], CONFIG["es_min_delta"])

        history = {"train_total": [], "train_data": [],
                   "train_phys":  [], "train_param": [],
                   "val_total":   []}

        hdr = (f"{'Epoch':>6} | {'TrTot':>8} | {'TrData':>8} | "
               f"{'TrPhys':>8} | {'TrPar':>8} | "
               f"{'ValMSE':>10} | {'ValMAE':>8} | {'LR':>8} | {'ES':>6}")
        print(hdr); print("─" * len(hdr))

        t0 = time.time()
        for epoch in range(1, CONFIG["max_epochs"] + 1):
            tr_tot, tr_dat, tr_phy, tr_par = train_epoch(
                model, train_dl, criterion, optimizer, scaler_amt, scaler_dv)
            val_mse, val_mae = evaluate_epoch(model, val_dl, scaler_dv)
            scheduler.step(val_mse)
            es(val_mse, model)
            cur_lr = optimizer.param_groups[0]['lr']

            history["train_total"].append(tr_tot)
            history["train_data"].append(tr_dat)
            history["train_phys"].append(tr_phy)
            history["train_param"].append(tr_par)
            history["val_total"].append(val_mse)

            if epoch % CONFIG["print_every"] == 0 or epoch == 1:
                def _f(v, w=8):
                    s = f"{v:.4f}"
                    return (s[:w - 3] + "...") if len(s) > w else s.rjust(w)
                print(f"{epoch:>6} | {_f(tr_tot)} | {_f(tr_dat)} | "
                      f"{_f(tr_phy)} | {_f(tr_par)} | "
                      f"{_f(val_mse, 10)} | {_f(val_mae)} | "
                      f"{cur_lr:.2e} | "
                      f"{es.counter:>3}/{CONFIG['es_patience']}"
                      f"  [{time.time() - t0:.0f}s]")

            if es.early_stop:
                print(f"\n  Early Stop @ epoch {epoch} (best={es.best_loss:.6f})")
                break

        all_histories.append(history)

        # Save fold model
        torch.save(model.state_dict(),
                   OUTPUT_DIR / f"fold_{fold_idx + 1}_model.pt")

        fold_df   = predict_fold(model, val_dl, scaler_dv)
        fold_df_t = fold_df[fold_df["TIME"] > 0]
        metrics   = compute_metrics(fold_df_t["DV_pred"].values,
                                    fold_df_t["DV_obs"].values)
        print(f"\n  [Fold {fold_idx + 1}] Metrics:")
        for k, v in metrics.items():
            print(f"    {k:30s}: {v}")

        all_metrics.append(metrics)
        all_results.append(fold_df)

    print("\n5-Fold CV complete.")
    return all_metrics, all_results, all_histories


# Select best fold and load its model
def select_best_model(all_metrics, all_histories):
    best_fold_idx = int(np.argmax([m["R²"] for m in all_metrics]))
    best_fold_r2  = all_metrics[best_fold_idx]["R²"]
    print(f"Best fold: Fold {best_fold_idx + 1}  (Val R² = {best_fold_r2:.4f})")

    from src.model import build_model
    best_model = build_model()
    best_model.load_state_dict(
        torch.load(OUTPUT_DIR / f"fold_{best_fold_idx + 1}_model.pt",
                   map_location=DEVICE))
    best_model.eval()

    # Plot training curves for best fold
    hist   = all_histories[best_fold_idx]
    epochs = range(1, len(hist["train_total"]) + 1)
    best_ep = int(np.argmin(hist["val_total"])) + 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — Best Fold {best_fold_idx + 1} "
                 f"(Val R²={best_fold_r2:.4f})", fontsize=13, y=1.02)

    # Total loss
    ax = axes[0]
    ax.plot(epochs, hist["train_total"], lw=1.5, color="steelblue",  label="Train Total")
    ax.plot(epochs, hist["val_total"],   lw=1.5, color="darkorange", linestyle="--",
            label="Val Total")
    ax.axvline(best_ep, color="gray", lw=1.0, linestyle=":", label=f"Best epoch {best_ep}")
    ax.set_title("Total Loss (Train vs Val)")
    ax.set_xlabel("Epoch"); ax.set_yscale("log")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Loss components
    ax = axes[1]
    ax.plot(epochs, hist["train_data"],  lw=1.5, color="steelblue", label="Data loss")
    ax.plot(epochs, hist["train_phys"],  lw=1.5, color="coral",     label="Physics loss")
    ax.plot(epochs, hist["train_param"], lw=1.5, color="seagreen",  label="Param loss")
    ax.axvline(best_ep, color="gray", lw=1.0, linestyle=":")
    ax.set_title("Train Loss Components")
    ax.set_xlabel("Epoch"); ax.set_yscale("log")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Linear scale
    ax = axes[2]
    ax.plot(epochs, hist["train_total"], lw=1.5, color="steelblue",  label="Train Total")
    ax.plot(epochs, hist["val_total"],   lw=1.5, color="darkorange", linestyle="--",
            label="Val Total")
    ax.axvline(best_ep, color="gray", lw=1.0, linestyle=":")
    ax.set_title("Total Loss — Linear Scale")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    return best_model, best_fold_idx, best_fold_r2


# Save all artifacts needed for kernel-free inference
def save_artifacts(best_fold_idx, best_fold_r2, all_metrics,
                   train_ids_all, test_ids, folds, scalers):
    # Scalers
    with open(OUTPUT_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Train/test split info
    split_info = {
        'train_ids_all': train_ids_all,
        'test_ids':      list(test_ids),
        'folds':         folds,
        'best_fold_idx': best_fold_idx,
        'best_fold_r2':  float(best_fold_r2),
    }
    with open(OUTPUT_DIR / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # CV metrics
    with open(OUTPUT_DIR / 'all_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)

    # Best model copy
    shutil.copy(OUTPUT_DIR / f'fold_{best_fold_idx + 1}_model.pt',
                OUTPUT_DIR / 'best_model.pt')

    print(f"Artifacts saved to {OUTPUT_DIR}")


# Load saved artifacts for inference without retraining
def load_artifacts():
    import pickle, json
    with open(OUTPUT_DIR / 'scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open(OUTPUT_DIR / 'split_info.json', 'r') as f:
        split_info = json.load(f)
    with open(OUTPUT_DIR / 'all_metrics.pkl', 'rb') as f:
        all_metrics = pickle.load(f)

    best_fold_idx = split_info['best_fold_idx']

    from src.model import build_model
    best_model = build_model()
    best_model.load_state_dict(
        torch.load(OUTPUT_DIR / 'best_model.pt', map_location=DEVICE))
    best_model.eval()

    print(f"Artifacts loaded. Best fold: Fold {best_fold_idx + 1} "
          f"(R²={split_info['best_fold_r2']:.4f})")
    return best_model, scalers, split_info, all_metrics
