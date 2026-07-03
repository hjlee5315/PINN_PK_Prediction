import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.config import CONFIG, OUTPUT_DIR, DEVICE
from src.dataset import (load_and_preprocess, stratified_split,
                          PK_Dataset, collate_fn)
from src.train import (run_cv, select_best_model,
                        save_artifacts, load_artifacts)
from src.utils import compute_metrics, predict_fold
from evaluate.gof           import plot_gof_wres
from evaluate.pk_distribution import plot_pk_distribution
from evaluate.shap_analysis import compute_shap_values, plot_shap_summary
from evaluate.permutation   import permutation_importance, plot_permutation_importance


def train_pipeline():
    # Data loading and preprocessing
    df, df_normalized, scalers = load_and_preprocess("path/to/dataset.csv")

    # Stratified train/test split and fold construction
    (train_ids_all, test_ids,
     df_trainval, df_test_orig,
     df_trainval_n, df_test_n,
     folds) = stratified_split(df, df_normalized, scalers)

    # 5-Fold cross-validation
    all_metrics, all_results, all_histories = run_cv(
        df_trainval_n, df, folds, scalers)

    # Best fold selection
    best_model, best_fold_idx, best_fold_r2 = select_best_model(
        all_metrics, all_histories)

    # Save all artifacts
    save_artifacts(best_fold_idx, best_fold_r2, all_metrics,
                   train_ids_all, test_ids, folds, scalers)

    return (best_model, scalers, test_ids,
            df_test_n, df_test_orig, df, all_results)


def evaluate_pipeline(best_model, scalers, test_ids,
                      df_test_n, df_test_orig, df_original, all_results):
    scaler_dv = scalers['scaler_dv']

    # Test set prediction
    test_ds = PK_Dataset(df_test_n, df_test_orig)
    test_dl = DataLoader(test_ds, batch_size=CONFIG["batch_size"],
                         shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_df   = predict_fold(best_model, test_dl, scaler_dv)
    test_df_t = test_df[test_df["TIME"] > 0]

    print("\nTest Set Metrics:")
    for k, v in compute_metrics(test_df_t["DV_pred"].values,
                                test_df_t["DV_obs"].values).items():
        print(f"  {k:30s}: {v}")

    # Save predictions
    test_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Evaluation plots
    all_cv_df = pd.concat(all_results, ignore_index=True)

    plot_gof_wres(test_df, save_name="gof_wres.png")
    plot_pk_distribution(test_df, save_name="pk_distribution.png")

    # SHAP analysis
    (shap_CL, shap_V1, shap_Q, shap_V2, shap_conc,
     val_cov_orig, val_cov_norm) = compute_shap_values(
        best_model, test_ids, df_original, df_test_n, scaler_dv)
    plot_shap_summary(shap_CL, shap_V1, shap_Q, shap_V2, shap_conc,
                      val_cov_orig, val_cov_norm)

    # Permutation importance
    importance_df = permutation_importance(best_model, test_dl, scaler_dv)
    plot_permutation_importance(importance_df)

    return test_df


if __name__ == "__main__":
    # ── Option A: Full pipeline (train + evaluate) ────────────────
    (best_model, scalers, test_ids,
     df_test_n, df_test_orig, df_original, all_results) = train_pipeline()

    evaluate_pipeline(best_model, scalers, test_ids,
                      df_test_n, df_test_orig, df_original, all_results)

