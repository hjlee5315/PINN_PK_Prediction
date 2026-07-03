import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
from collections import Counter
from src.config import CONFIG, SEED


# Data loading and normalization
def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    print(f"Data shape     : {df.shape}")
    print(f"Unique patients: {df['ID'].nunique()}")

    scaler_continuous = StandardScaler()
    scaler_time       = StandardScaler()
    scaler_amt        = StandardScaler()
    scaler_dv         = StandardScaler()

    df_normalized = df.copy()
    df_normalized[['BW', 'EGFR']] = scaler_continuous.fit_transform(df[['BW', 'EGFR']])
    df_normalized['TIME'] = scaler_time.fit_transform(df[['TIME']])
    df_normalized['AMT']  = scaler_amt.fit_transform(df[['AMT']])
    df_normalized['DV']   = scaler_dv.fit_transform(df[['DV']])

    scalers = {
        'scaler_continuous': scaler_continuous,
        'scaler_time':       scaler_time,
        'scaler_amt':        scaler_amt,
        'scaler_dv':         scaler_dv,
    }
    return df, df_normalized, scalers


# Stratified train/test split by patient-level demographics
def stratified_split(df, df_normalized, scalers):
    patient_df = df.groupby('ID').first().reset_index()[
        ['ID', 'SEX', 'BPS', 'RAAS']
    ].copy()

    patient_df['BW_cat']   = (df.groupby('ID')['BW'].first().values
                               >= df.groupby('ID')['BW'].first().median()).astype(int)
    patient_df['EGFR_cat'] = (df.groupby('ID')['EGFR'].first().values
                               >= df.groupby('ID')['EGFR'].first().median()).astype(int)

    patient_df['strat_key'] = (
        patient_df['SEX'].astype(str) + '_' +
        patient_df['BPS'].astype(str) + '_' +
        patient_df['RAAS'].astype(str) + '_' +
        patient_df['BW_cat'].astype(str) + '_' +
        patient_df['EGFR_cat'].astype(str)
    )

    unique_ids  = patient_df['ID'].values
    strat_keys  = patient_df['strat_key'].values
    key_counts  = Counter(strat_keys)
    strat_keys_safe = np.array([
        k if key_counts[k] >= 2 else 'other' for k in strat_keys
    ])

    train_ids_all, test_ids_arr = train_test_split(
        unique_ids,
        test_size    = CONFIG["test_ratio"],
        random_state = SEED,
        stratify     = strat_keys_safe,
    )
    test_ids      = set(test_ids_arr.tolist())
    train_ids_all = train_ids_all.tolist()

    print(f"Train+Val: {len(train_ids_all)} | Test: {len(test_ids)}")

    df_trainval   = df[df['ID'].isin(train_ids_all)].copy()
    df_test_orig  = df[df['ID'].isin(test_ids)].copy()
    df_trainval_n = df_normalized[df_normalized['ID'].isin(train_ids_all)].copy()
    df_test_n     = df_normalized[df_normalized['ID'].isin(test_ids)].copy()

    # GroupKFold splits
    train_ids_arr = np.array(train_ids_all)
    gkf   = GroupKFold(n_splits=CONFIG["n_folds"])
    folds = [(train_ids_arr[tr].tolist(), train_ids_arr[va].tolist())
             for tr, va in gkf.split(np.zeros(len(train_ids_arr)),
                                     groups=train_ids_arr)]

    return (train_ids_all, test_ids,
            df_trainval, df_test_orig,
            df_trainval_n, df_test_n,
            folds)


# PyTorch Dataset
class PK_Dataset(Dataset):
    def __init__(self, df_norm, df_orig):
        self.data = []
        for pid in df_norm['ID'].unique():
            dn = df_norm[df_norm['ID'] == pid].sort_values('TIME')
            do = df_orig[df_orig['ID'] == pid].sort_values('TIME')
            self.data.append({
                'patient_id':         pid,
                'times_norm':         torch.FloatTensor(dn['TIME'].values),
                'amts_norm':          torch.FloatTensor(dn['AMT'].values),
                'dvs_norm':           torch.FloatTensor(dn['DV'].values),
                'times_orig':         torch.FloatTensor(do['TIME'].values),
                'dvs_orig':           torch.FloatTensor(do['DV'].values),
                'demographics_norm':  torch.FloatTensor([
                    dn['BW'].iloc[0],   dn['EGFR'].iloc[0],
                    dn['SEX'].iloc[0],  dn['BPS'].iloc[0],
                    dn['RAAS'].iloc[0]
                ]),
                'demographics_orig':  torch.FloatTensor([
                    do['BW'].iloc[0],   do['EGFR'].iloc[0],
                    do['SEX'].iloc[0],  do['BPS'].iloc[0],
                    do['RAAS'].iloc[0]
                ]),
                # Observation mask: exclude dosing time points (DV=0)
                'obs_mask':           torch.FloatTensor(
                    (do['DV'].values > 0).astype(float)
                ),
            })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


# Collate function for variable-length sequences
def collate_fn(batch):
    max_len   = max(b['times_norm'].shape[0] for b in batch)
    keys_pad  = ['times_norm', 'amts_norm', 'dvs_norm', 'times_orig', 'obs_mask']
    out       = {k: [] for k in keys_pad}
    out.update({'demographics_norm': [], 'demographics_orig': [],
                'masks': [], 'patient_ids': [], 'dvs_orig': []})

    for b in batch:
        L   = b['times_norm'].shape[0]
        pad = max_len - L
        for k in keys_pad:
            out[k].append(torch.cat([b[k], torch.zeros(pad)]))
        out['dvs_orig'].append(torch.cat([b['dvs_orig'], torch.zeros(pad)]))
        out['demographics_norm'].append(b['demographics_norm'])
        out['demographics_orig'].append(b['demographics_orig'])
        out['masks'].append(torch.cat([torch.ones(L), torch.zeros(pad)]))
        out['patient_ids'].append(b['patient_id'])

    return {
        'times_norm':         torch.stack(out['times_norm']).unsqueeze(-1),
        'amts_norm':          torch.stack(out['amts_norm']).unsqueeze(-1),
        'dvs_norm':           torch.stack(out['dvs_norm']).unsqueeze(-1),
        'dvs_orig':           torch.stack(out['dvs_orig']).unsqueeze(-1),
        'times_orig':         torch.stack(out['times_orig']).unsqueeze(-1),
        'demographics_norm':  torch.stack(out['demographics_norm']),
        'demographics_orig':  torch.stack(out['demographics_orig']),
        'masks':              torch.stack(out['masks']).unsqueeze(-1),
        'obs_mask':           torch.stack(out['obs_mask']).unsqueeze(-1),
        'patient_ids':        out['patient_ids'],
    }


# DataLoader builder
def build_dataloaders(df_trainval_n, df_test_n, df_original, df_test_orig,
                      folds, fold_idx, test_ids):
    tr_ids, va_ids = folds[fold_idx]

    tr_norm = df_trainval_n[df_trainval_n['ID'].isin(tr_ids)]
    va_norm = df_trainval_n[df_trainval_n['ID'].isin(va_ids)]
    tr_orig = df_original[df_original['ID'].isin(tr_ids)]
    va_orig = df_original[df_original['ID'].isin(va_ids)]

    train_dl = DataLoader(
        PK_Dataset(tr_norm, tr_orig),
        batch_size=CONFIG["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(
        PK_Dataset(va_norm, va_orig),
        batch_size=CONFIG["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(
        PK_Dataset(df_test_n, df_test_orig),
        batch_size=CONFIG["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    return train_dl, val_dl, test_dl

