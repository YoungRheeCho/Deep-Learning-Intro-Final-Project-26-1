import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KNOWN_ATTACKS = [
    'DDoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris',
    'PortScan', 'FTP-Patator', 'SSH-Patator'
]

# Labels in CSV use replacement char � instead of en-dash
UNKNOWN_ATTACKS = [
    'Bot',
    'Web Attack � Brute Force',
    'Web Attack � XSS',
    'Web Attack � Sql Injection',
    'Infiltration',
]


def load_data(data_dir='archive'):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not dfs:
        raise RuntimeError("No CSV files loaded.")
    data = pd.concat(dfs, ignore_index=True)
    return data


def preprocess(data_dir='archive'):
    print("Loading CSV files...")
    df = load_data(data_dir)

    # Strip column names
    df.columns = df.columns.str.strip()

    # Unify label column
    label_col = [c for c in df.columns if c.strip().lower() == 'label']
    if not label_col:
        raise ValueError("Label column not found.")
    df.rename(columns={label_col[0]: 'Label'}, inplace=True)

    # Replace inf/-inf with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with NaN. Remaining: {len(df)}")

    # Strip label values
    df['Label'] = df['Label'].str.strip()

    # Filter available unknown attacks
    available_unknown = [a for a in UNKNOWN_ATTACKS if a in df['Label'].unique()]
    missing = set(UNKNOWN_ATTACKS) - set(available_unknown)
    if missing:
        print(f"[WARN] Missing unknown attacks (skipped): {len(missing)} types")

    # Split known / unknown
    benign_mask = df['Label'] == 'BENIGN'
    known_mask = df['Label'].isin(KNOWN_ATTACKS)
    unknown_mask = df['Label'].isin(available_unknown)

    df_known = df[benign_mask | known_mask].copy()
    df_unknown = df[unknown_mask].copy()

    print(f"Known data: {len(df_known)} rows | Unknown data: {len(df_unknown)} rows")
    known_dist = df_known['Label'].value_counts().to_dict()
    print("Known label distribution:")
    for k, v in known_dist.items():
        print(f"  {k.encode('ascii', errors='replace').decode()}: {v}")
    unknown_dist = df_unknown['Label'].value_counts().to_dict()
    print("Unknown label distribution:")
    for k, v in unknown_dist.items():
        print(f"  {k.encode('ascii', errors='replace').decode()}: {v}")

    # Feature columns
    feature_cols = [c for c in df.columns if c != 'Label']

    # Binary labels: BENIGN=0, Attack=1
    df_known['binary_label'] = (df_known['Label'] != 'BENIGN').astype(int)
    df_unknown['binary_label'] = 1  # all unknown are attacks

    X_known = df_known[feature_cols].values.astype(np.float32)
    y_known = df_known['binary_label'].values

    X_unknown = df_unknown[feature_cols].values.astype(np.float32)
    y_unknown = df_unknown['binary_label'].values

    # Train/Val/Test split on known data (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_known, y_known, test_size=0.30, random_state=42, stratify=y_known
    )
    X_val, X_test_known, y_val, y_test_known = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # StandardScaler fit on train set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_known = scaler.transform(X_test_known)
    X_unknown = scaler.transform(X_unknown)

    print(f"\nSplit shapes:")
    print(f"  X_train:        {X_train.shape}, attack ratio: {y_train.mean():.3f}")
    print(f"  X_val:          {X_val.shape},   attack ratio: {y_val.mean():.3f}")
    print(f"  X_test_known:   {X_test_known.shape}, attack ratio: {y_test_known.mean():.3f}")
    print(f"  X_test_unknown: {X_unknown.shape}, attack ratio: {y_unknown.mean():.3f}")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test_known, y_test_known,
        X_unknown, y_unknown,
        scaler
    )


if __name__ == '__main__':
    preprocess()
