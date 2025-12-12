# data_loader.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch

# Define columns to drop (normalized)
COLS_TO_DROP = {
    'hr_time_series',
    'resp_time_series',
    'stress_time_series',
    'act_activetime',
    'day',              # ← explicitly removed to match test set
    'unnamed: 0'
}

def load_client_data(
    client_id: int,
    return_preprocessors: bool = False,
    batch_size: int = 32
):
    """Load, clean, and preprocess client data.
    
    Args:
        client_id: Client ID (0-8)
        return_preprocessors: If True, also return scaler and imputer
        batch_size: Batch size for DataLoaders

    Returns:
        If return_preprocessors=False:
            (train_loader, val_loader, input_dim)
        If return_preprocessors=True:
            (train_loader, val_loader, input_dim, scaler, imputer)
    """
    client_dir = os.path.join("CSV_train", f"group{client_id}")
    dfs = []
    
    for filename in os.listdir(client_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(client_dir, filename)
            df = pd.read_csv(
                filepath,
                sep=';',
                quoting=3,
                quotechar=None,
                escapechar=None,
                header=0,
                on_bad_lines='skip',
                engine='python'
            )
            df.columns = df.columns.str.strip().str.lower()
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)

    # Drop unwanted columns (case-insensitive due to .lower() above)
    cols_to_drop = [col for col in COLS_TO_DROP if col in combined.columns]
    combined = combined.drop(columns=cols_to_drop)

    if 'label' not in combined.columns:
        raise ValueError(f"'label' missing in client {client_id}. Columns: {list(combined.columns)}")

    # Separate features and target
    feature_cols = [col for col in combined.columns if col != 'label']
    X = combined[feature_cols]
    y = combined[['label']]

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    y_imp = SimpleImputer(strategy='median').fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Convert to tensors
    X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(y_imp.astype(np.float32))

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    input_dim = X_scaled.shape[1]
    print(f"[Client {client_id}] Using {input_dim} features")

    if return_preprocessors:
        return train_loader, val_loader, input_dim, scaler, imputer
    else:
        return train_loader, val_loader, input_dim