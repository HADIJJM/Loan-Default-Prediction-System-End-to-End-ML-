import pandas as pd
import joblib
import os

def apply_encoding(df: pd.DataFrame, trained_columns=None) -> pd.DataFrame:
    """Standardize One-Hot Encoding across training and prediction."""
    df_encoded = pd.get_dummies(df)
    
    if trained_columns:
        # Match training columns
        for col in trained_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[trained_columns]
    
    return df_encoded

def scale_data(X, scaler=None, is_training=True):
    """Apply StandardScaler and save it if training."""
    if is_training:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return scaler.transform(X)