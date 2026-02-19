import os
import joblib
import pandas as pd
import xgboost as xgb
from visuals import *


from sklearn.model_selection import train_test_split
from data_prep import clean_and_refine_data
from features import apply_encoding, scale_data


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'Loan_Default.csv')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Pipeline
df = pd.read_csv(DATA_PATH)
df = clean_and_refine_data(df)

TARGET = 'Status'
leaky = ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']
X_raw = df.drop(columns=[TARGET] + [c for c in leaky if c in df.columns])
y = df[TARGET]

plot_target_distribution(y)
plot_loan_distribution(df)
plot_income_distribution(df)
plot_correlation_heatmap(df)


X_encoded = apply_encoding(X_raw)
plot_correlation_heatmap(X_encoded)
feature_cols = X_encoded.columns.tolist()

# 2. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
X_train_scaled, scaler = scale_data(X_train, is_training=True)

# 3. XGBoost
spw = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(scale_pos_weight=spw, n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

plot_confusion_matrix(model, X_test, y_test)
plot_roc_curve(model, X_test, y_test)
plot_feature_importance(model)

# 4. Save
joblib.dump(model, os.path.join(MODELS_DIR, 'loan_default_xgb_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))

print("Model trained and artifacts saved.")