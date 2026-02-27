import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.visuals import *
from src.data_prep import clean_and_refine_data
from src.features import apply_encoding, scale_data

def train_pipeline():
    # ————— Paths —————
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'Loan_Default.csv')
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ————— 1. Load & Prepare Data —————
    df = pd.read_csv(DATA_PATH)
    df = clean_and_refine_data(df)

    TARGET = 'Status'
    leaky_features = ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']

    X_raw = df.drop(columns=[TARGET] + [c for c in leaky_features if c in df.columns])
    y = df[TARGET]


    # ————— 2. EDA Visualizations —————
    plot_target_distribution(y)
    plot_loan_distribution(df)
    plot_income_distribution(df)
    plot_correlation_heatmap(df)


    # ————— 3. Feature Engineering —————
    X_encoded = apply_encoding(X_raw)
    plot_correlation_heatmap(X_encoded)
    feature_cols = X_encoded.columns.tolist()

    # ————— 4. Train-Test Split & Scaling —————
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_scaled, scaler = scale_data(X_train, is_training=True)
    X_test_scaled = scale_data(X_test, is_training=False, scaler=scaler)

    # ————— 5. Train XGBoost Model —————
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)


    # ————— 6. Model Evaluation —————
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n———————————————————————————— ")
    print(" CLASSIFICATION REPORT ")
    print("n—————————————————————————————\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}\n")

    # ————— 7. Evaluation Plots —————
    plot_confusion_matrix(model, X_test_scaled, y_test)
    plot_roc_curve(model, X_test_scaled, y_test)
    plot_feature_importance(model)

    # ————— 8. Save Artifacts —————
    joblib.dump(model, os.path.join(MODELS_DIR, 'loan_default_xgb_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    print("Model trained, evaluated, and artifacts saved successfully.")

if __name__ == "__main__":
    train_pipeline()