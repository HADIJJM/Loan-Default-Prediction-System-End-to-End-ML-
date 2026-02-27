import os
import joblib
import pandas as pd
import numpy as np
from src.data_prep import clean_and_refine_data
from src.features import apply_encoding, scale_data
def run_demo_predictions():

    # ————— 1. Load Artifacts —————
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

    # Load the trained model, scaler, and the list of features used during training
    model = joblib.load(os.path.join(MODELS_DIR, 'loan_default_xgb_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    def get_prediction_with_explanation(raw_input_dict):
        """
        Processes a single raw input dictionary and returns the 
        prediction, probability, and top contributing features.
        """
        # Create DataFrame from input dictionary
        df_in = pd.DataFrame([raw_input_dict])
        
        # 1. Clean and handle missing values
        df_clean = clean_and_refine_data(df_in)
        
        # 2. Apply One-Hot Encoding (align with training columns)
        df_encoded = apply_encoding(df_clean, trained_columns=feature_cols)
        
        # 3. Apply Scaling (using the saved scaler)
        X_scaled = scale_data(df_encoded, scaler=scaler, is_training=False) # False To ensure the use of original training scales.
        
        # 4. Generate Probability (class 1: Default/Risky)
        prob = model.predict_proba(X_scaled)[0, 1]
        label = "RISKY ⚠️" if prob > 0.5 else "SAFE ✅"

        # --- Local Feature Contribution Calculation ---
        # multiply the current input values by the global feature importance
        # to understand what drives the decision for THIS specific row.
        feature_importance = model.feature_importances_
        
        current_input_values = df_encoded.iloc[0].values
        local_impact = current_input_values * feature_importance
        
        # Sort indices by highest impact
        indices = np.argsort(local_impact)[::-1]
        
        top_factors = []
        # Select top 3 factors where the feature value is greater than zero
        count = 0
        for i in indices:
            if current_input_values[i] > 0 and count < 3:
                feature_name = feature_cols[i]
                top_factors.append(f"{feature_name}")
                count += 1
                
        return label, prob, top_factors

    # ————— 2. Five Scenarios from Excellent to Rejected —————
    scenarios = [
        {
            "name": "1. Perfect Candidate (Elite)",
            "desc": "High income, zero debt, high asset value.",
            "data": {
                'loan_limit': 'cf', 'Gender': 'Male', 'income': 25000, 
                'loan_amount': 50000, 'property_value': 800000, 
                'Credit_Score': 850, 'LTV': 6.25, 'dtir1': 10.0, 
                'age': '45-54', 'Region': 'North'
            }
        },
        {
            "name": "2. Very Good (Standard Safe)",
            "desc": "Stable income, good credit, reasonable loan.",
            "data": {
                'loan_limit': 'cf', 'Gender': 'Female', 'income': 8000, 
                'loan_amount': 200000, 'property_value': 450000, 
                'Credit_Score': 720, 'LTV': 44.4, 'dtir1': 25.0, 
                'age': '35-44', 'Region': 'South'
            }
        },
        {
            "name": "3. Fair / Borderline (Moderate)",
            "desc": "Medium income, average credit, slightly high LTV.",
            "data": {
                'loan_limit': 'cf', 'Gender': 'Male', 'income': 5000, 
                'loan_amount': 350000, 'property_value': 400000, 
                'Credit_Score': 610, 'LTV': 87.5, 'dtir1': 42.0, 
                'age': '25-34', 'Region': 'West'
            }
        },
        {
            "name": "4. High Risk (Probable Reject)",
            "desc": "Low income for the loan size, poor credit score.",
            "data": {
                'loan_limit': 'non_cf', 'Gender': 'Joint', 'income': 2500, 
                'loan_amount': 450000, 'property_value': 400000, 
                'Credit_Score': 520, 'LTV': 112.5, 'dtir1': 55.0, 
                'age': '55-64', 'Region': 'Central'
            }
        },
        {
            "name": "5. Critical Danger (Absolute Reject)",
            "desc": "Minimal income, very high debt, failed credit history.",
            "data": {
                'loan_limit': 'non_cf', 'Gender': 'Joint', 'income': 1000, 
                'loan_amount': 600000, 'property_value': 350000, 
                'Credit_Score': 400, 'LTV': 171.4, 'dtir1': 75.0, 
                'age': '25-34', 'Region': 'South'
            }
        }
    ]

    # ————— 3. Main Execution Loop —————
    print("\n" + "="*60)
    print("LOAN PREDICTION SYSTEM - DECISION ANALYSIS")
    print("="*60)

    for scenario in scenarios:
        data = scenario["data"]
        label, score, factors = get_prediction_with_explanation(data)
        
        print(f"Scenario    : {scenario['name']}")
        print(f"Decision    : {label}")
        print(f"Certainty   : {score:.2%}")
        print(f"Why this decision?")
        
        for idx, f in enumerate(factors):
            # Attempt to retrieve original raw value for display
            val = data.get(f, "Categorical/Encoded")
            print(f"  {idx+1}. {f:<20} | Impact Value: {val}")
            
        print("-" * 40)

if __name__ == "__main__":
    run_demo_predictions()