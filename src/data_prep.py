import pandas as pd
import numpy as np

def clean_and_refine_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and numeric refinements."""
    # Numerical columns
    num_cols = [
        'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
        'property_value', 'income', 'LTV', 'dtir1', 'term', 
        'loan_amount', 'Credit_Score'
    ]
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    cat_cols = [
        'loan_limit', 'approv_in_adv', 'Gender', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by', 'credit_type',
        'co_applicant_credit_type', 'age', 'submission_of_application',
        'Region', 'Security_Type'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if 'Credit_Score' in df.columns:
        df['Credit_Score'] = df['Credit_Score'].clip(upper=900)
    
    if 'income' in df.columns:
        df['income'] = np.log1p(df['income'])

    return df