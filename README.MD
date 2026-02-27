# Loan Default Prediction System (End-to-End ML)

This project is a professional end-to-end machine learning system designed to predict the probability of loan default. It leverages **XGBoost** for high-performance classification and follows a modular pipeline architecture, separating data preprocessing, feature engineering, training, evaluation, visualization, and inference logic.

The system now operates through a **unified entry point** (`main.py`), making it more robust and easier to deploy or integrate.

---

## Key Features
* Explainable AI (XAI): The system explains each prediction by identifying the Top 3 contributing features behind the decision.
* End-to-End ML Pipeline: Full workflow from raw data cleaning to final model evaluation.
* Modular Package Design: Organized as a Python package using `src/__init__.py` for better maintainability.
* Automated Visualization: Automatically generates and updates professional evaluation plots.
* Production-Oriented: Model artifacts, encoders, and scalers are persisted for consistent inference.

---

## Project Structure

Loan-Default-Prediction-System/
│
├── main.py                 # Main Entry Point (Run the system from here)
│
├── data/
│   └── Loan_Default.csv    # Raw dataset
│
├── models/                 # Saved ML Artifacts
│   ├── loan_default_xgb_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── reports/
│   └── figures/            # Auto-generated plots (ROC, Confusion Matrix, etc.)
│
├── src/                    # Source Code Package
│   ├── __init__.py         # Makes 'src' a package
│   ├── train.py            # Training & evaluation logic
│   ├── predict.py          # Inference & XAI reasoning
│   ├── data_prep.py        # Data cleaning
│   ├── features.py         # Encoding & scaling logic
│   └── visuals.py          # Visualization functions
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

---

## How to Use

### 1. Install Dependencies
Ensure Python 3.x is installed, then run:

pip install -r requirements.txt

### 2. Run the System
You can now run the entire process (or specific parts) via the main script:

python main.py

---

## Understanding the Output

When running the system, it provides detailed decision analysis:

- Decision: SAFE or RISKY ⚠️
- Certainty: The probability percentage of default.
- Reasoning: The top 3 features that influenced the model's decision.

### Example Output:

Scenario: 3. Fair / Borderline  
Decision: RISKY ⚠️  
Certainty: 72.31%

Why this decision?  
1. loan_amount  
2. LTV  
3. income

---

## Model Evaluation & Reports

All evaluation figures are automatically generated and saved inside:
reports/figures/

Including:

- Correlation Heatmaps
- Confusion Matrix & ROC Curve
- Feature Importance (XGBoost)
- Distribution Plots (Loan Amount, Income)

---

## Tech Stack

- Language: Python 3.x
- Core Libraries: Pandas, NumPy, Scikit-Learn
- ML Algorithm: XGBoost
- Visualization: Matplotlib, Seaborn

---

## Project Goals

- Build a production-ready ML system for financial risk assessment.
- Provide interpretable predictions using Explainable AI techniques.
- Deliver a fully automated and modular pipeline.
