# Loan Default Prediction System (End-to-End ML) 🏦

This project is a professional **end-to-end machine learning system** designed to predict the probability of loan default. It leverages **XGBoost** for high-performance classification and follows a **modular pipeline architecture**, separating data preprocessing, feature engineering, training, evaluation, visualization, and inference logic.

The system not only provides predictions, but also delivers **Explainable AI (XAI)** insights, enabling transparent and interpretable decision-making.

---

## Key Features

- **Explainable AI (XAI)**  
  The system explains each prediction by identifying the **Top 3 contributing features** behind the decision.

- **End-to-End ML Pipeline**  
  Full pipeline including data preprocessing, feature engineering, training, evaluation, visualization, and prediction.

- **Modular Architecture**  
  Clean separation between training, inference, preprocessing, and visualization logic.

- **Automated Visualization & Reporting**  
  Automatically generates professional evaluation plots and saves them for documentation.

- **Production-Oriented Design**  
  Model artifacts, encoders, and scalers are persisted to ensure **consistent inference in deployment**.

---

## Project Structure 📂

Project_Root/
│
├── data/               # Raw dataset (Loan_Default.csv)
├── models/             # Trained model & preprocessing artifacts (.pkl)
├── reports/
│   └── figures/        # Auto-generated plots & evaluation figures
├── src/
│   ├── data_prep.py    # Data cleaning and missing value handling
│   ├── features.py     # Encoding & feature scaling logic
│   ├── visuals.py      # Visualization and reporting functions
│   ├── train.py        # Training pipeline and artifact generation
│   └── predict.py      # Inference pipeline with XAI explanations
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

---

# 🚀 How to Use

## 1. Install Dependencies
Ensure Python 3.x is installed, then run:

pip install -r requirements.txt

---

## 2. Train the Model (Optional if models already exist)
To retrain the model using the dataset located in /data:

python src/train.py

Note: This generates trained artifacts (.pkl) inside the models/ directory and saves evaluation figures inside reports/figures/.

---

## 3. Run Predictions & Decision Analysis
To test the model with predefined financial scenarios (from Elite to Critical Risk):

python src/predict.py

---

# 📊 Understanding the Output

When running predict.py, the system provides:

Decision: SAFE or RISKY  
Certainty: The probability percentage of default  
Reasoning: A list of the top 3 features that influenced the prediction  

Example Output:

Scenario    : 3. Fair / Borderline  
Decision    : RISKY ⚠️  
Certainty   : 72.31%  
Why this decision?  
1. loan_amount  
2. LTV  
3. income  

---

# 📈 Model Evaluation & Visual Reports

All important evaluation figures are automatically generated and saved inside:

reports/figures/

Including:

- Target Distribution  
- Loan Amount Distribution  
- Income Distribution  
- Correlation Heatmap  
- Confusion Matrix  
- ROC Curve  
- Feature Importance  

These visualizations allow detailed performance analysis and full project documentation.

---

# 🛠️ Tech Stack

Language: Python 3.x  
Core Libraries: Pandas, NumPy, Scikit-Learn  
ML Algorithm: XGBoost  
Visualization: Matplotlib, Seaborn  
Serialization: Joblib  

---

# 🎯 Project Goals

- Build a **production-ready ML system** for financial risk assessment.
- Provide **interpretable predictions** using explainable AI techniques.
- Deliver a **fully automated pipeline** from raw data to final decision.

---

# 🏆 Project Level

This project follows **industry-level ML engineering practices** and is suitable for:

- Final Year Projects (FYP)
- Professional Data Science Portfolios
- Applied Machine Learning Case Studies
- Production-Oriented ML System Design

---

Developed as a complete **End-to-End Machine Learning System**.
