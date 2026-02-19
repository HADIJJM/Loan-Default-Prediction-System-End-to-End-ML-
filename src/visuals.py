import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, '..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ======================
# EDA VISUALIZATIONS
# ======================

def plot_target_distribution(y):
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "target_distribution.png"))
    plt.close()


def plot_loan_distribution(df):
    plt.figure(figsize=(7,4))
    sns.histplot(df['loan_amount'], bins=40, kde=True)
    plt.title("Loan Amount Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loan_amount_distribution.png"))
    plt.close()


def plot_income_distribution(df):
    plt.figure(figsize=(7,4))
    sns.histplot(df['income'], bins=40, kde=True)
    plt.title("Income Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "income_distribution.png"))
    plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap (Numeric Features Only)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
    plt.close()



# ======================
# MODEL EVALUATION
# ======================

def plot_confusion_matrix(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"))
    plt.close()


def plot_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"))
    plt.close()


def plot_feature_importance(model, max_features=15):
    xgb.plot_importance(model, max_num_features=max_features)
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"))
    plt.close()
