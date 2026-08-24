import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def train_models():
    """Load data, preprocess, train models, and save them."""
    # 1. Load data
    df = load_raw_data()
    
    # 2. Train-test split
    # We split before fitting the preprocessor to prevent data leakage
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # 3. Preprocess
    X_train, y_train, preprocessor = preprocess_data(df_train, fit=True, save_path="models/preprocessor.joblib")
    X_test, y_test, _ = preprocess_data(df_test, fit=False, preprocessor=preprocessor)
    
    # 4. Train Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_preds, lr_probs)
    logger.info(f"LR Metrics: {lr_metrics}")
    
    joblib.dump(lr_model, "models/logistic_regression.joblib")
    joblib.dump(lr_metrics, "models/lr_metrics.joblib")
    
    # 5. Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_preds, xgb_probs)
    logger.info(f"XGB Metrics: {xgb_metrics}")
    
    joblib.dump(xgb_model, "models/xgboost.joblib")
    joblib.dump(xgb_metrics, "models/xgb_metrics.joblib")
    
    # Also save the test sets for explainability / insights if needed
    os.makedirs('data/processed', exist_ok=True)
    df_test.to_csv('data/processed/test_data.csv', index=False)
    
    logger.info("Training complete. Models and metrics saved.")

if __name__ == "__main__":
    train_models()
