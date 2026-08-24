import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.models.train import evaluate_model

def test_evaluate_model():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    assert "Accuracy" in metrics
    assert "ROC-AUC" in metrics
    assert metrics["Accuracy"] == 0.75
    assert len(metrics["Confusion Matrix"]) == 2
