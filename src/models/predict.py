import joblib
import pandas as pd
import numpy as np
from src.data.preprocess import clean_data

class ChurnPredictor:
    def __init__(self, model_type="xgboost"):
        self.preprocessor = joblib.load("models/preprocessor.joblib")
        if model_type == "logistic_regression":
            self.model = joblib.load("models/logistic_regression.joblib")
        else:
            self.model = joblib.load("models/xgboost.joblib")
            
    def predict(self, input_data: pd.DataFrame):
        """
        Takes raw input DataFrame, preprocesses it, and returns predictions and probabilities.
        """
        # We need to make sure input_data matches the expected structure.
        df_clean = clean_data(input_data)
        
        # In case the input does not have the 'Churn' column, clean_data just returns it.
        # But our preprocessor was fit on X without 'Churn'.
        if 'Churn' in df_clean.columns:
            X = df_clean.drop(columns=['Churn'])
        else:
            X = df_clean
            
        X_processed = self.preprocessor.transform(X)
        
        preds = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)[:, 1]
        
        return preds, probs
