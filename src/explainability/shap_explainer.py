import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model_type="xgboost"):
        self.preprocessor = joblib.load("models/preprocessor.joblib")
        if model_type == "logistic_regression":
            self.model = joblib.load("models/logistic_regression.joblib")
        else:
            self.model = joblib.load("models/xgboost.joblib")
            
        self.model_type = model_type
        
        # Get feature names from preprocessor
        numeric_features = self.preprocessor.transformers_[0][2]
        categorical_transformer = self.preprocessor.transformers_[1][1]
        
        # We need the categories from OneHotEncoder
        ohe = categorical_transformer.named_steps['onehot']
        cat_features_in = self.preprocessor.transformers_[1][2]
        
        if hasattr(ohe, 'get_feature_names_out'):
            categorical_features_out = ohe.get_feature_names_out(cat_features_in)
        else:
            # Fallback for older scikit-learn
            categorical_features_out = ohe.get_feature_names(cat_features_in)
            
        self.feature_names = numeric_features + list(categorical_features_out)
        
        # Initialize shap explainer
        if model_type == "xgboost":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # For Logistic Regression, we use LinearExplainer with independent masker
            # For simplicity, if we don't have background data at init, we can use an Independent masker
            self.explainer = shap.LinearExplainer(self.model, shap.maskers.Independent(np.zeros((1, len(self.feature_names)))))
            
    def get_shap_values(self, X_processed):
        """Get SHAP values for preprocessed data."""
        if self.model_type == "xgboost":
            shap_values = self.explainer.shap_values(X_processed)
        else:
            shap_values = self.explainer.shap_values(X_processed)
        return shap_values
        
    def plot_summary(self, X_processed, max_display=10, show=False):
        """Returns a summary plot matplotlib figure."""
        shap_values = self.get_shap_values(X_processed)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_processed, feature_names=self.feature_names, max_display=max_display, show=show)
        return fig
        
    def explain(self, X_processed_single):
        """Returns the Explanation object for a single instance."""
        explanation = self.explainer(X_processed_single)
        if hasattr(explanation, 'values') and len(explanation.values.shape) > 1:
            return explanation[0]
        return explanation

    def plot_waterfall(self, X_processed_single, show=False):
        """Returns a waterfall plot for a single instance."""
        explanation = self.explain(X_processed_single)
        fig = plt.figure()
        shap.plots.waterfall(explanation, show=show)
        return fig
