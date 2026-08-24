import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe."""
    df = df.copy()
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # TotalCharges is object, contains ' ' for tenure=0.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing TotalCharges with 0 (since tenure is 0)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
    # Map Churn to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_preprocessor(df: pd.DataFrame):
    """
    Creates and returns a sklearn ColumnTransformer for preprocessing.
    """
    # Exclude target from features
    feature_cols = [c for c in df.columns if c != 'Churn']
    
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric pipeline: Impute missing with median (just in case), then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing with constant, then OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor, numeric_features, categorical_features
    
def preprocess_data(df: pd.DataFrame, fit: bool = True, save_path: str = None, preprocessor=None):
    """
    Clean data, apply preprocessing pipeline, and return X, y.
    If fit=True, fits the preprocessor and optionally saves it.
    If fit=False, uses the provided preprocessor.
    """
    df_clean = clean_data(df)
    
    y = df_clean['Churn'] if 'Churn' in df_clean.columns else None
    X = df_clean.drop(columns=['Churn']) if 'Churn' in df_clean.columns else df_clean
    
    if fit:
        preprocessor, num_cols, cat_cols = get_preprocessor(df_clean)
        logger.info("Fitting preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(preprocessor, save_path)
            logger.info(f"Preprocessor saved to {save_path}")
            
        return X_processed, y, preprocessor
    else:
        if preprocessor is None:
            raise ValueError("preprocessor must be provided if fit=False")
        logger.info("Transforming data with existing preprocessor...")
        X_processed = preprocessor.transform(X)
        return X_processed, y, preprocessor
