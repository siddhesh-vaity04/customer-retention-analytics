import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import clean_data, get_preprocessor

def test_clean_data():
    df = pd.DataFrame({
        'customerID': ['123', '456'],
        'TotalCharges': ['100.5', ' '],
        'Churn': ['Yes', 'No'],
        'tenure': [5, 0]
    })
    
    df_clean = clean_data(df)
    
    assert 'customerID' not in df_clean.columns
    assert df_clean['Churn'].iloc[0] == 1
    assert df_clean['Churn'].iloc[1] == 0
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])
    assert df_clean['TotalCharges'].iloc[1] == 0.0
    
def test_get_preprocessor():
    df = pd.DataFrame({
        'tenure': [5, 10],
        'MonthlyCharges': [50.5, 60.0],
        'Contract': ['Month-to-month', 'One year'],
        'Churn': [1, 0]
    })
    
    preprocessor, num_features, cat_features = get_preprocessor(df)
    
    assert 'tenure' in num_features
    assert 'Contract' in cat_features
    assert 'Churn' not in num_features
    assert 'Churn' not in cat_features
