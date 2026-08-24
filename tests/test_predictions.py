import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock

# We need to mock the joblib load to not require actual models during testing if we want fast tests,
# OR we can just test if the models are loaded properly. I will do a basic mock to verify logic.
from src.models.predict import ChurnPredictor

@patch('joblib.load')
def test_churn_predictor_initialization(mock_load):
    mock_preprocessor = MagicMock()
    mock_model = MagicMock()
    
    # joblib.load is called twice (preprocessor, then model)
    mock_load.side_effect = [mock_preprocessor, mock_model]
    
    predictor = ChurnPredictor(model_type="xgboost")
    
    assert predictor.preprocessor == mock_preprocessor
    assert predictor.model == mock_model
    
@patch('joblib.load')
def test_churn_predictor_predict(mock_load):
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.array([[1, 2, 3]])
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    mock_load.side_effect = [mock_preprocessor, mock_model]
    
    predictor = ChurnPredictor()
    
    df_input = pd.DataFrame({'tenure': [5]})
    preds, probs = predictor.predict(df_input)
    
    assert preds[0] == 1
    assert probs[0] == 0.8
