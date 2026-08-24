import os
import pandas as pd
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_data(raw_data_path: str = "data/raw/Telco-Customer-Churn.csv") -> None:
    """Download the IBM Telco dataset if it doesn't exist."""
    if not os.path.exists(raw_data_path):
        logger.info(f"Downloading dataset from {URL} to {raw_data_path}...")
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(URL, raw_data_path)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    else:
        logger.info(f"Dataset already exists at {raw_data_path}.")

def load_raw_data(path: str = "data/raw/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load the raw dataset into a pandas DataFrame."""
    if not os.path.exists(path):
        download_data(path)
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    
    # Basic validation
    expected_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                     'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                     'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
                     
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
        
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
