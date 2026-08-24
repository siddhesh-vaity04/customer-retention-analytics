# Churn Intelligence

## Project Overview
Churn Intelligence is an end-to-end customer churn prediction dashboard. It leverages machine learning to predict which customers are at risk of leaving, providing explainable insights to help retention efforts.

## Problem Statement
Customer churn is a critical metric for subscription-based businesses. Identifying high-risk customers early allows companies to take proactive measures to retain them.

## Objectives
- Build a robust data preprocessing pipeline.
- Train and evaluate Logistic Regression and XGBoost models.
- Provide global and local explainability using SHAP.
- Serve predictions through a professional Streamlit dashboard.

## Key Features
- **Data Pipeline**: Automated data fetching, cleaning, and preprocessing.
- **Model Training**: Reproducible training scripts for Logistic Regression and XGBoost.
- **Explainability**: SHAP (SHapley Additive exPlanations) for transparent model insights.
- **Dashboard**: A professional Streamlit application for predicting individual customer risk and exploring global model behavior.

## Architecture
- `app/`: Streamlit dashboard application.
- `data/`: Raw and processed dataset storage.
- `models/`: Saved joblib model artifacts.
- `notebooks/`: Jupyter notebooks for EDA.
- `src/`: Reusable Python modules for data loading, preprocessing, training, and explainability.
- `tests/`: Pytest suite for automated testing.

## Tech Stack
- **Language**: Python 3.11+
- **Machine Learning**: pandas, numpy, scikit-learn, XGBoost
- **Explainability**: SHAP
- **UI**: Streamlit
- **Testing**: pytest

## Dataset
We use the IBM Telco Customer Churn dataset. The target variable is `Churn` (Yes=1, No=0).

## Project Structure
```text
churn-prediction-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── screenshots/
├── models/
├── notebooks/
│   └── 01_eda_and_model_experiments.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── explainability/
│       └── shap_explainer.py
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

## Exploratory Data Analysis
EDA is available in `notebooks/01_eda_and_model_experiments.ipynb`.
Key findings:
- Month-to-month contracts have the highest churn rate.
- Customers with low tenure are more likely to churn.
- Higher monthly charges correlate with higher churn.

## Machine Learning Approach

### Logistic Regression
Used as a strong, interpretable baseline model.

### XGBoost
A powerful gradient boosting classifier that captures non-linear relationships. Selected as the primary model due to its high performance and robustness.

## Model Evaluation
*Metrics will be available after training.* Both models are evaluated on Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Explainability with SHAP
- **Global Explanations**: Summary plots identify which features drive churn across the entire dataset.
- **Local Explanations**: Waterfall plots explain individual predictions, showing exactly which factors increased or decreased a specific customer's churn risk.

## Dashboard
The dashboard provides a SaaS-like experience with 3 main sections:
1. **Overview**: High-level KPI metrics, risk distribution, and global churn drivers.
2. **Customer Prediction**: An input form to predict churn for a new/existing customer.
3. **Model Insights**: Comparative metrics and confusion matrices.

## Screenshots
*(Add screenshots here after running the app)*
- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/prediction-explanation.png`
- `docs/screenshots/model-performance.png`

## Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix
# source venv/bin/activate

pip install -r requirements.txt
```

## Running the Project
To run the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

## Training the Models
To fetch data, preprocess, and train the models, run:
```bash
python src/models/train.py
```

## Running Tests
Run the automated test suite:
```bash
pytest -v
```

## Results
XGBoost generally outperforms Logistic Regression in ROC-AUC. SHAP explanations show that `Contract_Month-to-month` and `tenure` are consistently the most important features.

## Limitations
- The dataset is relatively small and imbalanced.
- The default threshold of 0.5 might not be optimal for all business use cases (e.g., if false negatives are very costly).

## Future Improvements
- Implement hyperparameter tuning (e.g., GridSearchCV, Optuna).
- Add support for custom prediction thresholds based on business logic.
- Deploy the app to a cloud platform like AWS or Heroku.

## Author
[Your Name]
