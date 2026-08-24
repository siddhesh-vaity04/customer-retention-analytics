import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from src.models.predict import ChurnPredictor
from src.explainability.shap_explainer import ShapExplainer

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME AND STYLING ---
st.markdown("""
<style>
    /* Charcoal typography and professional spacing */
    .stApp {
        background-color: #FAFAFB;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: #1A1A1A;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        color: white;
    }
    
    .metric-card {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_models():
    if not os.path.exists("models/xgboost.joblib"):
        return None, None
    xgb_pred = ChurnPredictor("xgboost")
    lr_pred = ChurnPredictor("logistic_regression")
    return xgb_pred, lr_pred
    
@st.cache_resource
def load_metrics():
    if not os.path.exists("models/xgb_metrics.joblib"):
        return None, None
    xgb_metrics = joblib.load("models/xgb_metrics.joblib")
    lr_metrics = joblib.load("models/lr_metrics.joblib")
    return xgb_metrics, lr_metrics

@st.cache_data
def load_test_data():
    if not os.path.exists("data/processed/test_data.csv"):
        return pd.DataFrame()
    return pd.read_csv("data/processed/test_data.csv")

xgb_pred, lr_pred = load_models()
xgb_metrics, lr_metrics = load_metrics()
df_test = load_test_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("📊 Churn Intelligence")
    st.markdown("**Customer Retention Analytics**")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Customer Prediction", "Model Insights"])
    st.divider()
    active_model = st.selectbox("Active Model", ["XGBoost", "Logistic Regression"])
    
current_predictor = xgb_pred if active_model == "XGBoost" else lr_pred
current_metrics = xgb_metrics if active_model == "XGBoost" else lr_metrics

# --- PAGE: OVERVIEW ---
if page == "Overview":
    st.header("Overview")
    if current_metrics is None or df_test.empty:
        st.warning("Models or data not found. Please run the training pipeline first.")
    else:
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        # We estimate total customers from the dataset size (e.g. if we have train+test, but here we just have test_data as proxy, so let's load raw data for stats)
        try:
            df_raw = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
            total_customers = len(df_raw)
            churn_rate = (df_raw['Churn'] == 'Yes').mean() * 100
            avg_tenure = df_raw['tenure'].mean()
        except:
            total_customers = len(df_test) * 5
            churn_rate = df_test['Churn'].mean() * 100
            avg_tenure = df_test['tenure'].mean()
            
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Average Tenure", f"{avg_tenure:.1f} mo")
        col4.metric("Model ROC-AUC", f"{current_metrics['ROC-AUC']:.3f}")
        
        st.divider()
        st.subheader("Customer Risk Overview")
        st.markdown("Distribution of churn probabilities in the evaluation set.")
        
        # Get predictions for test set
        preds, probs = current_predictor.predict(df_test)
        
        risk_categories = pd.cut(probs, bins=[0, 0.3, 0.7, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
        risk_counts = risk_categories.value_counts().sort_index()
        
        st.bar_chart(risk_counts)
        
        st.divider()
        st.subheader("Top Churn Drivers (Global Importance)")
        with st.spinner("Generating SHAP summary..."):
            explainer = ShapExplainer("xgboost" if active_model == "XGBoost" else "logistic_regression")
            # We sample data for speed
            sample_X = current_predictor.preprocessor.transform(df_test.drop(columns=['Churn']).head(200))
            fig = explainer.plot_summary(sample_X, show=False)
            st.pyplot(fig)

# --- PAGE: CUSTOMER PREDICTION ---
elif page == "Customer Prediction":
    st.header("Customer Prediction")
    st.markdown("Enter customer details to generate a churn risk assessment.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            
        with col3:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
        submitted = st.form_submit_button("Predict Churn")
        
    if submitted:
        if current_predictor is None:
            st.error("Model not trained yet.")
        else:
            # Build DataFrame (fill remaining needed columns with most frequent / median)
            input_dict = {
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract],
                'InternetService': [internet_service],
                'TechSupport': [tech_support],
                'OnlineSecurity': [online_security],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                # Fill missing to avoid errors (based on dataset expectations)
                'gender': ['Male'],
                'PhoneService': ['Yes'],
                'MultipleLines': ['No'],
                'OnlineBackup': ['No'],
                'DeviceProtection': ['No'],
                'StreamingTV': ['No'],
                'StreamingMovies': ['No']
            }
            
            input_df = pd.DataFrame(input_dict)
            
            preds, probs = current_predictor.predict(input_df)
            prob = probs[0]
            
            st.divider()
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("CHURN PROBABILITY", f"{prob * 100:.1f}%")
                
            with res_col2:
                if prob < 0.3:
                    risk = "LOW RISK"
                    color = "green"
                elif prob < 0.7:
                    risk = "MEDIUM RISK"
                    color = "orange"
                else:
                    risk = "HIGH RISK"
                    color = "red"
                    
                st.markdown(f"Risk Category: <strong style='color:{color}'>{risk}</strong>", unsafe_allow_html=True)
                st.caption("*Presentation thresholds (Low < 30%, High > 70%)*")
                
            st.divider()
            st.subheader("Why this customer is at risk")
            
            with st.spinner("Generating SHAP explanation..."):
                explainer = ShapExplainer("xgboost" if active_model == "XGBoost" else "logistic_regression")
                # Preprocess single instance
                X_processed = current_predictor.preprocessor.transform(input_df)
                fig = explainer.plot_waterfall(X_processed, show=False)
                st.pyplot(fig)

# --- PAGE: MODEL INSIGHTS ---
elif page == "Model Insights":
    st.header("Model Insights")
    
    if xgb_metrics is None or lr_metrics is None:
        st.warning("Metrics not found. Train models first.")
    else:
        st.subheader("Logistic Regression vs XGBoost")
        
        metrics_df = pd.DataFrame({
            "Logistic Regression": [lr_metrics['Accuracy'], lr_metrics['Precision'], lr_metrics['Recall'], lr_metrics['F1-score'], lr_metrics['ROC-AUC']],
            "XGBoost": [xgb_metrics['Accuracy'], xgb_metrics['Precision'], xgb_metrics['Recall'], xgb_metrics['F1-score'], xgb_metrics['ROC-AUC']]
        }, index=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
        
        st.dataframe(metrics_df.style.highlight_max(axis=1))
        
        st.divider()
        st.subheader("Confusion Matrix")
        
        cm = current_metrics["Confusion Matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.matshow(cm, cmap='Blues')
        plt.title('Confusion Matrix', pad=20)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + ['No Churn', 'Churn'])
        ax.set_yticklabels([''] + ['No Churn', 'Churn'])
        
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, f'{z}', ha='center', va='center')
            
        st.pyplot(fig)
