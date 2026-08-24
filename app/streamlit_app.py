import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Add the project root to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import ChurnPredictor
from src.explainability.shap_explainer import ShapExplainer

# --- CONFIGURATION CONSTANTS ---
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

EXPECTED_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

st.set_page_config(
    page_title="Churn Intelligence",
    layout="wide"
)

# --- THEME AND STYLING ---
st.markdown("""
<style>
    /* Clean Enterprise Background */
    .stApp {
        background: #F7F8FA;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, p, label, li {
        color: #111827;
    }
    
    /* Restrained White Cards */
    [data-testid="stMetric"], [data-testid="stForm"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        border-radius: 8px !important;
        padding: 24px !important;
    }
    
    /* Input Widgets */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #F9FAFB !important;
        border: 1px solid #D1D5DB !important;
        color: #111827 !important;
        border-radius: 6px !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus, .stNumberInput>div>div>input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }
    
    /* Primary Submit Button */
    [data-testid="baseButton-secondaryFormSubmit"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        width: 100% !important;
        transition: background 0.2s ease !important;
        margin-top: 2rem !important;
    }
    [data-testid="baseButton-secondaryFormSubmit"]:hover {
        background-color: #1D4ED8 !important;
    }
    
    /* Secondary Download Button */
    [data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #374151 !important;
        border: 1px solid #D1D5DB !important;
        font-weight: 500 !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
    }
    
    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        padding: 10px 0;
        color: #4B5563; 
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: transparent;
        border-bottom: 2px solid #2563EB;
        color: #111827 !important;
        font-weight: 600;
    }
    
    /* Subheaders inside forms - Adjusted top margin for breathing room */
    .form-subheader {
        color: #111827;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    /* First subheader doesn't need top margin */
    .form-subheader:first-of-type {
        margin-top: 0;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: #FFFFFF; 
        border: 1px solid #E5E7EB; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); 
        border-radius: 8px; 
        padding: 30px; 
        text-align: center; 
        border-left: 4px solid;
    }
    
    /* Transparent top header */
    [data-testid="stHeader"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_models():
    return {
        "XGBoost": ChurnPredictor("xgboost"),
        "Logistic Regression": ChurnPredictor("logistic_regression")
    }
    
@st.cache_resource
def load_metrics():
    return {
        "XGBoost": joblib.load("models/xgb_metrics.joblib"),
        "Logistic Regression": joblib.load("models/lr_metrics.joblib")
    }

@st.cache_data
def load_test_data():
    if not os.path.exists("data/processed/test_data.csv"):
        return pd.DataFrame()
    return pd.read_csv("data/processed/test_data.csv")

@st.cache_resource
def get_shap_explainer(model_name):
    internal_name = "xgboost" if model_name == "XGBoost" else "logistic_regression"
    return ShapExplainer(internal_name)

models_dict = load_models()
metrics_dict = load_metrics()
df_default_test = load_test_data()

# --- MAIN HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("<h1 style='margin-bottom: 0; padding-bottom: 0;'>Churn Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #4B5563; margin-top: 5px;'>Customer Retention Analytics</p>", unsafe_allow_html=True)
with col_head2:
    st.markdown("<div style='text-align: right; color: #6B7280; padding-top: 5px; font-weight: 500; font-size: 0.9rem;'>Prediction Model</div>", unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Prediction Model", 
        ["XGBoost", "Logistic Regression"],
        label_visibility="collapsed"
    )
    st.markdown("<div style='text-align: right; color: #6B7280; font-size: 0.8rem; margin-top: -10px;'>Switch models to compare predictions and explanations.</div>", unsafe_allow_html=True)

st.write("")

selected_predictor = models_dict.get(selected_model_name)
selected_metrics = metrics_dict.get(selected_model_name)

# --- TABS FOR NAVIGATION ---
tab_overview, tab_predict, tab_batch, tab_insights = st.tabs([
    "Overview", 
    "Customer Prediction", 
    "Batch Scoring", 
    "Model Insights"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.write("")
    if selected_metrics is None or df_default_test.empty:
        st.warning("Models or data not found. Please run the training pipeline first.")
    else:
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            df_raw = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
            total_customers = f"{len(df_raw):,}"
            churn_rate = f"{(df_raw['Churn'] == 'Yes').mean() * 100:.1f}%"
            avg_tenure = f"{df_raw['tenure'].mean():.1f} mo"
        except FileNotFoundError:
            total_customers = "N/A"
            churn_rate = "N/A"
            avg_tenure = "N/A"
            
        col1.metric("Customers", total_customers)
        col2.metric("Churn Rate", churn_rate)
        col3.metric("Avg. Tenure", avg_tenure)
        col4.metric("ROC-AUC", f"{selected_metrics['ROC-AUC']:.3f}")
        
        st.write("")
        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown(f"### Customer Risk Distribution ({selected_model_name})")
            preds, probs = selected_predictor.predict(df_default_test)
            risk_categories = pd.cut(probs, bins=[0, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0], labels=['Low', 'Medium', 'High'])
            risk_counts = risk_categories.value_counts().sort_index()
            st.bar_chart(risk_counts, color="#2563EB")
            
        with col_chart2:
            st.markdown(f"### Top Churn Drivers ({selected_model_name})")
            with st.spinner("Loading SHAP explainer..."):
                explainer = get_shap_explainer(selected_model_name)
                sample_X = selected_predictor.preprocessor.transform(df_default_test.drop(columns=['Churn'], errors='ignore').head(200))
                fig = explainer.plot_summary(sample_X, show=False)
                st.pyplot(fig)

# --- TAB 2: CUSTOMER PREDICTION ---
with tab_predict:
    st.write("")
    st.markdown("### Customer Prediction")
    st.markdown("Enter customer attributes to estimate churn probability.")
    
    with st.form("prediction_form"):
        st.markdown("<div class='form-subheader'>Demographics</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: gender = st.selectbox("Gender", ["Male", "Female"])
        with col2: senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        with col3: partner = st.selectbox("Partner", ["Yes", "No"])
        with col4: dependents = st.selectbox("Dependents", ["Yes", "No"])
            
        st.markdown("<div class='form-subheader'>Services</div>", unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5: phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        with col6: multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with col7: internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        st.markdown("<div class='form-subheader'>Internet Add-ons</div>", unsafe_allow_html=True)
        col8, col9, col10 = st.columns(3)
        with col8:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with col9:
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with col10:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
        st.markdown("<div class='form-subheader'>Contract & Billing</div>", unsafe_allow_html=True)
        col11, col12, col13 = st.columns(3)
        with col11:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with col12:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        with col13:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)
            
        submitted = st.form_submit_button("Predict Churn")
        
    if submitted:
        if selected_predictor is None:
            st.error("Model not trained yet.")
        else:
            input_df = pd.DataFrame({
                'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner], 'Dependents': [dependents],
                'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines], 
                'InternetService': [internet_service], 'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup], 
                'DeviceProtection': [device_protection], 'TechSupport': [tech_support], 'StreamingTV': [streaming_tv], 
                'StreamingMovies': [streaming_movies], 'Contract': [contract], 'PaperlessBilling': [paperless_billing], 
                'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
            })
            
            preds, probs = selected_predictor.predict(input_df)
            prob = probs[0]
            
            st.write("")
            st.divider()
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                border_color = "#10B981" if prob < LOW_RISK_THRESHOLD else "#F59E0B" if prob < HIGH_RISK_THRESHOLD else "#EF4444"
                risk_label = "Low Risk" if prob < LOW_RISK_THRESHOLD else "Medium Risk" if prob < HIGH_RISK_THRESHOLD else "High Risk"
                
                st.markdown(f"""
                <div class="prediction-card" style="border-left-color: {border_color};">
                    <div style="color: #6B7280; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Prediction Model<br><b style="color: #111827;">{selected_model_name}</b></div>
                    <div style="font-size: 3.5rem; font-weight: 700; color: #111827; margin: 15px 0 5px 0;">{prob * 100:.1f}%</div>
                    <div style="color: {border_color}; font-weight: 700; font-size: 1.2rem;">{risk_label}</div>
                    <div style="color: #9CA3AF; font-size: 0.8rem; margin-top: 15px; line-height: 1.2;">* Risk bands are presentation thresholds, not calibrated business decisions.</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("### Why this prediction?")
                st.markdown("""
                The following chart shows how each customer attribute contributes to the overall risk. 
                - **Red features** increase churn probability.
                - **Blue features** reduce churn probability.
                """)
                with st.spinner(f"Analyzing risk factors with {selected_model_name}..."):
                    explainer = get_shap_explainer(selected_model_name)
                    X_processed = selected_predictor.preprocessor.transform(input_df)
                    shap_obj = explainer.explain(X_processed)
                    
                    try:
                        vals = shap_obj.values
                        feature_names = selected_predictor.preprocessor.get_feature_names_out()
                        
                        def clean_feat(name):
                            return name.split('__')[-1].replace('_', ' ')
                            
                        contribs = list(zip([clean_feat(f) for f in feature_names], vals))
                        contribs.sort(key=lambda x: x[1], reverse=True)
                        
                        top_inc = [c for c in contribs if c[1] > 0][:3]
                        top_dec = [c for c in reversed(contribs) if c[1] < 0][:3]
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("<div style='color: #111827; font-weight: 600; margin-bottom: 8px;'>Factors contributing to higher predicted churn risk</div>", unsafe_allow_html=True)
                            for feat, val in top_inc:
                                st.markdown(f"<div style='color: #4B5563; font-size: 0.95rem; margin-bottom: 4px;'>• {feat}: <span style='color: #EF4444;'>+{val:.2f}</span></div>", unsafe_allow_html=True)
                                
                        with c2:
                            st.markdown("<div style='color: #111827; font-weight: 600; margin-bottom: 8px;'>Factors contributing to lower predicted churn risk</div>", unsafe_allow_html=True)
                            for feat, val in top_dec:
                                st.markdown(f"<div style='color: #4B5563; font-size: 0.95rem; margin-bottom: 4px;'>• {feat}: <span style='color: #2563EB;'>{val:.2f}</span></div>", unsafe_allow_html=True)
                                
                        st.caption(f"*Values represent exact SHAP contributions to the {selected_model_name} model's baseline output.*")
                    except Exception as e:
                        logging.error(f"SHAP text generation failed: {e}")
                    
                    st.write("")
                    fig = explainer.plot_waterfall(X_processed, show=False)
                    st.pyplot(fig)

# --- TAB 3: BATCH SCORING ---
with tab_batch:
    st.write("")
    st.markdown("### Batch Scoring")
    st.markdown("Upload a CSV to generate churn probabilities and risk levels for multiple customers.")
    
    template_df = pd.DataFrame(columns=EXPECTED_COLS)
    st.download_button(
        "Download sample CSV template", 
        data=template_df.to_csv(index=False).encode('utf-8'), 
        file_name="churn_template.csv", 
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in EXPECTED_COLS if col not in user_df.columns]
            
            if missing_cols:
                st.error("Uploaded dataset is missing required columns.")
            else:
                with st.spinner(f"Generating predictions using {selected_model_name}..."):
                    preds, probs = selected_predictor.predict(user_df)
                    user_df.insert(0, 'Predicted_Risk', pd.cut(probs, bins=[0, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0], labels=['Low', 'Medium', 'High']))
                    user_df.insert(0, 'Churn_Prediction', np.where(preds == 1, 'Yes', 'No'))
                    user_df.insert(0, 'Churn_Probability', np.round(probs * 100, 2))
                    
                    st.success(f"Successfully processed {len(user_df):,} records.")
                    st.markdown(f"**Prediction Model:** {selected_model_name}")
                    st.dataframe(user_df.head(100))
                    
                    csv = user_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f'churn_batch_predictions_{selected_model_name.replace(" ", "_").lower()}.csv',
                        mime='text/csv',
                    )
        except pd.errors.EmptyDataError:
            st.error("Unable to read the uploaded file. Please upload a valid CSV.")
        except Exception as e:
            logging.error(f"Batch prediction failure: {e}")
            st.error("Unable to generate predictions for this dataset.")

# --- TAB 4: MODEL INSIGHTS ---
with tab_insights:
    st.write("")
    st.markdown("### Model Performance")
    
    xgb_metrics = metrics_dict.get("XGBoost")
    lr_metrics = metrics_dict.get("Logistic Regression")
    
    if xgb_metrics is None or lr_metrics is None:
        st.warning("Metrics not found. Train models first.")
    else:
        st.markdown("Compare model performance across key classification metrics.")
        
        metrics_df = pd.DataFrame({
            "XGBoost": [xgb_metrics['ROC-AUC'], xgb_metrics['Accuracy'], xgb_metrics['Precision'], xgb_metrics['Recall'], xgb_metrics['F1-score']],
            "Logistic Regression": [lr_metrics['ROC-AUC'], lr_metrics['Accuracy'], lr_metrics['Precision'], lr_metrics['Recall'], lr_metrics['F1-score']]
        }, index=["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-score"])
        
        st.dataframe(metrics_df.style.highlight_max(axis=1, color='#E5E7EB'))
        
        st.markdown(f"**Currently Selected Model: {selected_model_name}**")
        if selected_model_name == "XGBoost":
            st.markdown("*Selected due to its superior ROC-AUC on the evaluation dataset and robustness to non-linear feature interactions.*")
        else:
            st.markdown("*Serves as a robust, interpretable linear baseline.*")
        
        st.divider()
        st.markdown(f"### Confusion Matrix ({selected_model_name})")
        
        cm = selected_metrics["Confusion Matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=['Retained', 'Churned'])
        disp.plot(cmap='Blues', ax=ax, colorbar=True)
        
        st.pyplot(fig)
