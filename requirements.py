"""
Loan Approval Predictor - Streamlit Web Application
ML Assignment 2
Author: Sumit Mondal (2024dc04216)
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
ALL_MODELS_PATH = "all_models.pkl"
METRICS_PATH = "model_comparison_metrics.csv"

# Title and Description
st.markdown('<p class="main-header">🏦 Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 - Binary Classification</p>', unsafe_allow_html=True)

st.success("**Submitted By:** Sumit Mondal (2024dc04216)")

st.markdown("""
This application predicts whether a loan application will be **Approved** or **Rejected** 
based on the applicant's personal and financial information. You can:
- 📊 Compare performance of 6 different ML models
- 🎯 Make individual predictions with manual input
- 📁 Upload a CSV file for batch predictions
""")

# Load Resources
@st.cache_resource
def load_all_resources():
    """Load all models, scaler, and feature names from single pickle file"""
    if os.path.exists(ALL_MODELS_PATH):
        all_data = joblib.load(ALL_MODELS_PATH)
        return all_data['models'], all_data['scaler'], all_data['feature_names']
    else:
        st.error(f"❌ Model file '{ALL_MODELS_PATH}' not found! Please run the notebook first.")
        return None, None, None

def get_model(model_name, all_models):
    """Get a specific model by name"""
    if all_models is None:
        return None
    if model_name in all_models:
        return all_models[model_name]
    else:
        st.error(f"❌ Model '{model_name}' not found!")
        return None

# Load resources
all_models, scaler, feature_names = load_all_resources()

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Model Selection
available_models = [
    "Random Forest",
    "XGBoost", 
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes"
]

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    available_models,
    help="Choose a model to use for predictions"
)

# Load selected model
model = get_model(selected_model_name, all_models)

if model is None or scaler is None or feature_names is None:
    st.error("❌ Required files are missing. Please run the notebook first.")
    st.stop()

# Display Model Performance
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance")

try:
    metrics_df = pd.read_csv(METRICS_PATH)
    model_metrics = metrics_df[metrics_df["ML Model Name"] == selected_model_name]
    
    if not model_metrics.empty:
        metrics = model_metrics.iloc[0]
        st.sidebar.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.sidebar.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        st.sidebar.metric("AUC", f"{metrics['AUC']:.4f}")
        st.sidebar.metric("MCC", f"{metrics['MCC']:.4f}")
    else:
        st.sidebar.warning(f"No metrics found for {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Could not load metrics: {e}")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Model Comparison", "ℹ️ About Dataset"])

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================
with tab1:
    input_mode = st.radio(
        "Choose Input Mode:",
        ["Manual Entry", "Upload CSV"],
        horizontal=True,
        help="Enter data manually or upload a CSV file for batch predictions"
    )
    
    if input_mode == "Manual Entry":
        st.subheader("📝 Enter Applicant Information")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Personal Information**")
                person_age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=30,
                    step=1
                )
                person_gender = st.selectbox(
                    "Gender",
                    ["male", "female"]
                )
                person_education = st.selectbox(
                    "Education Level",
                    ["High School", "Bachelor", "Master", "Associate", "Doctorate"]
                )
                person_home_ownership = st.selectbox(
                    "Home Ownership",
                    ["RENT", "OWN", "MORTGAGE", "OTHER"]
                )
            
            with col2:
                st.markdown("**💼 Employment & Income**")
                person_income = st.number_input(
                    "Annual Income ($)",
                    min_value=0,
                    max_value=1000000,
                    value=50000,
                    step=1000
                )
                person_emp_exp = st.number_input(
                    "Employment Experience (years)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1
                )
            
            with col3:
                st.markdown("**💰 Loan Information**")
                loan_amnt = st.number_input(
                    "Loan Amount ($)",
                    min_value=500,
                    max_value=100000,
                    value=10000,
                    step=500
                )
                loan_intent = st.selectbox(
                    "Loan Intent",
                    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
                )
                loan_int_rate = st.number_input(
                    "Interest Rate (%)",
                    min_value=5.0,
                    max_value=30.0,
                    value=10.0,
                    step=0.1
                )
                
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("**📊 Credit History**")
                cb_person_cred_hist_length = st.number_input(
                    "Credit History Length (years)",
                    min_value=0,
                    max_value=30,
                    value=5,
                    step=1
                )
                credit_score = st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=650,
                    step=10
                )
            
            with col5:
                st.markdown("**🔍 Additional Info**")
                previous_loan_defaults_on_file = st.selectbox(
                    "Previous Loan Defaults",
                    ["No", "Yes"]
                )
                
                # Calculate loan percent income automatically
                if person_income > 0:
                    loan_percent_income = loan_amnt / person_income
                    st.metric("Loan/Income Ratio", f"{loan_percent_income:.2%}")
                else:
                    loan_percent_income = 0
            
            submitted = st.form_submit_button("🔮 Predict Loan Status", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'person_age': [float(person_age)],
                'person_gender': [person_gender],
                'person_education': [person_education],
                'person_income': [float(person_income)],
                'person_emp_exp': [person_emp_exp],
                'person_home_ownership': [person_home_ownership],
                'loan_amnt': [float(loan_amnt)],
                'loan_intent': [loan_intent],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_percent_income],
                'cb_person_cred_hist_length': [float(cb_person_cred_hist_length)],
                'credit_score': [credit_score],
                'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
            })
            
            try:
                # Encode categorical variables (same as training)
                input_encoded = pd.get_dummies(input_data, drop_first=True)
                
                # Align columns with training data
                for col in feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                input_encoded = input_encoded[feature_names]
                
                # Scale if needed for certain models
                models_requiring_scaling = ["Logistic Regression", "KNN"]
                if selected_model_name in models_requiring_scaling:
                    input_scaled = scaler.transform(input_encoded)
                    prediction = model.predict(input_scaled)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_scaled)[0]
                else:
                    prediction = model.predict(input_encoded)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_encoded)[0]
                
                # Display prediction
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if prediction == 1:
                        st.success("### ✅ APPROVED")
                        st.balloons()
                    else:
                        st.error("### ❌ REJECTED")
                
                with col_res2:
                    if hasattr(model, "predict_proba"):
                        approval_prob = proba[1] * 100
                        rejection_prob = proba[0] * 100
                        
                        st.metric("Approval Probability", f"{approval_prob:.2f}%")
                        st.metric("Rejection Probability", f"{rejection_prob:.2f}%")
                        
                        # Probability chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Rejected', 'Approved'],
                                y=[rejection_prob, approval_prob],
                                marker_color=['#ff4b4b', '#00cc66'],
                                text=[f'{rejection_prob:.1f}%', f'{approval_prob:.1f}%'],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Confidence",
                            yaxis_title="Probability (%)",
                            showlegend=False,
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.exception(e)
    
    else:  # Upload CSV mode
        st.subheader("📁 Batch Prediction from CSV")
        
        st.info("""
        **CSV Format Requirements:**
        Your CSV must contain these columns: `person_age`, `person_gender`, `person_education`, 
        `person_income`, `person_emp_exp`, `person_home_ownership`, `loan_amnt`, `loan_intent`, 
        `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`, 
        `previous_loan_defaults_on_file`
        """)
        
        # Download template
        template_data = {
            'person_age': [30.0],
            'person_gender': ['male'],
            'person_education': ['Bachelor'],
            'person_income': [50000.0],
            'person_emp_exp': [5],
            'person_home_ownership': ['RENT'],
            'loan_amnt': [10000.0],
            'loan_intent': ['PERSONAL'],
            'loan_int_rate': [10.0],
            'loan_percent_income': [0.2],
            'cb_person_cred_hist_length': [5.0],
            'credit_score': [650],
            'previous_loan_defaults_on_file': ['No']
        }
        template_df = pd.DataFrame(template_data)
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Template CSV",
            data=template_csv,
            file_name="loan_template.csv",
            mime="text/csv",
            help="Download this template and fill it with your data"
        )
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"📊 Uploaded {len(batch_df)} records")
                st.dataframe(batch_df.head(10), use_container_width=True)
                
                if st.button("🔮 Run Batch Prediction", type="primary"):
                    # Encode
                    batch_encoded = pd.get_dummies(batch_df, drop_first=True)
                    
                    # Align columns
                    for col in feature_names:
                        if col not in batch_encoded.columns:
                            batch_encoded[col] = 0
                    
                    batch_encoded = batch_encoded[feature_names]
                    
                    # Predict
                    models_requiring_scaling = ["Logistic Regression", "KNN"]
                    if selected_model_name in models_requiring_scaling:
                        batch_scaled = scaler.transform(batch_encoded)
                        predictions = model.predict(batch_scaled)
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(batch_scaled)[:, 1]
                    else:
                        predictions = model.predict(batch_encoded)
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(batch_encoded)[:, 1]
                    
                    # Add results to dataframe
                    batch_df['Predicted_Status'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
                    if hasattr(model, "predict_proba"):
                        batch_df['Approval_Probability'] = probabilities
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Show statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Total Applications", len(batch_df))
                        st.metric("Approved", sum(predictions == 1))
                    with col_stat2:
                        st.metric("Approval Rate", f"{sum(predictions == 1)/len(predictions)*100:.1f}%")
                        st.metric("Rejected", sum(predictions == 0))
                    
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "loan_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.exception(e)

# ============================================================================
# TAB 2: MODEL COMPARISON
# ============================================================================
with tab2:
    st.subheader("📊 All Models Performance Comparison")
    
    try:
        metrics_df = pd.read_csv(METRICS_PATH)
        
        # Display metrics table
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'MCC': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['F1 Score']),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # F1 Score comparison
            fig1 = px.bar(
                metrics_df.sort_values('F1 Score', ascending=True),
                y='ML Model Name',
                x='F1 Score',
                title='F1 Score Comparison',
                color='F1 Score',
                color_continuous_scale='Blues',
                orientation='h'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Accuracy comparison
            fig2 = px.bar(
                metrics_df.sort_values('Accuracy', ascending=True),
                y='ML Model Name',
                x='Accuracy',
                title='Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='Greens',
                orientation='h'
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Radar chart for all metrics
        fig3 = go.Figure()
        
        for idx, row in metrics_df.iterrows():
            fig3.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score'], row['AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                fill='toself',
                name=row['ML Model Name']
            ))
        
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="All Metrics Comparison (Radar Chart)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load metrics: {e}")

# ============================================================================
# TAB 3: ABOUT DATASET
# ============================================================================
with tab3:
    st.subheader("ℹ️ About the Dataset")
    
    st.markdown("""
    ### 📚 Dataset Information
    
    **Source:** [Kaggle - Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
    
    **Objective:** Predict whether a loan application will be approved or rejected based on applicant information.
    
    ### 📊 Features Description
    
    | Feature | Type | Description |
    |---------|------|-------------|
    | **person_age** | Numeric | Age of the applicant |
    | **person_gender** | Categorical | Gender (male/female) |
    | **person_education** | Categorical | Highest education level |
    | **person_income** | Numeric | Annual income in dollars |
    | **person_emp_exp** | Numeric | Years of employment experience |
    | **person_home_ownership** | Categorical | Home ownership status (RENT/OWN/MORTGAGE/OTHER) |
    | **loan_amnt** | Numeric | Requested loan amount |
    | **loan_intent** | Categorical | Purpose of the loan |
    | **loan_int_rate** | Numeric | Interest rate (%) |
    | **loan_percent_income** | Numeric | Loan amount as percentage of income |
    | **cb_person_cred_hist_length** | Numeric | Length of credit history in years |
    | **credit_score** | Numeric | Credit score (300-850) |
    | **previous_loan_defaults_on_file** | Categorical | Whether applicant has previous defaults |
    | **loan_status** | Binary | Target variable (0=Rejected, 1=Approved) |
    
    ### 🎯 Models Used
    
    1. **Logistic Regression** - Linear classification with good interpretability
    2. **Decision Tree** - Non-linear, easy to interpret
    3. **K-Nearest Neighbors (KNN)** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Sequential ensemble method
    
    ### 📈 Evaluation Metrics
    
    - **Accuracy**: Overall correctness
    - **Precision**: Accuracy of positive predictions
    - **Recall**: Coverage of actual positives
    - **F1 Score**: Harmonic mean of precision and recall
    - **AUC**: Area Under ROC Curve
    - **MCC**: Matthews Correlation Coefficient
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>ML Assignment 2 | Loan Approval Classification | 2024</p>",
    unsafe_allow_html=True
)
