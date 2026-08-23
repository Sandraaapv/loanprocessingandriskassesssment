"""
Loan Approval and Risk Assessment System
Streamlit Dashboard
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Loan Risk Assessment",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0A0A0B !important;
        color: #E5E7EB !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #0A0A0B !important;
    }

    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 1.5rem !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: #9CA3AF !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.75rem 0.25rem !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #E5E7EB !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #3B82F6 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #3B82F6 !important;
    }
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    label, label p, label span, div[data-testid="stWidgetLabel"] p {
        color: #9CA3AF !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 0.25rem !important;
    }
    div[data-baseweb="select"] > div,
    input[type="number"],
    input[type="text"],
    div[data-baseweb="input"] > div {
        background-color: #1A1A1D !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    input:focus,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #9CA3AF !important;
    }
    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: #141416 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    li[data-baseweb="option"] {
        background-color: #141416 !important;
        color: #E5E7EB !important;
    }
    li[data-baseweb="option"]:hover, li[aria-selected="true"] {
        background-color: #1A1A1D !important;
        color: #FFFFFF !important;
    }

    div.stButton > button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    .card {
        background-color: #141416;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-header {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 1rem;
    }

    .result-panel {
        background-color: #141416;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.25rem 0;
    }
    .badge-approved {
        background-color: #10B981;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.35rem 0.9rem;
        border-radius: 16px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-conditional {
        background-color: #F59E0B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.35rem 0.9rem;
        border-radius: 16px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #EF4444;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.35rem 0.9rem;
        border-radius: 16px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stat-label {
        color: #9CA3AF;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #3B82F6 !important;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #1A1A1D !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .stMarkdown p {
        color: #D1D5DB;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="color: #FFFFFF; font-weight: 600; font-size: 1.8rem; margin: 0;">Loan Approval & Risk Assessment</h1>
        <p style="color: #9CA3AF; font-size: 0.9rem; margin-top: 0.25rem;">Automated credit risk evaluation model</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model_path = "loan_rf_model.joblib"
    cols_path = "feature_columns.joblib"
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return None, None
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    return model, feature_cols

model, feature_cols = load_artifacts()

if model is None:
    st.error("Model artifacts missing. Run `python loan_model.py` to train and export the model.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Applicant Assessment", "Model Performance", "Documentation"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="card">
                <div class="card-header">Personal Details</div>
        """, unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Married", "Single"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="card">
                <div class="card-header">Financial Profile</div>
        """, unsafe_allow_html=True)
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=4500, step=250)
        coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, value=1500, step=250)
        loan_amount = st.number_input("Loan Amount ($ in thousands)", min_value=10, value=150, step=10)
        loan_term = st.selectbox("Loan Term (Months)", [360, 240, 180, 120, 60], index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="card">
                <div class="card-header">Credit & Property</div>
        """, unsafe_allow_html=True)
        credit_history = st.selectbox("Credit History Met", ["Yes (Good)", "No (Poor/None)"])
        property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])
        st.markdown("</div>", unsafe_allow_html=True)

    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Married" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    credit_val = 1.0 if credit_history.startswith("Yes") else 0.0

    total_income = applicant_income + coapplicant_income
    lti_ratio = loan_amount / (total_income / 1000 + 1e-5)

    prop_semiurban = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0

    input_data = {
        "Gender": gender_val,
        "Married": married_val,
        "Dependents": dependents_val,
        "Education": education_val,
        "Self_Employed": self_employed_val,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_val,
        "Total_Income": total_income,
        "Loan_to_Income_Ratio": lti_ratio,
        "Property_Area_Semiurban": prop_semiurban,
        "Property_Area_Urban": prop_urban
    }

    input_df = pd.DataFrame([input_data])[feature_cols]

    if st.button("Evaluate Application", type="primary", use_container_width=True):
        with st.spinner("Processing application..."):
            prob_approval = model.predict_proba(input_df)[0][1]
            approval_pct = round(prob_approval * 100, 1)
            risk_score = round((1 - prob_approval) * 100, 1)

            if approval_pct >= 75:
                status_html = "<span class='badge-approved'>APPROVED</span>"
                tier = "Low Risk"
            elif approval_pct >= 50:
                status_html = "<span class='badge-conditional'>CONDITIONAL</span>"
                tier = "Moderate Risk"
            else:
                status_html = "<span class='badge-rejected'>REJECTED</span>"
                tier = "High Risk"

            st.markdown(f"""
                <div class="result-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <div>
                            <div class="stat-label">Decision</div>
                            <div style="margin-top: 0.3rem;">{status_html}</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="stat-label">Risk Category</div>
                            <div style="color: #FFFFFF; font-weight: 600; font-size: 1rem; margin-top: 0.2rem;">{tier}</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 2.5rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem;">
                        <div>
                            <div class="stat-label">Approval Probability</div>
                            <div class="stat-value">{approval_pct}%</div>
                        </div>
                        <div>
                            <div class="stat-label">Risk Score</div>
                            <div class="stat-value">{risk_score}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='stat-label' style='margin-bottom: 0.3rem;'>Model Score</div>", unsafe_allow_html=True)
            st.progress(approval_pct / 100.0)

            st.markdown("<h3 style='font-size: 1.1rem; margin-top: 1.5rem;'>Financial Metrics</h3>", unsafe_allow_html=True)

            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.markdown(f"""
                    <div class="card">
                        <div class="stat-label">Total Monthly Income</div>
                        <div class="stat-value">${total_income:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with f_col2:
                st.markdown(f"""
                    <div class="card">
                        <div class="stat-label">Loan-to-Income Ratio</div>
                        <div class="stat-value">{lti_ratio:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with f_col3:
                est_monthly_payment = (loan_amount * 1000) / (loan_term if loan_term > 0 else 360)
                st.markdown(f"""
                    <div class="card">
                        <div class="stat-label">Est. Monthly Repayment</div>
                        <div class="stat-value">${est_monthly_payment:,.0f}/mo</div>
                    </div>
                """, unsafe_allow_html=True)

            strengths = []
            concerns = []

            if credit_val == 1.0:
                strengths.append("Verified credit history")
            else:
                concerns.append("Unverified or poor credit history")

            if lti_ratio < 0.035:
                strengths.append(f"Low loan-to-income ratio ({lti_ratio:.2f})")
            else:
                concerns.append(f"Elevated loan-to-income ratio ({lti_ratio:.2f})")

            if total_income >= 5000:
                strengths.append(f"Household income (${total_income:,.0f})")
            else:
                concerns.append(f"Lower household income (${total_income:,.0f})")

            if married_val == 1 and coapplicant_income > 0:
                strengths.append(f"Coapplicant income (${coapplicant_income:,.0f})")

            s_col, c_col = st.columns(2)
            with s_col:
                st.markdown("""
                    <div class="card">
                        <div class="card-header" style="color: #10B981;">Positive Factors</div>
                """, unsafe_allow_html=True)
                if strengths:
                    for s in strengths:
                        st.markdown(f"<div style='color: #D1D5DB; font-size: 0.88rem; margin-bottom: 0.3rem;'>• {s}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #9CA3AF; font-size: 0.88rem;'>None flagged.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_col:
                st.markdown("""
                    <div class="card">
                        <div class="card-header" style="color: #EF4444;">Risk Factors</div>
                """, unsafe_allow_html=True)
                if concerns:
                    for c in concerns:
                        st.markdown(f"<div style='color: #D1D5DB; font-size: 0.88rem; margin-bottom: 0.3rem;'>• {c}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #9CA3AF; font-size: 0.88rem;'>None flagged.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<h4 style='font-size: 0.95rem; color: #E5E7EB;'>Model Comparison</h4>", unsafe_allow_html=True)
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", use_container_width=True)
        else:
            st.info("Run `loan_model.py` to generate `model_comparison.png`")

        st.markdown("<h4 style='font-size: 0.95rem; color: #E5E7EB; margin-top: 1rem;'>Feature Importance</h4>", unsafe_allow_html=True)
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)
        else:
            st.info("Run `loan_model.py` to generate `feature_importance.png`")

    with col_chart2:
        st.markdown("<h4 style='font-size: 0.95rem; color: #E5E7EB;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `loan_model.py` to generate `confusion_matrix.png`")

        st.markdown("<h4 style='font-size: 0.95rem; color: #E5E7EB; margin-top: 1rem;'>ROC Curves</h4>", unsafe_allow_html=True)
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", use_container_width=True)
        else:
            st.info("Run `loan_model.py` to generate `roc_curve.png`")

with tab3:
    st.markdown("<h3>Documentation</h3>", unsafe_allow_html=True)

    if os.path.exists("results_summary.txt"):
        with open("results_summary.txt", "r") as f:
            summary_content = f.read()
        st.code(summary_content, language="text")

    if os.path.exists("PRESENTATION_GUIDE.md"):
        with open("PRESENTATION_GUIDE.md", "r") as f:
            guide_content = f.read()
        with st.expander("Presentation Notes", expanded=False):
            st.markdown(guide_content)
