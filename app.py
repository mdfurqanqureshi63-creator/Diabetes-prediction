import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="MedPredict AI - Multiple Disease Prognosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #0077b6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #0077b6;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #ffe5e5;
        border-left: 6px solid #d90429;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .risk-low {
        background-color: #e8f8f5;
        border-left: 6px solid #2ec4b6;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 85%;
        font-weight: 700;
        border-radius: 10rem;
        color: #fff;
        background-color: #0077b6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Load predictor and model metrics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_disease_predictor():
    from src.inference import DiseasePredictor
    return DiseasePredictor()

predictor = load_disease_predictor()

def get_metadata():
    meta_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

metadata = get_metadata()

# Helper for Gauge Chart
def create_gauge_chart(prob_percent, title="Risk Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#0d3b66'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#1d3557'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#d90429" if prob_percent >= 50 else "#2ec4b6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ced4da",
            'steps': [
                {'range': [0, 40], 'color': '#d8f3dc'},
                {'range': [40, 65], 'color': '#fff3cd'},
                {'range': [65, 100], 'color': '#ffccd5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def set_nav_page(page_name):
    st.session_state["nav"] = page_name

# Sidebar Navigation
if "nav" not in st.session_state:
    st.session_state["nav"] = "🏠 Dashboard & System Overview"

with st.sidebar:
    st.markdown("### 🏥 **MedPredict AI**")
    st.markdown("*Intelligent Clinical Risk Engine*")
    st.divider()
    
    app_mode = st.radio(
        "Navigation",
        [
            "🏠 Dashboard & System Overview",
            "🩺 Diabetes Risk Prediction",
            "📊 Model Benchmarks & Analytics",
            "ℹ️ Medical Disclaimer & About"
        ],
        key="nav"
    )
    
    st.divider()
    st.markdown("#### ⚡ System Status")
    st.success("● ML Models: Active & Loaded")
    st.caption("Engines: Random Forest, AdaBoost, Logistic Regression, SVC")
    st.divider()
    st.info("💡 **Quick Tip:** Use the Preset Patient Buttons inside each predictor to test samples with 1 click!")

# -------------------------------------------------------------
# PAGE 1: DASHBOARD & SYSTEM OVERVIEW
# -------------------------------------------------------------
if app_mode == "🏠 Dashboard & System Overview":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 2.4rem;">🏥 MedPredict AI Diabetes Risk Prediction System</h1>
        <p style="margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.95;">
            High-accuracy clinical screening powered by validated machine learning models trained on national healthcare benchmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🩺 Diabetes Assessment</h3>
        <p>Predicts diabetes risk based on glucose, insulin, BMI, and other metabolic indicators.</p>
        <p><b>Model:</b> Random Forest Classifier<br>
        <b>Validation Accuracy:</b> 77.92%<br>
        <b>ROC-AUC Score:</b> 81.79%</p>
    </div>
    """, unsafe_allow_html=True)
    st.button("Open Diabetes Screener →", key="btn_goto_diab", on_click=set_nav_page, args=("🩺 Diabetes Risk Prediction",))

    st.markdown("### 🔬 How The Clinical Pipeline Works")
    st.markdown("""
    1. **Data Acquisition & Imputation**: Benchmark clinical datasets are cleaned; non-physiological zeros (e.g., fasting glucose, blood pressure) undergo median imputation.
    2. **Multi-Model Tournament**: Algorithms including **Random Forest, Logistic Regression, Support Vector Machines, Gradient Boosting, AdaBoost, and KNN** are evaluated using stratified 5-fold cross-validation for diabetes prediction.
    3. **Standardization & Champion Deployment**: Champion models and their associated `StandardScaler` transformations are deployed for real-time inference.
    4. **Probabilistic Risk Metering**: Predictions output calibrated probabilities with interactive diagnostic meters and clinical factor breakdowns.
    """)

# -------------------------------------------------------------
# PAGE 2: DIABETES RISK PREDICTION
# -------------------------------------------------------------
elif app_mode == "🩺 Diabetes Risk Prediction":
    st.markdown("## 🩺 Diabetes Risk Diagnostic Engine")
    st.markdown("Screening for diabetes mellitus based on metabolic profile and glucose tolerance measurements.")
    
    # Presets
    st.markdown("##### ⚡ Quick Patient Presets:")
    p_col1, p_col2, p_col3 = st.columns(3)
    if "diab_vals" not in st.session_state:
        st.session_state["diab_vals"] = {
            "preg": 2, "gluc": 115.0, "bp": 72.0, "skin": 25.0,
            "ins": 110.0, "bmi": 27.2, "ped": 0.38, "age": 33
        }
    if p_col1.button("👤 Normal / Healthy Patient", use_container_width=True):
        st.session_state["diab_vals"] = {
            "preg": 1, "gluc": 88.0, "bp": 70.0, "skin": 20.0,
            "ins": 79.0, "bmi": 22.4, "ped": 0.25, "age": 28
        }
        st.rerun()
    if p_col2.button("⚠️ Pre-Diabetic / Borderline Profile", use_container_width=True):
        st.session_state["diab_vals"] = {
            "preg": 3, "gluc": 125.0, "bp": 78.0, "skin": 28.0,
            "ins": 140.0, "bmi": 28.5, "ped": 0.45, "age": 42
        }
        st.rerun()
    if p_col3.button("🚨 High-Risk Diabetic Profile", use_container_width=True):
        st.session_state["diab_vals"] = {
            "preg": 6, "gluc": 168.0, "bp": 88.0, "skin": 35.0,
            "ins": 210.0, "bmi": 36.5, "ped": 0.85, "age": 52
        }
        st.rerun()

    dv = st.session_state["diab_vals"]

    with st.form("diabetes_form"):
        st.markdown("### 📋 Patient Clinical Indicators")
        c1, c2 = st.columns(2)
        
        with c1:
            pregnancies = st.number_input("Pregnancies (Number of times pregnant)", min_value=0, max_value=20, value=int(dv["preg"]), step=1,
                                          help="Number of full-term pregnancies")
            glucose = st.number_input("Fasting Glucose Level (mg/dL)", min_value=40.0, max_value=300.0, value=float(dv["gluc"]), step=1.0,
                                      help="Normal fasting glucose: < 100 mg/dL. Pre-diabetes: 100-125 mg/dL. Diabetes: ≥ 126 mg/dL.")
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40.0, max_value=180.0, value=float(dv["bp"]), step=1.0,
                                             help="Normal diastolic BP is typically 60 - 80 mm Hg.")
            skin_thickness = st.number_input("Triceps Skinfold Thickness (mm)", min_value=5.0, max_value=99.0, value=float(dv["skin"]), step=1.0,
                                             help="Estimate of subcutaneous body fat.")

        with c2:
            insulin = st.number_input("2-Hour Serum Insulin (μU/mL)", min_value=10.0, max_value=850.0, value=float(dv["ins"]), step=1.0,
                                      help="Normal fasting insulin: 15 - 160 μU/mL.")
            bmi = st.number_input("Body Mass Index - BMI (kg/m²)", min_value=12.0, max_value=65.0, value=float(dv["bmi"]), step=0.1,
                                  help="Normal BMI: 18.5 - 24.9. Overweight: 25 - 29.9. Obese: ≥ 30.")
            pedigree = st.number_input("Diabetes Pedigree Function (Genetic Score)", min_value=0.05, max_value=2.5, value=float(dv["ped"]), step=0.01,
                                       help="Genetic score quantifying familial diabetes history.")
            age = st.number_input("Patient Age (Years)", min_value=10, max_value=110, value=int(dv["age"]), step=1)
            
        submitted = st.form_submit_button("🩺 Run Diabetes Prognosis", use_container_width=True)

    if submitted:
        input_data = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree, "Age": age
        }
        res = predictor.predict_diabetes(input_data)
        
        st.divider()
        st.markdown("### 📊 Diagnostic Evaluation Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(create_gauge_chart(res["probability"], "Diabetes Risk Score"), use_container_width=True)
            
        with res_col2:
            if res["is_high_risk"]:
                st.markdown(f"""
                <div class="risk-high">
                    <h3 style="color:#d90429; margin-top:0;">🚨 High Risk: {res['label']}</h3>
                    <p><b>Estimated Probability:</b> {res['probability']}%</p>
                    <p>Clinical features indicate an elevated probability of impaired glucose tolerance or diabetes mellitus.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3 style="color:#2ec4b6; margin-top:0;">✅ Low Risk: {res['label']}</h3>
                    <p><b>Estimated Probability:</b> {res['probability']}%</p>
                    <p>Patient health metrics are consistent with standard metabolic benchmarks.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("##### 🔍 Critical Parameter Observations:")
            obs = []
            if glucose >= 126:
                obs.append("⚠️ **Fasting Glucose is significantly elevated (≥ 126 mg/dL)**")
            elif glucose >= 100:
                obs.append("⚠️ **Fasting Glucose is in the pre-diabetic window (100-125 mg/dL)**")
            if bmi >= 30:
                obs.append("⚠️ **BMI is in the obese range (≥ 30 kg/m²)**")
            elif bmi >= 25:
                obs.append("ℹ️ **BMI is in the overweight range (25-29.9 kg/m²)**")
            if pedigree >= 0.7:
                obs.append("⚠️ **Strong genetic predisposition (Pedigree score > 0.7)**")
            if not obs:
                obs.append("✨ All key metabolic indicators fall within normal clinical guidelines.")
            for o in obs:
                st.write(o)

# -------------------------------------------------------------
# PAGE 5: MODEL BENCHMARKS & ANALYTICS
# -------------------------------------------------------------
elif app_mode == "📊 Model Benchmarks & Analytics":
    st.markdown("## 📊 Machine Learning Model Benchmarks & Metrics")
    st.markdown("Comparative performance evaluation across 6 classification algorithms trained on the diabetes dataset.")
    
    selected_disease = st.selectbox("Select Disease Model to Inspect:", ["diabetes"],
                                    format_func=lambda x: {"diabetes": "Diabetes Mellitus"}[x])
    
    disease_meta = metadata.get(selected_disease, {})
    if disease_meta:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        best_m = disease_meta.get("best_metrics", {})
        col_m1.metric("🏆 Champion Model", disease_meta.get("best_model_name", "N/A"))
        col_m1.caption(f"Selected for deployment")
        col_m2.metric("🎯 Test Accuracy", f"{best_m.get('test_accuracy', 0)}%")
        col_m3.metric("📈 F1-Score", f"{best_m.get('f1_score', 0)}%")
        col_m4.metric("🌟 ROC-AUC", f"{best_m.get('roc_auc', 0)}%")
        
        st.divider()
        
        # Comparison Bar Chart across all models
        all_models = disease_meta.get("all_model_results", {})
        chart_data = []
        for m_name, m_vals in all_models.items():
            chart_data.append({
                "Algorithm": m_name,
                "Accuracy (%)": m_vals["test_accuracy"],
                "F1-Score (%)": m_vals["f1_score"],
                "ROC-AUC (%)": m_vals["roc_auc"],
                "5-Fold CV (%)": m_vals["cv_accuracy_mean"]
            })
        df_chart = pd.DataFrame(chart_data)
        
        st.markdown("#### 📈 Multi-Algorithm Tournament Results")
        fig_bar = px.bar(
            df_chart,
            x="Algorithm",
            y=["Accuracy (%)", "F1-Score (%)", "ROC-AUC (%)", "5-Fold CV (%)"],
            barmode="group",
            title=f"Performance Comparison for {selected_disease.title()} Prediction",
            color_discrete_sequence=["#0077b6", "#2ec4b6", "#e76f51", "#f4a261"]
        )
        fig_bar.update_layout(yaxis_range=[50, 100], height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Feature Importance
        feat_imp = disease_meta.get("feature_importance", {})
        if feat_imp:
            st.markdown("#### 🔍 Top Predictive Features / Biomarkers")
            top_feats = list(feat_imp.items())[:10]
            df_imp = pd.DataFrame(top_feats, columns=["Feature", "Relative Importance (%)"]).sort_values("Relative Importance (%)", ascending=True)
            
            fig_imp = px.bar(
                df_imp,
                x="Relative Importance (%)",
                y="Feature",
                orientation="h",
                color="Relative Importance (%)",
                color_continuous_scale="Blues",
                title=f"Top Biomarkers Influencing {selected_disease.title()} Classification"
            )
            fig_imp.update_layout(height=360)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        # Detailed Table
        st.markdown("#### 📋 Detailed Metrics Breakdown")
        st.dataframe(df_chart.set_index("Algorithm"), use_container_width=True)

# -------------------------------------------------------------
# PAGE 6: ABOUT & MEDICAL DISCLAIMER
# -------------------------------------------------------------
elif app_mode == "ℹ️ Medical Disclaimer & About":
    st.markdown("## ℹ️ About MedPredict AI & Medical Safety Guidelines")
    
    st.warning("""
    ### ⚠️ Medical & Clinical Disclaimer
    **MedPredict AI is an educational, research, and diagnostic screening support demonstration.**
    - It is **NOT** a certified medical diagnostic device and does **NOT** substitute for professional medical consultation, clinical diagnosis, laboratory testing, or physician treatment.
    - If you are experiencing symptoms or health concerns, immediately seek advice from a licensed medical healthcare professional.
    - Machine learning models reflect patterns derived from historical benchmark cohorts and may exhibit demographic or epidemiological variance in real-world clinical populations.
    """)
    
    st.markdown("""
    ### 🔬 Scientific Data Sources & Citations:
    1. **Diabetes Mellitus Dataset**: National Institute of Diabetes and Digestive and Kidney Diseases (Pima Indians Cohort).
    
    ### 🛠️ Architecture & Technology Stack:
    - **Modeling Core**: Scikit-Learn (`RandomForestClassifier`, `AdaBoostClassifier`, `StandardScaler`, `LogisticRegression`, `SVC`, `KNeighborsClassifier`)
    - **Inference Runtime**: Python 3.13 + NumPy + Pandas
    - **Visualization**: Plotly Interactive Scientific Charts (`go.Indicator` Gauges, `px.bar`)
    - **Frontend**: Streamlit Cloud / Desktop Web Framework
    """)
