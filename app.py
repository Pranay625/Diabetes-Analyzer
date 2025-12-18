import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Diabetes Risk Analyzer", page_icon="🩸", layout="centered")

# Enhanced Dark Mode Theme + Pill-Shaped Active Tab
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    /* Top navbar (header) - make it fully black */
    header[data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* All general text */
    h1, h2, h3, h4, h5, h6, p, div, label, span, li {
        color: #ffffff !important;
    }
    
    /* Slider values */
    .stSlider > div > div > div > div {
        color: #ffffff;
    }
    
    /* Help tooltip */
    .stTooltip, .stTooltipContent {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333;
    }
    
    /* Button style */
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        width: 100%;
        height: 3.5em;
        font-size: 1.2rem;
    }
    
    /* Success and Error boxes */
    .stSuccess, .stError {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Disclaimer */
    .stAlert {
        background-color: #333333;
        color: #ffffff;
        border-radius: 10px;
    }
    
    /* Links */
    a {
        color: #ff6b6b !important;
    }
    
    /* Matplotlib toolbar fix */
    .matplotlib-toolbar, .mpl-toolbar {
        background-color: #000000 !important;
    }
    .mpl-toolbar button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    .mpl-toolbar button:hover {
        background-color: #222222 !important;
    }
    .mpl-tooltip {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }
    .mpl-toolbar svg, .mpl-toolbar path {
        fill: #ffffff !important;
    }
    
    figure {
        background-color: #000000 !important;
    }

    /* Larger Tabs Container */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[role="tablist"] {
        padding: 1.5rem 0 !important;
        background-color: #111111;
        border-radius: 16px;
        margin: 2.5rem 0 2rem 0;
    }

    /*/* Tab Buttons */
    button[role="tab"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        padding: 1rem 2rem !important;
        font-family: 'Helvetica Neue', 'Arial', sans-serif !important;
        letter-spacing: 0.5px;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.3s ease;
    }*/

    

    /* Active Tab - Pill Shape Red Background */
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #d32f2f, #b71c1c) !important;
        color: white !important;
        border-radius: 50px !important;
        box-shadow: 0 6px 20px rgba(211, 47, 47, 0.4) !important;
    }

    /* Hover Effect */
    button[role="tab"]:hover {
        background-color: #333333 !important;
        color: white !important;
    }

    /* Remove Default Underline */
    div[role="tablist"] > div {
        border-bottom: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/xgboost_model.json')
    return scaler, model

scaler, model = load_model()

# Branded Header - line moved up
st.markdown("""
    <div style="text-align: center; padding: 1rem 0.5rem; border-bottom: 1px solid #333;">
        <h1 style="margin: 0; color: #d32f2f;">🩸 Diabetes Risk Analyzer</h1>
        <p style="margin: 0.5rem 0 0 0; color: #aaaaaa;">by Pranay Rajesh | v2.0</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**AI-powered risk assessment using XGBoost + SHAP**")
st.info("⚠️ Educational tool only. Not a medical diagnosis. Consult a doctor for health advice.")

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar inputs
st.sidebar.header("Enter Your Health Data")

def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
    if glucose == 0:
        st.sidebar.warning("Glucose cannot be 0. Take 120")
        glucose = 120
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    if skin == 0:
        st.sidebar.warning("SkinThickness cannot be 0. Take 29mm for average.")
        skin = 29
    
    insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 85)
    if insulin == 0:
        st.sidebar.warning("Insulin cannot be 0. Take 156 mu U/ml for average.") 
        insulin = 156  

    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 30.0, step=0.1)

    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.420, 0.470, step=0.001)
    if dpf == 0:
        st.sidebar.warning("DPF cannot be 0. Take 0.47.")
        dpf=0.47
    age = st.sidebar.slider('Age (years)', 21, 81, 33)

    data = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    return pd.DataFrame(data, index=[0]), glucose, bmi, age

df_input, glucose_val, bmi_val, age_val = user_input_features()

# Averages
averages = {
    'Pregnancies': 3.85, 'Glucose': 120.89, 'BloodPressure': 69.11, 'SkinThickness': 20.54,
    'Insulin': 79.80, 'BMI': 31.99, 'DiabetesPedigreeFunction': 0.471, 'Age': 33.24
}

# Prediction
if st.sidebar.button("🔍 Predict My Risk", use_container_width=True):
    with st.spinner("Analyzing your health data..."):
        input_scaled = scaler.transform(df_input)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.session_state.history.append({
            'probability': probability,
            'risk_level': "High" if probability > 0.7 else "Moderate" if probability > 0.3 else "Low"
        })

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Result", "🔍 Why This Prediction", "💡 Recommendations", "ℹ️ About"])

    with tab1:
        st.markdown("<h2 style='text-align: center;'>Your Diabetes Risk</h2>", unsafe_allow_html=True)
        st.metric(
            label="Diabetes Risk Probability",
            value=f"{probability:.1%}",
            delta=f"{'High' if probability > 0.7 else 'Moderate' if probability > 0.3 else 'Low'} Risk"
        )
        if prediction == 1:
            st.error("**High Risk Detected**")
        else:
            st.success("**Low Risk**")

    with tab2:
        st.markdown("### 1. How each value affects the risk")
        st.markdown("🔵 Blue = Lowers risk &nbsp;&nbsp; 🔴 Red = Raises risk")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        feature_names = df_input.columns.tolist()
        
        plt.rcParams.update({'figure.facecolor': '#000000', 'axes.facecolor': '#000000', 'text.color': 'white',
                             'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
        
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                             data=df_input.iloc[0].values, feature_names=feature_names), show=False)
        st.pyplot(fig1)
        
        st.markdown("### 2. Your values vs. average")
        
        comparison_df = pd.DataFrame({'Your Value': df_input.iloc[0], 'Average Value': averages})
        comparison_df['Difference'] = comparison_df['Your Value'] - comparison_df['Average Value']
        comparison_df = comparison_df.sort_values('Difference', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(feature_names))
        width = 0.35
        ax2.bar(x - width/2, comparison_df['Your Value'], width, label='Your Value', color='#ff6b6b')
        ax2.bar(x + width/2, comparison_df['Average Value'], width, label='Average', color='#4ecdc4')
        ax2.set_facecolor('#000000')
        fig2.patch.set_facecolor('#000000')
        ax2.grid(axis='y', alpha=0.3, color='gray')
        ax2.set_ylabel('Value', color='white')
        ax2.set_title('Your Health Metrics vs. Average', color='white')
        ax2.tick_params(colors='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df.index, rotation=45, ha='right', color='white')
        ax2.legend(facecolor='#000000', labelcolor='white')
        st.pyplot(fig2)

    with tab3:
        st.markdown("### Personalized Recommendations")
        if glucose_val > 140:
            st.warning("• High glucose — reduce sugar intake and increase activity")
        if bmi_val > 30:
            st.warning("• High BMI — aim for balanced diet and regular exercise")
        if age_val > 45:
            st.info("• Age factor — regular check-ups recommended")
        if prediction == 1:
            st.error("Overall high risk — consult a doctor soon")
        else:
            st.success("Low risk — maintain healthy lifestyle!")

    with tab4:
        st.markdown("### About This Tool")
        st.markdown("""
        - **Model**: Tuned XGBoost classifier (~83% accuracy)
        - **Explainability**: SHAP values for transparency
        - **Dataset**: Pima Indians Diabetes Database
        - **Features**: 8 medical predictors
        - **Limitations**: Educational only — not for clinical use
        """)
        st.markdown("Source code: [GitHub](https://github.com/Pranay625/Diabetes-Analyzer)")

# History in sidebar
with st.sidebar.expander("📜 Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            st.write(f"**Prediction {len(st.session_state.history)-i}**")
            st.write(f"Risk: {entry['risk_level']} ({entry['probability']:.1%})")
            st.write("---")
    else:
        st.write("No predictions yet")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using XGBoost + SHAP | [GitHub Repo](https://github.com/Pranay625/Diabetes-Analyzer) | Educational Project")
