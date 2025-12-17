import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Diabetes Risk Analyzer", page_icon="🩸", layout="centered")

# Enhanced Dark Mode Theme + Full Fix for Matplotlib Toolbar Buttons
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
    .stTooltip {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    .stTooltipContent {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333;
    }
    
    /* Button style */
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        width: 100%;
        height: 3em;
    }
    
    /* Success and Error boxes */
    .stSuccess {
        background-color: #1b5e20;
        color: #c8e6c9;
    }
    .stError {
        background-color: #8f0000;
        color: #ffcdd2;
    }
    
    /* Disclaimer */
    .stAlert {
        background-color: #333333;
        color: #ffffff;
    }
    
    /* Links */
    a {
        color: #ff6b6b !important;
    }
    
    /* --- FIX: Matplotlib toolbar buttons (Fullscreen, Reset, Download, etc.) --- */
    /* Full black background for toolbar */
    .matplotlib-toolbar, .mpl-toolbar {
        background-color: #000000 !important;
    }
    
    /* Black background for all toolbar buttons */
    .mpl-toolbar button, button.matplotlib-button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    /* Hover effect */
    .mpl-toolbar button:hover, button.matplotlib-button:hover {
        background-color: #222222 !important;
    }
    
    /* Tooltip (Fullscreen label) - black with white text */
    .mpl-tooltip {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }
    
    /* Icons inside buttons - ensure white */
    .mpl-toolbar svg, .mpl-toolbar path {
        fill: #ffffff !important;
    }
    
    /* Plot background */
    figure, .js-plotly-plot {
        background-color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/xgboost_model.json')
    return scaler, model

scaler, model = load_model()

# Title and disclaimer
st.title("🩸 Diabetes Risk Analyzer")
st.markdown("**AI-powered risk assessment based on your health metrics**")
st.info("⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.")

# Sidebar inputs
st.sidebar.header("Enter Your Health Data")

def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120, help="Plasma glucose concentration 2 hours after oral glucose tolerance test")
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 85)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 30.0, step=0.1)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.420, 0.470, step=0.001, help="Genetic risk score based on family history")
    age = st.sidebar.slider('Age (years)', 21, 81, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Dataset averages
averages = {
    'Pregnancies': 3.85,
    'Glucose': 120.89,
    'BloodPressure': 69.11,
    'SkinThickness': 20.54,
    'Insulin': 79.80,
    'BMI': 31.99,
    'DiabetesPedigreeFunction': 0.471,
    'Age': 33.24
}

# Prediction button
if st.sidebar.button("🔍 Predict My Risk", use_container_width=True):
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("📊 Your Diabetes Risk Result")
    
    if prediction == 1:
        risk_level = "High" if probability > 0.7 else "Moderate"
        st.error("**High Risk of Diabetes Detected**")
    else:
        risk_level = "Low"
        st.success("**Low Risk of Diabetes**")
    
    st.write(f"**Probability of Diabetes:** {probability:.1%}")
    st.write(f"**Risk Level:** {risk_level}")
    
    # Patient-Friendly Explanations
    st.markdown("<h2 style='color: #ffffff; margin-bottom: 0;'>🔍 Why This Prediction?</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #888888; margin-top: 5px; font-weight: normal;'>?</h4>", unsafe_allow_html=True)
    
    # Graph 1 Explanation
    st.markdown("""
        <div style='font-size: 1.4rem; font-weight: bold; margin: 20px 0 10px 0;'>
        1. How each of your values affects the risk
        </div>
        <div style='font-size: 1.2rem; color: #cccccc; margin-bottom: 20px;'>
        🔵 Blue = Lowers risk &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Red = Raises risk
        </div>
    """, unsafe_allow_html=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    feature_names = df_input.columns.tolist()
    
    # Dark theme for SHAP plot
    plt.rcParams['figure.facecolor'] = '#000000'
    plt.rcParams['axes.facecolor'] = '#000000'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=df_input.iloc[0].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig1, bbox_inches='tight')
    
    # Graph 2 Explanation
    st.markdown("""
        <div style='font-size: 1.4rem; font-weight: bold; margin: 40px 0 10px 0;'>
        2. How your values compare to average people in the study
        </div>
    """, unsafe_allow_html=True)
    
    comparison_df = pd.DataFrame({
        'Your Value': df_input.iloc[0],
        'Average Value': [averages[col] for col in feature_names]
    })
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
    
    plt.tight_layout()
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.caption("Built with XGBoost + SHAP • Educational Project • Data: Pima Indians Diabetes Dataset")