# 🩸 Diabetes Risk Analyzer

An advanced, explainable AI-powered web app for diabetes risk assessment using the Pima Indians Diabetes Dataset.

## 🚀 Features
- Accurate prediction using tuned **XGBoost** (~83% accuracy)
- Patient-friendly **SHAP waterfall plot** showing why the prediction was made
- Comparison of your values vs. dataset averages
- Beautiful **dark mode** UI built with Streamlit
- Real feature names and values (no technical jargon)
- Fully responsive and mobile-friendly

## 🩺 How to Use
1. Enter your health metrics in the sidebar
2. Click "Predict My Risk"
3. View your risk level + detailed explanations

**Disclaimer**: This is an educational ML project. Not for medical diagnosis. Always consult a doctor.

## 📊 Tech Stack
- Python
- XGBoost + SHAP for explainability
- Streamlit for web app
- Matplotlib for visualizations

## 📂 Project Structure
- `app.py`: Main Streamlit application
- `src/train_model.py`: Model training script
- `data/raw/diabetes.csv`: Dataset
- `models/`: Saved model and scaler

## 🔧 Local Setup
```bash
git clone https://github.com/YOUR-USERNAME/diabetes-analyzer.git
cd diabetes-analyzer
pip install -r requirements.txt
python src/train_model.py   # Train model (run once)
streamlit run app.py