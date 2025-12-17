

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# Load data
df = pd.read_csv('data/raw/diabetes.csv')

# Preprocessing: Replace invalid zeros
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
df[columns_with_zeros] = df[columns_with_zeros].fillna(df[columns_with_zeros].median())

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline with SMOTE and XGBoost
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

# Hyperparameter tuning
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [4, 6, 8],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__subsample': [0.8, 1.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler separately
os.makedirs('models', exist_ok=True)
joblib.dump(best_model.named_steps['scaler'], 'models/scaler.pkl')
joblib.dump(best_model.named_steps['xgb'], 'models/xgboost_model.json')
joblib.dump(best_model, 'models/full_pipeline.pkl')  # Optional full pipeline

print("\nModel and scaler saved to models/") 
