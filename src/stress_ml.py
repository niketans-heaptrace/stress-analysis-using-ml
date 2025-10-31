# stress_ml_pro.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)
from scipy.stats import randint

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
data_path = '../data/raw/Stress.csv'
df = pd.read_csv(data_path)
df.dropna(inplace=True)

# ======================================================
# 2️⃣ Preprocessing
# ======================================================
X = df[['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']]
y = df['sl']

use_label_encoder = False
if y.dtype == 'object':
    problem_type = 'classification'
    le = LabelEncoder()
    y = le.fit_transform(y)
    use_label_encoder = True
elif len(y.unique()) <= 5:
    problem_type = 'classification'
else:
    problem_type = 'regression'

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# 3️⃣ Split Dataset
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================================================
# 4️⃣ Hyperparameter Tuning
# ======================================================
if problem_type == 'classification':
    base_model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
else:
    base_model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }

print("🔍 Running Hyperparameter Tuning... (this may take a while)")
search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, 
    n_iter=20, cv=3, n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_train)

model = search.best_estimator_
print("\n✅ Best Parameters:", search.best_params_)

# ======================================================
# 5️⃣ Evaluate Model
# ======================================================
y_pred = model.predict(X_test)

if problem_type == 'classification':
    print("\n📊 Problem Type: Classification")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
else:
    print("\n📊 Problem Type: Regression")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

# ======================================================
# 6️⃣ Feature Importance Visualization
# ======================================================
feature_importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
os.makedirs('../reports', exist_ok=True)
plt.savefig('../reports/feature_importance.png')
plt.show()

# ======================================================
# 7️⃣ Save Model, Scaler, and Encoder
# ======================================================
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/stress_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
if use_label_encoder:
    joblib.dump(le, '../models/label_encoder.pkl')

print("\n💾 Model and Scaler saved successfully!")

# ======================================================
# 8️⃣ Predict Function
# ======================================================
def predict_stress(new_data):
    """
    Predict stress level for new data
    new_data: list of 8 features [sr, rr, t, lm, bo, rem, sh, hr]
    """
    model = joblib.load('../models/stress_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    new_scaled = scaler.transform([new_data])

    try:
        le = joblib.load('../models/label_encoder.pkl')
        pred = model.predict(new_scaled)
        return le.inverse_transform(pred)[0]
    except FileNotFoundError:
        pred = model.predict(new_scaled)
        return float(pred[0])

# Example usage:
# sample_input = [0.5, 18, 36.5, 0.2, 98, 1, 7, 72]
# print("Predicted Stress Level:", predict_stress(sample_input))
