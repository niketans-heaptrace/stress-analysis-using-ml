# stress_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
data_path = '../data/raw/Stress.csv'  # apna path yahan set karo
df = pd.read_csv(data_path)

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Handle missing values
df = df.dropna()  # ya df.fillna(df.mean())

# Features and target
X = df[['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']]
y = df['sl']

# Detect problem type and encode labels if needed
use_label_encoder = False
if y.dtype == 'object':
    problem_type = 'classification'
    le = LabelEncoder()
    y = le.fit_transform(y)
    use_label_encoder = True
elif len(y.unique()) <= 5:
    problem_type = 'classification'  # numeric 0,1,2 etc.
else:
    problem_type = 'regression'

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train Model
# -----------------------------
if problem_type == 'classification':
    model = RandomForestClassifier(n_estimators=200, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

if problem_type == 'classification':
    print("Problem Type: Classification")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
else:
    print("Problem Type: Regression")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 6. Save Model and Scaler
# -----------------------------
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/stress_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
if use_label_encoder:
    joblib.dump(le, '../models/label_encoder.pkl')

# -----------------------------
# 7. Predict Function
# -----------------------------
def predict_stress(new_data):
    """
    new_data: list of 8 features [sr, rr, t, lm, bo, rem, sh, hr]
    """
    model = joblib.load('../models/stress_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    new_scaled = scaler.transform([new_data])

    # Load label encoder only if it was used
    if use_label_encoder:
        le = joblib.load('../models/label_encoder.pkl')
        pred = model.predict(new_scaled)
        return le.inverse_transform(pred)[0]
    else:
        pred = model.predict(new_scaled)
        return pred[0]

# Example usage
# sample_input = [0.5, 18, 36.5, 0.2, 98, 1, 7, 72]
# print("Predicted Stress Level:", predict_stress(sample_input))
