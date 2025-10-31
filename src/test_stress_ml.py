# test_stress_ml.py

import joblib
from stress_ml import predict_stress

def test_predict_stress_valid_input():
    sample_input = [0.5, 18, 36.5, 0.2, 98, 1, 7, 72]
    result = predict_stress(sample_input)
    assert result is not None
    print("Predicted Stress Level:", result)

def test_model_files_exist():
    assert joblib.load('../models/stress_model.pkl')
    assert joblib.load('../models/scaler.pkl')
    print("Model and scaler files exist and are loadable.")

if __name__ == "__main__":
    test_predict_stress_valid_input()
    test_model_files_exist()
