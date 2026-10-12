import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self._load_all()

    def _load_all(self):
        # Load metadata
        meta_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        for disease in ["diabetes"]:
            model_path = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
            scaler_path = os.path.join(MODELS_DIR, f"{disease}_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[disease] = joblib.load(model_path)
                self.scalers[disease] = joblib.load(scaler_path)

    def predict_diabetes(self, input_data: dict) -> dict:
        feature_order = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        df_input = pd.DataFrame([[input_data[feat] for feat in feature_order]], columns=feature_order)
        X_scaled = self.scalers["diabetes"].transform(df_input)
        model = self.models["diabetes"]
        
        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        
        return {
            "prediction": pred,
            "label": "Diabetic Tendency Detected" if pred == 1 else "Non-Diabetic / Low Risk",
            "probability": round(prob * 100, 2),
            "is_high_risk": pred == 1 or prob >= 0.5
        }

if __name__ == "__main__":
    predictor = DiseasePredictor()
    print("Testing predictor...")
    sample_diabetes = {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 125, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    print("Diabetes test result:", predictor.predict_diabetes(sample_diabetes))
