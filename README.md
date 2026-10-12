# MedPredict AI — Diabetes Risk Prediction

A machine-learning web application for educational diabetes-risk screening using a Streamlit interface.

## Features
- Diabetes risk prediction using clinical indicators.
- One-click example patient profiles.
- Risk probability gauge and observations about key indicators.
- Model benchmark dashboard for diabetes classifiers.

## Project Structure
```text
ML-Project-main/
├── data/
│   └── diabetes.csv
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   └── model_metrics.json
├── src/
│   ├── download_data.py
│   ├── inference.py
│   └── train_models.py
├── app.py
├── requirements.txt
├── run_app.bat
└── train_models.bat
```

## Run the application
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder and run:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python -m streamlit run app.py
   ```

The app should open at `http://localhost:8501`.

## Retrain the model
Run:
```bash
python src/train_models.py
```

## Important disclaimer
This project is for academic and educational demonstration only. It is not a medical diagnostic device and must not replace advice, diagnosis, or treatment from a qualified healthcare professional.
