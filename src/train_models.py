import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_candidate_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

def train_and_evaluate_disease(disease_name, df, feature_cols, target_col, zero_impute_cols=None):
    print(f"\n=======================================================")
    print(f"  Training & Evaluating Models for: {disease_name.upper()}")
    print(f"=======================================================")
    
    data = df.copy()
    
    # Impute biological zero values if specified (e.g. for diabetes)
    if zero_impute_cols:
        for col in zero_impute_cols:
            if col in data.columns:
                median_val = data[data[col] != 0][col].median()
                data[col] = data[col].replace(0, median_val)
                print(f"  [Imputed] Zeros in '{col}' replaced with median ({median_val:.2f})")
                
    X = data[feature_cols]
    y = data[target_col]
    
    # Store feature descriptive stats for UI validation & presets
    stats = {}
    for col in feature_cols:
        stats[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
            "std": float(X[col].std()),
            "q25": float(X[col].quantile(0.25)),
            "q75": float(X[col].quantile(0.75))
        }

    # Split dataset (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate candidate algorithms
    candidates = get_candidate_models()
    results = {}
    best_model_name = None
    best_score = -1
    best_fitted_model = None
    
    for name, model in candidates.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = float(roc_auc_score(y_test, y_test_prob))
        else:
            auc = 0.0
            
        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        precision = float(precision_score(y_test, y_test_pred, zero_division=0))
        recall = float(recall_score(y_test, y_test_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_test_pred).tolist()
        
        # 5-fold cross validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        results[name] = {
            "train_accuracy": round(train_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc * 100, 2),
            "cv_accuracy_mean": round(cv_mean * 100, 2),
            "cv_accuracy_std": round(cv_std * 100, 2),
            "confusion_matrix": cm
        }
        
        print(f"  -> {name:24s} | Acc: {test_acc*100:5.2f}% | F1: {f1*100:5.2f}% | AUC: {auc*100:5.2f}% | CV: {cv_mean*100:5.2f}%")
        
        # Selection criterion: balanced combination of test F1 and test Accuracy
        composite_score = (test_acc * 0.5) + (f1 * 0.5)
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name
            best_fitted_model = model
            
    print(f"\n  [CHAMPION] Best Model for {disease_name}: {best_model_name} (Acc: {results[best_model_name]['test_accuracy']}%, F1: {results[best_model_name]['f1_score']}%)")
    
    # Save best model and scaler
    model_path = os.path.join(MODELS_DIR, f"{disease_name}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{disease_name}_scaler.pkl")
    joblib.dump(best_fitted_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  [SAVED] Saved model to {model_path}")
    print(f"  [SAVED] Saved scaler to {scaler_path}")
    
    # Feature importances if available
    feature_importance = {}
    if hasattr(best_fitted_model, "feature_importances_"):
        importances = best_fitted_model.feature_importances_
        for feat, imp in zip(feature_cols, importances):
            feature_importance[feat] = round(float(imp) * 100, 2)
        # sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(best_fitted_model, "coef_"):
        coefs = np.abs(best_fitted_model.coef_[0])
        total = np.sum(coefs)
        for feat, c in zip(feature_cols, coefs):
            feature_importance[feat] = round(float(c / total) * 100, 2)
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
    return {
        "best_model_name": best_model_name,
        "best_metrics": results[best_model_name],
        "all_model_results": results,
        "feature_columns": feature_cols,
        "feature_importance": feature_importance,
        "feature_stats": stats,
        "samples_count": len(data)
    }

def run_training_pipeline():
    summary_metrics = {}
    
    # 1. Diabetes
    diabetes_df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    diabetes_features = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    summary_metrics["diabetes"] = train_and_evaluate_disease(
        disease_name="diabetes",
        df=diabetes_df,
        feature_cols=diabetes_features,
        target_col="Outcome",
        zero_impute_cols=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    )
    
    # Save overall metadata and comparison metrics
    metadata_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metadata_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n[DONE] Successfully saved model comparison metrics to {metadata_path}")
    print("All models and scalers are trained and ready for deployment!")

if __name__ == "__main__":
    run_training_pipeline()
