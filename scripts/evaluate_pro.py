import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, roc_auc_score, accuracy_score

def evaluate_final():
    print("🛡️ Running Honest Evaluation (No Leakage)...")
    df = pd.read_pickle('data/processed/train_engineered.pkl')
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')
    
    # The SAME split used in training
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.2, stratify=df['TARGET'], random_state=42
    )
    
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_m.predict_proba(X_test)[:, 1]
    # Use our 55/45 blend
    final_probs = (p_l * 0.55) + (p_x * 0.45)
    
    threshold = 0.1484 # Your optimized finding
    y_pred = (final_probs >= threshold).astype(int)
    
    print("\n" + "="*40)
    print("HONEST PERFORMANCE REPORT")
    print("="*40)
    print(f"ROC-AUC:  {roc_auc_score(y_test, final_probs):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_final()