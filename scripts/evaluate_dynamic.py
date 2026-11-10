import pandas as pd
import numpy as np
import joblib
import requests
from sklearn import base
import toml
import os
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)

def get_live_macro_rate():
    """Retrieves live Federal Funds Rate from the root secrets.toml."""
    try:
        secrets_path = "secrets.toml"
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            api_key = secrets["FRED_API_KEY"]
        else:
            return 3.64 # Fallback to a baseline rate
            
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['observations'][0]['value'])
    except Exception:
        return 3.64

def run_comprehensive_evaluation():
    print("🛡️ Initializing CrediLogic Full Performance Audit (Blind Holdout Set)...")
    
    # 1. Environment & Asset Loading
    try:
        # CRITICAL: Load only the unseen test set to prevent Data Leakage
        X_test, y_test = joblib.load('data/processed/test_holdout.pkl')
        
        live_rate = get_live_macro_rate()
        lgbm = joblib.load('models/lgbm_model.pkl')
        xgb_m = joblib.load('models/xgb_model.pkl')
        features = joblib.load('models/golden_features.pkl')
        
        # We ensure X_test is a DataFrame for the .apply() logic
        df_test = pd.DataFrame(X_test)
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return
    
    # 2. Ensemble Probabilities
    print(f"Calculating Ensemble Probabilities on {len(df_test)} unseen candidates...")
    # Ensure we only use the Golden Features in the correct order
    p_lgbm = lgbm.predict_proba(df_test[features])[:, 1]
    p_xgb = xgb_m.predict_proba(df_test[features])[:, 1]
    final_probs = (p_lgbm * 0.55) + (p_xgb * 0.45)
    
    # 3. Decision Logic (Adaptive vs Static)
    def calculate_threshold(row):
    # Defensive checks for missing columns
        annuity_ratio = row.get('ANNUITY_INCOME_RATIO', 0.11) # Fallback to neutral
        credit_ratio = row.get('CREDIT_INCOME_RATIO', 4.0)
        emp_years = row.get('EMPLOYMENT_YEARS', 5.0)

        if annuity_ratio > 0.18 or credit_ratio > 6.5:
            base = 0.08
        elif annuity_ratio < 0.06 and emp_years > 10:
            base = 0.15
        else:
            base = 0.10
        
        return base * 0.85 if live_rate > 5.0 else base

    print(f"Applying Adaptive Thresholds based on Macro Rate: {live_rate}%")
    dynamic_thresholds = df_test.apply(calculate_threshold, axis=1)
    
    # Generate Predictions
    static_preds = (final_probs >= 0.10).astype(int) # Using your new 0.10 baseline
    dynamic_preds = (final_probs >= dynamic_thresholds).astype(int)

    # 4. Detailed Metrics Reporting
    print("\n" + "="*70)
    print("📈 CREDILOGIC COMPREHENSIVE PERFORMANCE REPORT")
    print("="*70)
    print(f"Market Environment: {live_rate}% | Threshold Strategy: Adaptive")
    print(f"Sample Size: {len(y_test)} candidates")
    print("-" * 70)
    
    # Static Baseline
    print("\n[CHAMPION: STATIC BASELINE - 0.10]")
    print(f"ROC-AUC:   {roc_auc_score(y_test, final_probs):.4f}")
    print(f"Recall:    {recall_score(y_test, static_preds):.2%}")
    print(f"F1-Score:  {f1_score(y_test, static_preds):.4f}")
    
    print("-" * 70)
    
    # Dynamic Adaptive Engine
    print("\n[CHALLENGER: LIVE ADAPTIVE ENGINE]")
    print(f"ROC-AUC:   {roc_auc_score(y_test, final_probs):.4f}")
    print(f"Recall:    {recall_score(y_test, dynamic_preds):.2%}")
    print(f"F1-Score:  {f1_score(y_test, dynamic_preds):.4f}")
    
    print("\nClassification Report (Adaptive):")
    print(classification_report(y_test, dynamic_preds))
    print("="*70)

if __name__ == "__main__":
    run_comprehensive_evaluation()