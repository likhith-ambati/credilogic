import pandas as pd
import numpy as np
import joblib
import requests
import toml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

def get_live_macro_rate():
    """Retrieves live Federal Funds Rate for macro-scaling."""
    try:
        # Check for secrets in root-level toml
        secrets_path = "secrets.toml"
        if os.path.exists(secrets_path):
            api_key = toml.load(secrets_path)["FRED_API_KEY"]
        else:
            api_key = None
        
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url, timeout=5)
        return float(response.json()['observations'][0]['value'])
    except Exception:
        # Fallback to verified baseline if API is unavailable
        return 3.64

def run_macro_visual_audit():
    print("🛡️ Initializing Macro-Aware Performance Audit (Unseen Data Only)...")
    
    # 1. System Assets & Live Macro Context
    live_rate = get_live_macro_rate()
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    # 2. Honest 20% Unseen Split (Replicating training conditions)
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], 
        test_size=0.2, 
        stratify=df['TARGET'], 
        random_state=42
    )

    # 3. Generate Ensemble Probabilities (55% LGBM / 45% XGB blend)
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_m.predict_proba(X_test)[:, 1]
    final_probs = (p_l * 0.55) + (p_x * 0.45)
    
    # 4. Macro-Adaptive Threshold Logic
    test_meta = df.loc[X_test.index]
    def get_macro_threshold(row):
        # Determine base segment thresholds
        if row['ANNUITY_INCOME_RATIO'] > 0.18 or row['CREDIT_INCOME_RATIO'] > 6.5:
            base = 0.08
        elif row['ANNUITY_INCOME_RATIO'] < 0.06 and row['EMPLOYMENT_YEARS'] > 10:
            base = 0.15
        else:
            base = 0.11
        
        # Apply 15% strictness multiplier (0.85) if interest rates exceed 5.0%
        return round(base * 0.85, 3) if live_rate > 5.0 else base

    dynamic_thresholds = test_meta.apply(get_macro_threshold, axis=1)
    predictions = (final_probs >= dynamic_thresholds).astype(int)

    # 5. Output Results
    roc_auc = roc_auc_score(y_test, final_probs)
    print("\n" + "="*60)
    print("📈 CREDILOGIC: MACRO-INTEGRATED PERFORMANCE REPORT")
    print("="*60)
    print(f"Current Macro Rate: {live_rate}% | Status: {'RESTRICTIVE' if live_rate > 5.0 else 'ACCOMMODATIVE'}")
    print(f"Ensemble AUC:       {roc_auc:.4f}")
    print(f"System Recall:      {recall_score(y_test, predictions):.4f}")
    print("-" * 60)
    print("Detailed Classification Report (Adaptive Logic):")
    print(classification_report(y_test, predictions))
    print("="*60)
    
    # 6. Visual Performance Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title(f'Confusion Matrix (Rate: {live_rate}%)', fontweight='bold')
    ax1.set_xticklabels(['Approved', 'Rejected']); ax1.set_yticklabels(['Approved', 'Rejected'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, final_probs)
    ax2.plot(fpr, tpr, color='#002e5d', lw=3, label=f'ROC (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax2.set_title('ROC Curve: Unseen Test Performance', fontweight='bold')
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_macro_visual_audit()