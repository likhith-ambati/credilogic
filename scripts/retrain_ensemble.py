import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score

def retrain_production_models():
    print("🔥 Loading Balanced Data for Production Retraining...")
    
    # 1. Load the balanced training set and the strict holdout test set
    try:
        X_res, y_res = joblib.load('data/processed/train_balanced.pkl')
        X_test, y_test = joblib.load('data/processed/test_holdout.pkl')
    except FileNotFoundError:
        print("Error: train_balanced.pkl not found. Run resample_data.py first.")
        return

    # 2. Retrain LightGBM (The 55% Component)
    print("🌲 Training Balanced LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        importance_type='gain',
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_res, y_res)
    joblib.dump(lgbm, 'models/lgbm_model.pkl')

    # 3. Retrain XGBoost (The 45% Component)
    print("🚀 Training Balanced XGBoost...")
    xgb_m = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_m.fit(X_res, y_res)
    joblib.dump(xgb_m, 'models/xgb_model.pkl')

    # 4. Final Ensemble Validation
    print("\n" + "="*45)
    print("BALANCED ENSEMBLE PERFORMANCE (55/45 BLEND)")
    print("="*45)
    
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    p_xgb = xgb_m.predict_proba(X_test)[:, 1]
    
    # The CrediLogic Weighted Average
    p_blend = (p_lgbm * 0.55) + (p_xgb * 0.45)
    
    # Hard predictions for Recall/F1 (using 0.11 threshold)
    y_preds = (p_blend >= 0.11).astype(int)
    
    auc = roc_auc_score(y_test, p_blend)
    rec = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Recall:  {rec:.2%}")
    print(f"F1-Score: {f1:.4f}")
    print("="*45)
    print("✅ Production models updated in /models/")

if __name__ == "__main__":
    retrain_production_models()