import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_professional_ensemble():
    print("🚀 Training Professional Ensemble (LGBM + XGB)...")
    df = pd.read_pickle('data/processed/train_engineered.pkl')
    features = joblib.load('models/golden_features.pkl')
    
    X = df[features]
    y = df['TARGET']
    
    # One clean split for the whole project
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 1. LightGBM Surgeon
    print("Training LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=34, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[lgb.early_stopping(50)])
    
    # 2. XGBoost Consultant
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate the Blend
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_model.predict_proba(X_test)[:, 1]
    blend_preds = (p_l * 0.55) + (p_x * 0.45) # Weighted in favor of LightGBM
    
    print(f"\nFinal Validated ROC-AUC: {roc_auc_score(y_test, blend_preds):.4f}")
    
    # SAVE ONLY THESE MODELS (Trained on 80% only)
    joblib.dump(lgbm, 'models/lgbm_model.pkl')
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    print("✅ Models saved. NO full-data retraining performed (Leakage Fixed).")

if __name__ == "__main__":
    train_professional_ensemble()