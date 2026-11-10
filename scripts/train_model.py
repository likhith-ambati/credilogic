import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

def train_baseline_model():
    print("Loading engineered data...")
    df = pd.read_pickle('data/processed/train_engineered.pkl')
    
    # Check if we have our pruned "Golden Features"
    golden_path = 'models/golden_features.pkl'
    if os.path.exists(golden_path):
        print("Using PRUNED Golden Features for training...")
        features = joblib.load(golden_path)
    else:
        print("Using ALL features for initial training...")
        features = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]

    X = df[features]
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {X_train.shape[1]} features...")
    
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=34,
        colsample_bytree=0.9,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    y_pred = clf.predict_proba(X_test)[:, 1]
    print(f"\nNew LightGBM ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")
    
    joblib.dump(clf, 'models/lgbm_model.pkl')
    print("Re-trained model saved.")

if __name__ == "__main__":
    train_baseline_model()