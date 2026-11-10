import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def prune_features(df_path):
    print("Starting Clean Feature Pruning...")
    df = pd.read_pickle(df_path)
    
    # 1. Create a strict 80/20 split immediately
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df['TARGET'], random_state=42)
    
    X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = train_df['TARGET']
    
    # 2. Train a temporary model ONLY on the training split to find importance
    temp_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    temp_model.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': temp_model.feature_importances_
    })
    
    # Drop features with zero importance
    zero_imp = feature_importances[feature_importances['importance'] == 0]['feature'].tolist()
    
    # 3. Handle Correlation on the training set
    corr_matrix = X.drop(columns=zero_imp).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    final_features = X.drop(columns=zero_imp + to_drop).columns.tolist()
    print(f"✅ Selection Complete: {len(final_features)} Golden Features identified.")
    
    joblib.dump(final_features, 'models/golden_features.pkl')
    return final_features

if __name__ == "__main__":
    prune_features('data/processed/train_engineered.pkl')