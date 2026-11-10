import pandas as pd
import joblib
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

def apply_smote_tomek():
    print("⚖️ Initializing SMOTE + Tomek Links Resampling...")
    
    # 1. Load Data
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    X = df[features]
    y = df['TARGET']
    
    # Handle NaNs (SMOTE requires full data)
    X = X.fillna(X.median())
    
    # 2. Split First (CRITICAL: Only resample training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 3. Apply SMOTE-Tomek
    # sampling_strategy=0.5 means synthetic defaults will be half the size of safe loans
    smt = SMOTETomek(sampling_strategy=0.5, random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    
    print(f"📊 Resampling Complete.")
    print(f"   Original 'Default' count: {y_train.sum()}")
    print(f"   New 'Default' count:      {y_res.sum()}")
    
    # 4. Save Balanced Data
    joblib.dump((X_res, y_res), 'data/processed/train_balanced.pkl')
    joblib.dump((X_test, y_test), 'data/processed/test_holdout.pkl')
    print("✅ Balanced training set saved to data/processed/train_balanced.pkl")

if __name__ == "__main__":
    apply_smote_tomek()