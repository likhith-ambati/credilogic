import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def train_logistic_baseline():
    print("📈 Training Logistic Regression Baseline...")
    
    # 1. Load Data
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    X = df[features]
    y = df['TARGET']
    
    # 2. Build a Robust Pipeline
    # - SimpleImputer: Fills missing values (NaNs) with the column's median.
    # - StandardScaler: Scales all numbers so massive incomes don't overpower tiny ratios.
    # - LogisticRegression: The linear baseline model.
    lr_pipeline = make_pipeline(
        SimpleImputer(strategy='median'), 
        StandardScaler(), 
        LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    )
    
    # 3. Train the Model
    lr_pipeline.fit(X, y)
    
    # 4. Save the Pipeline
    joblib.dump(lr_pipeline, 'models/lr_baseline.pkl')
    print("✅ Logistic Regression Baseline saved to models/lr_baseline.pkl")

if __name__ == "__main__":
    train_logistic_baseline()