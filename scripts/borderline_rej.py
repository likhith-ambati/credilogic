import pandas as pd
import joblib
import os

def find_borderline_candidates(top_n=50):
    print("🔍 Scanning CrediLogic Database for Borderline Rejects...")
    
    # 1. Load Assets
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    # 2. Calculate Probabilities for the entire dataset
    # (Using batch prediction for speed)
    X = df[features]
    p_l = lgbm.predict_proba(X)[:, 1]
    p_x = xgb_m.predict_proba(X)[:, 1]
    df['RISK_SCORE'] = (p_l * 0.55) + (p_x * 0.45)
    
    # 3. Calculate Dynamic Thresholds
    # Note: Assuming Accommodative market (Rate < 5.0%) for this baseline scan
    def get_base_threshold(row):
        if row['ANNUITY_INCOME_RATIO'] > 0.18 or row['CREDIT_INCOME_RATIO'] > 6.5:
            return 0.08
        elif row['ANNUITY_INCOME_RATIO'] < 0.06 and row['EMPLOYMENT_YEARS'] > 10:
            return 0.15
        else:
            return 0.11

    df['THRESHOLD'] = df.apply(get_base_threshold, axis=1)
    
    # 4. Calculate the Margin (How much they failed by)
    df['FAILURE_MARGIN'] = df['RISK_SCORE'] - df['THRESHOLD']
    
    # 5. Filter for ONLY Rejects (Margin >= 0)
    # Filter for Rejects with a margin of less than or equal to 0.02 (2%)
    rejects = df[(df['FAILURE_MARGIN'] >= 0.0001) & (df['FAILURE_MARGIN'] <= 0.02)].copy()
    
    # 6. Sort by lowest margin first
    borderline_rejects = rejects.sort_values(by='FAILURE_MARGIN', ascending=True)
    
    # Output the results
    print(f"\nTop {top_n} Borderline Candidates (Easiest to save with restructuring):")
    print("-" * 75)
    print(f"{'Candidate ID':<15} | {'Risk Score':<12} | {'Threshold':<12} | {'Margin':<10}")
    print("-" * 75)
    
    for _, row in borderline_rejects.head(top_n).iterrows():
        print(f"{int(row['SK_ID_CURR']):<15} | {row['RISK_SCORE']:.4f}       | {row['THRESHOLD']:.4f}       | +{row['FAILURE_MARGIN']:.4f}")
        
    print("-" * 75)
    print("💡 Tip: Test these IDs in the Streamlit UI to see the Path to Approval engine in action.")

if __name__ == "__main__":
    find_borderline_candidates()