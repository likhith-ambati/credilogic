import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def generate_visual_reports():
    print("📈 Generating CrediLogic Executive Analysis...")
    
    # 1. Load Assets
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    # 2. Split (Honest 20%)
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.2, stratify=df['TARGET'], random_state=42
    )

    # 3. Calculate Ensemble Probabilities (55/45 Blend)
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_m.predict_proba(X_test)[:, 1]
    probs = (p_l * 0.55) + (p_x * 0.45)

    # --- PLOT 1: PROBABILITY CALIBRATION ---
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    brier = brier_score_loss(y_test, probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(prob_pred, prob_true, "s-", color='#002e5d', label=f"CrediLogic Ensemble (Brier={brier:.4f})")
    plt.ylabel("Actual Default Rate")
    plt.xlabel("Predicted Probability of Default")
    plt.title("CrediLogic: Probability Calibration Curve", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # --- PLOT 2: DYNAMIC THRESHOLD HEATMAP ---
    macro_rates = np.linspace(3.0, 7.0, 5) # Simulating interest rates
    segments = {
        "High Stress": 0.08, 
        "Standard": 0.11, 
        "High Stability": 0.15
    }
    
    heatmap_data = []
    for rate in macro_rates:
        row = []
        for segment, base in segments.items():
            # Apply the 15% strictness multiplier logic
            val = round(base * 0.85, 3) if rate > 5.0 else base
            row.append(val)
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=[f"{r:.1f}%" for r in macro_rates], columns=segments.keys())
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Decision Threshold'})
    plt.title("Policy Matrix: Dynamic Threshold vs. Macro Rates", fontsize=14, fontweight='bold')
    plt.xlabel("Applicant Risk Segment")
    plt.ylabel("Federal Funds Rate (FRED API)")
    plt.show()

if __name__ == "__main__":
    generate_visual_reports()