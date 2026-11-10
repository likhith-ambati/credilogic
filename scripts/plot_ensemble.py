import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def generate_lift_graph():
    print("📈 Generating CrediLogic Architecture Justification (Ensemble Lift)...")
    
    # 1. Load Assets
    try:
        lgbm = joblib.load('models/lgbm_model.pkl')
        xgb_m = joblib.load('models/xgb_model.pkl')
        features = joblib.load('models/golden_features.pkl')
        df = pd.read_pickle("data/processed/train_engineered.pkl")
    except Exception as e:
        print(f"Error loading assets: {e}")
        return
    
    # 2. Honest Validation Split (Unseen Data)
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.2, stratify=df['TARGET'], random_state=42
    )

    # 3. Calculate Probabilities
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    p_xgb = xgb_m.predict_proba(X_test)[:, 1]
    
    # The 55/45 CrediLogic Blend
    p_ensemble = (p_lgbm * 0.55) + (p_xgb * 0.45)

    # 4. Calculate ROC Metrics
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, p_lgbm)
    auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    fpr_ens, tpr_ens, _ = roc_curve(y_test, p_ensemble)
    auc_ens = auc(fpr_ens, tpr_ens)

    # 5. Render Professional Visual
    plt.figure(figsize=(10, 7))
    
    # Plot Individual Models (Thinner, muted colors)
    plt.plot(fpr_xgb, tpr_xgb, color='#e67e22', linestyle='--', linewidth=2, 
             label=f'Pure XGBoost (AUC = {auc_xgb:.4f})')
    plt.plot(fpr_lgbm, tpr_lgbm, color='#3498db', linestyle=':', linewidth=2, 
             label=f'Pure LightGBM (AUC = {auc_lgbm:.4f})')
    
    # Plot The Ensemble (Thick, bold brand color)
    plt.plot(fpr_ens, tpr_ens, color='#002e5d', linestyle='-', linewidth=3.5, 
             label=f'CrediLogic 55/45 Blend (AUC = {auc_ens:.4f})')
    
    # Plot the Random Guess Line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting and Presentation
    plt.title("Architecture Justification: Ensemble Lift", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("False Positive Rate (Incorrectly Flagged as Risk)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Correctly Identified Defaults)", fontsize=12, labelpad=10)
    
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True, borderpad=1)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add a text annotation to highlight the "Lift" explicitly
    lift_pct = ((auc_ens - max(auc_lgbm, auc_xgb)) / max(auc_lgbm, auc_xgb)) * 100
    if lift_pct > 0:
        plt.text(0.6, 0.4, f"Ensemble Lift: +{lift_pct:.2f}% over best individual model", 
                 fontsize=11, color='#002e5d', style='italic',
                 bbox=dict(facecolor='white', edgecolor='#002e5d', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_lift_graph()