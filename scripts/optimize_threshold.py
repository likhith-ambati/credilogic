import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def find_optimal_threshold():
    print("🛡️ Loading Professional Ensemble for Threshold Optimization...")
    df = pd.read_pickle('data/processed/train_engineered.pkl')
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')

    # Use the same 20% test split as the training script
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.2, stratify=df['TARGET'], random_state=42
    )

    # 1. Get predictions from the two models
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_m.predict_proba(X_test)[:, 1]

    # 2. Apply the EXACT 55/45 blend used in the App and train_final.py
    final_probs = (p_l * 0.55) + (p_x * 0.45)

    # 3. Calculate Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, final_probs)
    
    # Calculate F1-Score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n✅ Professional Optimal Threshold: {best_threshold:.4f}")
    print(f"✅ Professional Best F1-Score: {f1_scores[best_idx]:.4f}")

    # Plot the trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', color='red')
    plt.plot(thresholds, recall[:-1], label='Recall', color='green')
    plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimal: {best_threshold:.2f}')
    plt.title('Professional Ensemble: Precision-Recall Trade-off')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # Save the threshold for the App to use
    joblib.dump(best_threshold, 'models/optimal_threshold.pkl')

if __name__ == "__main__":
    find_optimal_threshold()