import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

def find_recall_sweet_spot():
    print("🔍 Hunting for the Optimal Recall/Accuracy Balance...")
    df = pd.read_pickle('data/processed/train_engineered.pkl')
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')

    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.2, stratify=df['TARGET'], random_state=42
    )

    # Blend predictions (55/45)
    p_l = lgbm.predict_proba(X_test)[:, 1]
    p_x = xgb_m.predict_proba(X_test)[:, 1]
    final_probs = (p_l * 0.55) + (p_x * 0.45)

    # Sweep thresholds to see the trade-off
    results = []
    for t in np.arange(0.10, 0.25, 0.01):
        y_pred = (final_probs >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_report(y_test, y_pred, output_dict=True)['1']['precision']
        results.append({'Threshold': t, 'Accuracy': acc, 'Recall': rec, 'Precision': prec})

    report = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RECALL VS ACCURACY TRADE-OFF")
    print("="*50)
    print(report.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    from sklearn.metrics import recall_score, classification_report as precision_report
    find_recall_sweet_spot()