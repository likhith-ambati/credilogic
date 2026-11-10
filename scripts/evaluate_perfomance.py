import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, brier_score_loss, 
                             accuracy_score, precision_score, 
                             recall_score, f1_score)

def generate_comprehensive_report():
    print("🧪 Initializing CrediLogic Comprehensive Evaluation Matrix...")
    
    # 1. Load All Models and Data
    try:
        lgbm = joblib.load('models/lgbm_model.pkl')
        xgb_m = joblib.load('models/xgb_model.pkl')
        lr_pipeline = joblib.load('models/lr_baseline.pkl')
        features = joblib.load('models/golden_features.pkl')
        df = pd.read_pickle("data/processed/train_engineered.pkl")
    except Exception as e:
        print(f"System Error loading assets: {e}")
        return

    # 2. Isolate the Strict Blind Holdout Set
    _, X_test, _, y_test = train_test_split(
        df[features], df['TARGET'], test_size=0.20, stratify=df['TARGET'], random_state=42
    )

    # 3. Generate Continuous Probabilities
    print("Generating predictions across all models...")
    probs = {
        "Linear Baseline (LR)": lr_pipeline.predict_proba(X_test)[:, 1],
        "Pure LightGBM": lgbm.predict_proba(X_test)[:, 1],
        "Pure XGBoost": xgb_m.predict_proba(X_test)[:, 1],
    }
    # Calculate the 55/45 Blend
    probs["CrediLogic 55/45 Blend"] = (probs["Pure LightGBM"] * 0.55) + (probs["Pure XGBoost"] * 0.45)

    # 4. Standard Operating Threshold
    # This is the baseline threshold your Streamlit app uses for standard macro conditions
    THRESHOLD = 0.11 

    # 5. Calculate Metrics
    results = []
    for model_name, p_vals in probs.items():
        # Hard binary predictions based on the threshold
        preds = (p_vals >= THRESHOLD).astype(int)
        
        # Calculate all metrics
        auc = roc_auc_score(y_test, p_vals)
        brier = brier_score_loss(y_test, p_vals)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        results.append({
            "Model": model_name,
            "ROC-AUC": auc,
            "Brier": brier,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1-Score": f1
        })

    # Convert to DataFrame for beautiful tabular formatting
    results_df = pd.DataFrame(results)
    
    # --- TERMINAL REPORT ---
    print("\n" + "="*85)
    print(f"CREDILOGIC CHAMPION/CHALLENGER MATRIX (Evaluated at {THRESHOLD} Threshold)")
    print("="*85)
    
    # Formatting the DataFrame for terminal output
    # Formatting the DataFrame for terminal output
    format_dict = {
        "ROC-AUC": "{:.4f}".format,
        "Brier": "{:.4f}".format,
        "Accuracy": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "F1-Score": "{:.4f}".format
    }
    
    # Print the stylized table
    print(results_df.to_string(index=False, formatters=format_dict, justify='left'))
    print("="*85)
    
    # Executive Summary Calculation
    blend_auc = results_df[results_df['Model'] == 'CrediLogic 55/45 Blend']['ROC-AUC'].values[0]
    lr_auc = results_df[results_df['Model'] == 'Linear Baseline (LR)']['ROC-AUC'].values[0]
    
    print(f"\nExecutive Summary:")
    print(f"-> Non-Linear Lift: The 55/45 Blend outperforms the Logistic Baseline by +{(blend_auc - lr_auc):.4f} AUC.")
    print(f"-> Threshold Note: Accuracy is misleading in imbalanced risk data. Focus on the F1-Score to balance Recall (catching defaults) and Precision (funding good loans).")

if __name__ == "__main__":
    generate_comprehensive_report()