import pandas as pd
import numpy as np
import joblib

def generate_approval_path(applicant_index=0):
    print("🛣️ Initializing CrediLogic 'Path to Approval' Engine...")
    
    # 1. Load Production Assets
    lgbm = joblib.load('models/lgbm_model.pkl')
    xgb_m = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/golden_features.pkl')
    df = pd.read_pickle("data/processed/train_engineered.pkl")
    
    # 2. Select a Rejected Applicant (Simulated from training data for demo)
    # We will pick someone who is close to the threshold but rejected
    applicant = df.iloc[applicant_index].copy()
    applicant_df = pd.DataFrame([applicant])
    
    # Calculate Initial Score
    p_l = lgbm.predict_proba(applicant_df[features])[:, 1][0]
    p_x = xgb_m.predict_proba(applicant_df[features])[:, 1][0]
    initial_score = (p_l * 0.55) + (p_x * 0.45)
    
    # Assuming standard segment threshold for this example
    threshold = 0.110 
    
    print("\n" + "="*50)
    print(f"APPLICANT ID: {applicant['SK_ID_CURR'] if 'SK_ID_CURR' in applicant else 'SIM-101'}")
    print(f"Current Risk Score: {initial_score:.4f} | Threshold: {threshold}")
    
    if initial_score <= threshold:
        print("Status: ALREADY APPROVED. No action needed.")
        return
        
    print("Status: REJECTED. Calculating optimal path to approval...")
    print("-" * 50)
    
    # 3. The Simulation Engine
    # We will incrementally reduce the requested loan amount (and thus annuity)
    # until the model approves them.
    
    simulated_applicant = applicant.copy()
    original_credit = simulated_applicant['AMT_CREDIT']
    original_annuity = simulated_applicant['AMT_ANNUITY']
    income = simulated_applicant['AMT_INCOME_TOTAL']
    
    step_reduction = 0.05 # Reduce loan by 5% each iteration
    max_iterations = 10
    approved = False
    
    for i in range(1, max_iterations + 1):
        # Apply reductions
        sim_credit = original_credit * (1 - (step_reduction * i))
        sim_annuity = original_annuity * (1 - (step_reduction * i))
        
        # Update actionable features
        simulated_applicant['AMT_CREDIT'] = sim_credit
        simulated_applicant['AMT_ANNUITY'] = sim_annuity
        simulated_applicant['CREDIT_INCOME_RATIO'] = sim_credit / income
        simulated_applicant['ANNUITY_INCOME_RATIO'] = sim_annuity / income
        
        # Recalculate Score
        sim_df = pd.DataFrame([simulated_applicant])
        sim_p_l = lgbm.predict_proba(sim_df[features])[:, 1][0]
        sim_p_x = xgb_m.predict_proba(sim_df[features])[:, 1][0]
        new_score = (sim_p_l * 0.55) + (sim_p_x * 0.45)
        
        if new_score <= threshold:
            approved = True
            print(f"✅ PATH FOUND (Iteration {i}):")
            print(f"   Reduce Loan Amount by:  ${(original_credit - sim_credit):,.2f}")
            print(f"   New Max Loan Amount:    ${sim_credit:,.2f}")
            print(f"   New Monthly Payment:    ${sim_annuity:,.2f}")
            print(f"   New Risk Score:         {new_score:.4f}")
            break
            
    if not approved:
        print("❌ NO VIABLE PATH. Applicant risk is too high even with maximum loan reductions.")
    
    print("="*50)

if __name__ == "__main__":
    # You can change this index to test different rejected applicants
    generate_approval_path(applicant_index=5)