import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from data_loader import load_and_optimize_data

def engineer_application_features(df):
    print("Engineering Masterclass Application features...")
    
    # 1. Handle anomalous data
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # 2. Phil Method Ratios + Interaction Terms
    new_features = pd.DataFrame({
        'CREDIT_ANNUITY_RATIO': df['AMT_CREDIT'] / df['AMT_ANNUITY'],
        'CREDIT_GOODS_PRICE_RATIO': df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],
        'ANNUITY_INCOME_RATIO': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        'CREDIT_INCOME_RATIO': df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
        'DAYS_EMPLOYED_PERCENT': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
        'AGE_YEARS': df['DAYS_BIRTH'] / -365,
        'EMPLOYMENT_YEARS': df['DAYS_EMPLOYED'] / -365, # <--- ADDED THIS LINE
        'EXT_SOURCE_PROD': df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3'],
        'STRESS_INTERACTION': (df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']) * (1 - df['EXT_SOURCE_2'].fillna(0.5))
    }, index=df.index)
    
    df = pd.concat([df, new_features], axis=1)

    # --- SAFETY PATCH FOR XGBOOST ---
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Fill missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    print(f"Added new features. Total columns now: {df.shape[1]}")
    return df

def engineer_spatial_features(df):
    print("Calculating Magic Spatial Features (KNN)... this takes 2-3 minutes...")
    knn_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'ANNUITY_INCOME_RATIO']
    knn_df = df[knn_features].fillna(0)
    
    knn = NearestNeighbors(n_neighbors=500, n_jobs=-1)
    knn.fit(knn_df)
    _, indices = knn.kneighbors(knn_df)
    
    target_values = df['TARGET'].values
    
    # Avoid fragmentation by creating the Series first, then joining
    magic_feature = pd.Series([target_values[idx].mean() for idx in indices], 
                              index=df.index, name='NEIGHBORS_TARGET_MEAN_500')
    
    df = pd.concat([df, magic_feature], axis=1)
    return df

def engineer_installments(df):
    print("Engineering Masterclass Installment Time-Slices...")
    # Upcast float16 to float32 to avoid sorting error
    df['DAYS_INSTALMENT'] = df['DAYS_INSTALMENT'].astype('float32')
    df = df.dropna(subset=['DAYS_INSTALMENT'])  # Remove NaN before sorting
    df = df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])
    
    last_5 = df.groupby('SK_ID_CURR').tail(5).groupby('SK_ID_CURR')['AMT_PAYMENT'].mean().to_frame('INSTAL_LAST_5_PAY_MEAN')
    last_10 = df.groupby('SK_ID_CURR').tail(10).groupby('SK_ID_CURR')['AMT_PAYMENT'].mean().to_frame('INSTAL_LAST_10_PAY_MEAN')
    
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 0.00001)
    df['DPD'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
    
    inst_agg = df.groupby('SK_ID_CURR').agg({
        'PAYMENT_PERC': ['mean', 'min', 'std'],
        'DPD': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
    })
    inst_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
    
    return inst_agg.join(last_5, how='left').join(last_10, how='left').reset_index()

def engineer_bureau(df):
    print("Engineering Bureau history...")
    df['DEBT_CREDIT_RATIO'] = df['AMT_CREDIT_SUM_DEBT'] / (df['AMT_CREDIT_SUM'] + 0.00001)
    
    aggregations = {
        'DAYS_CREDIT': ['min', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],
        'DEBT_CREDIT_RATIO': ['mean', 'max']
    }
    bureau_agg = df.groupby('SK_ID_CURR').agg(aggregations)
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    return bureau_agg.reset_index()

def engineer_previous_applications(df):
    print("Engineering Previous Application history...")
    df['IS_REJECTED'] = (df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'IS_REJECTED': ['sum', 'mean'],
        'DAYS_DECISION': ['min', 'max'],
        'CNT_PAYMENT': ['mean', 'sum']
    }
    prev_agg = df.groupby('SK_ID_CURR').agg(aggregations)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    return prev_agg.reset_index()

def engineer_pos_cash(df):
    print("Engineering POS_CASH patterns...")
    df['MONTHS_BALANCE'] = df['MONTHS_BALANCE'].astype('float32')
    df['IS_LATE'] = (df['SK_DPD'] > 0).astype(int)
    
    pos_agg = df.groupby('SK_ID_CURR').agg({
        'SK_DPD': ['max', 'mean'],
        'IS_LATE': ['sum', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'last']
    })
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    return pos_agg.reset_index()

def engineer_credit_card(df):
    print("Engineering Credit Card balance patterns...")
    df['UTILIZATION'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 0.00001)
    
    cc_agg = df.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'UTILIZATION': ['mean', 'max'],
        'SK_DPD': ['max', 'sum']
    })
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    return cc_agg.reset_index()

def encode_and_merge(app_df, inst_df, bureau_df, prev_df, pos_df, cc_df):
    print("Merging all 6 datasets into Super-Dataset...")
    df_merged = app_df.merge(inst_df, on='SK_ID_CURR', how='left') \
                     .merge(bureau_df, on='SK_ID_CURR', how='left') \
                     .merge(prev_df, on='SK_ID_CURR', how='left') \
                     .merge(pos_df, on='SK_ID_CURR', how='left') \
                     .merge(cc_df, on='SK_ID_CURR', how='left')
    
    print("Encoding categories...")
    cat_cols = df_merged.select_dtypes(include=['category', 'object']).columns
    for col in cat_cols:
        df_merged[col] = df_merged[col].astype('category').cat.codes

    # --- FINAL SAFETY PATCH ---
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
        
    return df_merged


if __name__ == "__main__":
    # 1. Load data
    app = load_and_optimize_data('application_train.csv')
    
    # 2. Run standard engineering
    app = engineer_application_features(app)
    
    # 3. 🚨 REMOVE/COMMENT OUT THE SPATIAL FEATURE CALL 🚨
    # app = engineer_spatial_features(app)  <-- This line is the source of the 0.86 score
    
    # 4. Run the rest of the pipeline
    inst = engineer_installments(load_and_optimize_data('installments_payments.csv'))
    bur = engineer_bureau(load_and_optimize_data('bureau.csv'))
    prev = engineer_previous_applications(load_and_optimize_data('previous_application.csv'))
    pos = engineer_pos_cash(load_and_optimize_data('POS_CASH_balance.csv'))
    cc = engineer_credit_card(load_and_optimize_data('credit_card_balance.csv'))
    
    final_df = encode_and_merge(app, inst, bur, prev, pos, cc)
    
    # 5. Save the 'Clean' data
    final_df.to_pickle("data/processed/train_engineered.pkl")
    print("✅ Clean Data Generated: Leakage features removed.")